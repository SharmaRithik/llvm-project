//===- LibOpt.cpp - Optimize CIR raised C/C++ library idioms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes C/C++ standard library idioms in Clang IR.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LIBOPT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct LibOptPass : public impl::LibOptBase<LibOptPass> {
  LibOptPass() = default;
  mlir::LogicalResult
  initializeOptions(llvm::StringRef options,
                    llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override;
  void runOnOperation() override;

  // Raw libopt option string forwarded by the frontend. This will later control
  // which optimizations the pass enables.
  std::string optimizationOptions;
};
} // namespace

mlir::LogicalResult LibOptPass::initializeOptions(
    llvm::StringRef options,
    llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>) {
  optimizationOptions = options.str();
  // TODO(cir): Parse options to select the active transformations for the
  // pass.
  return mlir::success();
}

// Rewrites cir.std.find, cir.std.ranges.find, cir.std.ranges.find_range,
// and the byte-equality-predicate forms of cir.std.find_if and
// cir.std.find_if_not. All five search for the first equal byte, so they
// share everything except where the byte and the bounds come from: the
// by-reference pattern for the find forms, a closure capture or recorded
// constant for the predicate forms, and for the whole range form the
// bounds are loaded through the member paths its record's standard
// library identity carries. The cpo and proj operands of the ranges forms
// are carried only for lowering back and play no part here.
template <typename OpT>
static void rewriteFindLikeToMemchr(OpT findOp,
                                    mlir::SymbolTableCollection &symbolTables) {
  constexpr bool predOp =
      std::is_same_v<OpT, StdFindIfOp> || std::is_same_v<OpT, StdFindIfNotOp>;
  constexpr bool rangeOp = std::is_same_v<OpT, StdRangesFindRangeOp>;
  // A std contiguous iterator wraps the pointer as its only member.  The
  // narrow character fact CIRGen recorded is what licenses reading that
  // member as the address, so the rewrite runs on the wrapped pointer and
  // rebuilds the record around its result.
  mlir::Type resultTy = findOp.getResult().getType();
  auto wrapperTy = mlir::dyn_cast<cir::RecordType>(resultTy);
  auto iterTy = mlir::dyn_cast<cir::PointerType>(resultTy);
  if (wrapperTy) {
    if (wrapperTy.isUnion() || !wrapperTy.isComplete() ||
        wrapperTy.getMembers().size() != 1)
      return;
    iterTy = mlir::dyn_cast<cir::PointerType>(wrapperTy.getMembers()[0]);
  }
  if (!iterTy || iterTy.getAddrSpace())
    return;
  auto elemTy = mlir::dyn_cast<cir::IntType>(iterTy.getPointee());
  if (!elemTy || elemTy.getWidth() != 8)
    return;
  // The byte to search for arrives behind the by-reference pattern for find.
  // A predicate either holds it in its single capture, by value or through a
  // reference, or carries it in the typed value attribute.
  bool captureByRef = false;
  cir::IntAttr predicateValue;
  if constexpr (predOp) {
    predicateValue = findOp->template getAttrOfType<cir::IntAttr>(
        cir::CIRDialect::getByteEqPredValueAttrName());
    if (predicateValue) {
      if (predicateValue.getType() != elemTy)
        return;
    } else {
      auto closureTy =
          mlir::dyn_cast<cir::RecordType>(findOp.getPred().getType());
      if (!closureTy || closureTy.isUnion() || !closureTy.isComplete() ||
          closureTy.getMembers().size() != 1)
        return;
      mlir::Type capTy = closureTy.getMembers()[0];
      if (auto capPtrTy = mlir::dyn_cast<cir::PointerType>(capTy)) {
        if (capPtrTy.getAddrSpace() || capPtrTy.getPointee() != elemTy)
          return;
        captureByRef = true;
      } else if (capTy != elemTy) {
        return;
      }
    }
  } else {
    auto patternPtrTy =
        mlir::dyn_cast<cir::PointerType>(findOp.getPattern().getType());
    if (!patternPtrTy || patternPtrTy.getPointee() != elemTy)
      return;
  }

  // The whole range form owns no bounds operands. Its record's standard
  // library identity names the element and the member paths to the two
  // pointers bounding the storage, and the rewrite re-proves that every
  // path lands on a pointer to the searched element.
  cir::StructType rangeRecTy;
  cir::StdTypeInfoAttr rangeInfo;
  if constexpr (rangeOp) {
    auto rangePtrTy =
        mlir::dyn_cast<cir::PointerType>(findOp.getRange().getType());
    if (!rangePtrTy || rangePtrTy.getAddrSpace())
      return;
    rangeRecTy = mlir::dyn_cast<cir::StructType>(rangePtrTy.getPointee());
    if (rangeRecTy)
      rangeInfo = rangeRecTy.getStdTypeInfo();
    if (!rangeInfo || rangeInfo.getKind() != cir::StdTypeKind::StdVector ||
        rangeInfo.getElement() != mlir::Type(elemTy))
      return;
    auto pathLandsOnIter = [&](llvm::ArrayRef<uint32_t> path) {
      return cir::StdTypeInfoAttr::resolvePath(rangeRecTy, path) ==
             mlir::Type(iterTy);
    };
    if (!pathLandsOnIter(rangeInfo.getBeginPath()) ||
        !pathLandsOnIter(rangeInfo.getEndPath()))
      return;
  }

  // LibOpt runs before LoweringPrepare, so a global initializer is still a
  // cir.global here. Anything else is not a shape CIRGen produces.
  auto enclosing = findOp->template getParentOfType<cir::FuncOp>();
  auto enclosingGlobal = findOp->template getParentOfType<cir::GlobalOp>();
  if (!enclosing && !enclosingGlobal)
    return;

  // No builtin state rides on the raised call and on the enclosing function.
  // A global initializer has no function to carry the list.
  if (isNoBuiltin(findOp, "memchr") ||
      (enclosing && noBuiltinListDisables(enclosing, "memchr")))
    return;

  // Only the CIRGen facts license the rewrite, since an enum or atomic element
  // also lowers to a byte-wide integer and a closure of the right shape can
  // compute anything.
  if constexpr (predOp) {
    if (!predicateValue && !findOp->template getAttrOfType<mlir::UnitAttr>(
                               cir::CIRDialect::getByteEqPredAttrName()))
      return;
  } else if (!findOp->template getAttrOfType<mlir::UnitAttr>(
                 cir::CIRDialect::getNarrowCharParamsAttrName())) {
    return;
  }

  auto moduleOp = findOp->template getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return;

  // The rewrite introduces a libcall the program never named, so the target
  // has to provide it. TargetLibraryInfo is the availability oracle LLVM
  // transforms consult before creating a libcall, and it answers from the
  // triple. The ABI attributes of the introduced call stay with the general
  // call ABI work.
  auto tripleAttr = moduleOp->template getAttrOfType<mlir::StringAttr>(
      cir::CIRDialect::getTripleAttrName());
  if (!tripleAttr)
    return;
  llvm::TargetLibraryInfoImpl tliImpl(
      llvm::Triple(tripleAttr.getValue().str()));
  if (!llvm::TargetLibraryInfo(tliImpl).has(llvm::LibFunc_memchr))
    return;

  // The introduced operands take the int and size_t widths the module
  // records, which is what the cir.libc.memchr verifier checks them against.
  std::optional<unsigned> intWidth = cir::getRecordedIntegerWidth(
      moduleOp, cir::CIRDialect::getIntTypeWidthAttrName());
  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!intWidth || !sizeWidth)
    return;

  // A libc that implements memchr with std::find would call itself forever.
  if (enclosing && enclosing.getName() == "memchr")
    return;

  CIRBaseBuilderTy builder(*findOp.getContext());

  if (mlir::Operation *existing = symbolTables.lookupSymbolIn(
          moduleOp, mlir::StringAttr::get(findOp.getContext(), "memchr"))) {
    auto existingFn = mlir::dyn_cast<cir::FuncOp>(existing);
    if (!existingFn)
      return;
    auto libcallTy = cir::FuncType::get({builder.getVoidPtrTy(),
                                         builder.getSIntNTy(*intWidth),
                                         builder.getUIntNTy(*sizeWidth)},
                                        builder.getVoidPtrTy());
    // An alias forwards to some other symbol, so a matching prototype says
    // nothing about what the call would reach.
    if (existingFn.getAliasee() || existingFn.getFunctionType() != libcallTy ||
        existingFn.getCallingConv() != cir::CallingConv::C)
      return;
  }

  mlir::Location loc = findOp.getLoc();
  builder.setInsertionPointAfter(findOp);
  mlir::Value first, last;
  if constexpr (rangeOp) {
    // The same walk that licensed the paths supplies each step's member
    // type here, so the emitted accesses cannot drift from the proof.
    auto loadBound = [&](llvm::ArrayRef<uint32_t> path) {
      llvm::SmallVector<mlir::Type, 4> steps;
      cir::StdTypeInfoAttr::resolvePath(rangeRecTy, path, &steps);
      mlir::Value addr = findOp.getRange();
      for (auto [idx, fieldTy] : llvm::zip_equal(path, steps))
        addr = cir::GetMemberOp::create(builder, loc,
                                        cir::PointerType::get(fieldTy), addr,
                                        /*name=*/"", /*index=*/idx);
      return builder.createLoad(loc, addr);
    };
    first = loadBound(rangeInfo.getBeginPath());
    last = loadBound(rangeInfo.getEndPath());
  } else {
    first = findOp.getFirst();
    last = findOp.getLast();
    if (wrapperTy) {
      first = cir::ExtractMemberOp::create(builder, loc, first, 0);
      last = cir::ExtractMemberOp::create(builder, loc, last, 0);
    }
  }

  // An empty std::find range is allowed to be a pair of null pointers, while
  // C requires the memchr pointer to be valid even when the length is zero.
  mlir::Value isEmpty =
      builder.createCompare(loc, cir::CmpOpKind::eq, first, last);
  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, isEmpty,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, last);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value src =
                builder.createBitcast(loc, first, builder.getVoidPtrTy());
            mlir::Value byte;
            if constexpr (predOp) {
              if (predicateValue) {
                byte = builder.getConstant(loc, predicateValue);
              } else {
                byte = cir::ExtractMemberOp::create(builder, loc,
                                                    findOp.getPred(), 0);
                if (captureByRef)
                  byte = builder.createLoad(loc, byte);
              }
            } else {
              byte = builder.createLoad(loc, findOp.getPattern());
            }
            mlir::Value pattern =
                builder.createIntCast(byte, builder.getSIntNTy(*intWidth));
            mlir::Value len = cir::PtrDiffOp::create(
                builder, loc, builder.getUIntNTy(*sizeWidth), last, first);
            mlir::Value res =
                cir::MemChrOp::create(builder, loc, src, pattern, len);
            res = builder.createBitcast(loc, res, iterTy);
            builder.createYield(
                loc, builder.createSelect(loc, builder.createPtrIsNull(res),
                                          last, res));
          })
          .getResult();
  if (wrapperTy) {
    // The whole range form has no incoming wrapper value to rebuild
    // around, so the result record starts from undef and the insert
    // defines its only member.
    mlir::Value seed;
    if constexpr (rangeOp)
      seed = builder.getConstant(loc, cir::UndefAttr::get(wrapperTy));
    else
      seed = findOp.getFirst();
    result =
        cir::InsertMemberOp::create(builder, loc, seed, /*index=*/0, result);
  }
  findOp.getResult().replaceAllUsesWith(result);
  findOp.erase();
}

void LibOptPass::runOnOperation() {
  mlir::SymbolTableCollection symbolTables;
  getOperation()->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<StdFindOp, StdFindIfOp, StdFindIfNotOp, StdRangesFindOp,
              StdRangesFindRangeOp>(
            [&](auto find) { rewriteFindLikeToMemchr(find, symbolTables); });
  });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
