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
  if (!elemTy)
    return;
  // A wider element rewrites to wmemchr, which only the plain find form
  // licenses since the predicate and ranges proofs are byte facts. The
  // recorded wchar_t width check below rejects every width that is not
  // the target's wide character.
  constexpr bool plainFind = std::is_same_v<OpT, StdFindOp>;
  const bool wide = elemTy.getWidth() != 8;
  if (wide && !plainFind)
    return;
  const llvm::StringRef libcallName = wide ? "wmemchr" : "memchr";
  // The sought element arrives behind the pattern reference for find. A
  // predicate either holds it in its single capture, by value or through a
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
  if (isNoBuiltin(findOp, libcallName) ||
      (enclosing && noBuiltinListDisables(enclosing, libcallName)))
    return;

  // Only the CIRGen facts license the rewrite, since an enum or atomic element
  // also lowers to a byte-wide integer and a closure of the right shape can
  // compute anything.
  if constexpr (predOp) {
    if (!predicateValue && !findOp->template getAttrOfType<mlir::UnitAttr>(
                               cir::CIRDialect::getByteEqPredAttrName()))
      return;
  } else if (!findOp->template getAttrOfType<mlir::UnitAttr>(
                 wide ? cir::CIRDialect::getWideCharParamsAttrName()
                      : cir::CIRDialect::getNarrowCharParamsAttrName())) {
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
  llvm::Triple triple(tripleAttr.getValue().str());
  llvm::TargetLibraryInfoImpl tliImpl(triple);
  if (!llvm::TargetLibraryInfo(tliImpl).has(wide ? llvm::LibFunc_wmemchr
                                                 : llvm::LibFunc_memchr))
    return;

  // The introduced operands take the widths the module records, which is
  // what the introduced operation's verifier checks them against: int and
  // size_t for memchr, wchar_t and size_t for wmemchr. An element that is
  // not the target's wide character has no libcall and declines here.
  std::optional<unsigned> patternWidth = cir::getRecordedIntegerWidth(
      moduleOp, wide ? cir::CIRDialect::getWCharTypeWidthAttrName()
                     : cir::CIRDialect::getIntTypeWidthAttrName());
  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!patternWidth || !sizeWidth)
    return;
  // A program built with a nonstandard wchar_t width still links the C
  // library selected by the triple, and that library's wmemchr walks the
  // default width.
  if (wide && *patternWidth != triple.getDefaultWCharSize() * 8)
    return;
  // Hand written IR can claim the wide character marker while using an
  // element whose width disagrees with the recorded width.
  if (wide && *patternWidth != elemTy.getWidth())
    return;

  // A libc that implements the search with std::find would call itself
  // forever.
  if (enclosing && enclosing.getName() == libcallName)
    return;

  CIRBaseBuilderTy builder(*findOp.getContext());

  if (mlir::Operation *existing = symbolTables.lookupSymbolIn(
          moduleOp, mlir::StringAttr::get(findOp.getContext(), libcallName))) {
    auto existingFn = mlir::dyn_cast<cir::FuncOp>(existing);
    if (!existingFn)
      return;
    auto libcallTy =
        wide ? cir::FuncType::get({mlir::Type(iterTy), mlir::Type(elemTy),
                                   builder.getUIntNTy(*sizeWidth)},
                                  iterTy)
             : cir::FuncType::get({builder.getVoidPtrTy(),
                                   builder.getSIntNTy(*patternWidth),
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
            mlir::Value sought;
            if constexpr (predOp) {
              if (predicateValue) {
                sought = builder.getConstant(loc, predicateValue);
              } else {
                sought = cir::ExtractMemberOp::create(builder, loc,
                                                      findOp.getPred(), 0);
                if (captureByRef)
                  sought = builder.createLoad(loc, sought);
              }
            } else {
              sought = builder.createLoad(loc, findOp.getPattern());
            }
            mlir::Value len = cir::PtrDiffOp::create(
                builder, loc, builder.getUIntNTy(*sizeWidth), last, first);
            mlir::Value res;
            if (wide) {
              // wmemchr takes and returns the wide type itself, so nothing
              // is widened or cast at the boundary and len counts wide
              // characters, which is what ptr_diff yields.
              res = cir::WMemChrOp::create(builder, loc, first, sought, len);
            } else {
              mlir::Value src =
                  builder.createBitcast(loc, first, builder.getVoidPtrTy());
              mlir::Value pattern = builder.createIntCast(
                  sought, builder.getSIntNTy(*patternWidth));
              res = cir::MemChrOp::create(builder, loc, src, pattern, len);
              res = builder.createBitcast(loc, res, iterTy);
            }
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

// Rewrites the four iterator form of std::search over licensed bytes.
static void rewriteSearchToMemmem(StdSearchOp searchOp,
                                  mlir::SymbolTableCollection &symbolTables) {
  cir::RecordType haystackWrapperTy;
  cir::RecordType needleWrapperTy;
  auto unwrapIterator = [](mlir::Type ty, cir::RecordType &wrapperTy) {
    auto iterTy = mlir::dyn_cast<cir::PointerType>(ty);
    wrapperTy = mlir::dyn_cast<cir::RecordType>(ty);
    if (!wrapperTy)
      return iterTy;
    if (wrapperTy.isUnion() || !wrapperTy.isComplete() ||
        wrapperTy.getMembers().size() != 1)
      return cir::PointerType();
    return mlir::dyn_cast<cir::PointerType>(wrapperTy.getMembers()[0]);
  };

  cir::PointerType haystackIterTy =
      unwrapIterator(searchOp.getFirst1().getType(), haystackWrapperTy);
  cir::PointerType needleIterTy =
      unwrapIterator(searchOp.getFirst2().getType(), needleWrapperTy);
  if (!haystackIterTy || !needleIterTy ||
      static_cast<bool>(haystackWrapperTy) !=
          static_cast<bool>(needleWrapperTy) ||
      haystackIterTy.getAddrSpace() || needleIterTy.getAddrSpace())
    return;

  auto haystackElemTy =
      mlir::dyn_cast<cir::IntType>(haystackIterTy.getPointee());
  auto needleElemTy = mlir::dyn_cast<cir::IntType>(needleIterTy.getPointee());
  if (!haystackElemTy || haystackElemTy.getWidth() != 8 ||
      needleElemTy != haystackElemTy)
    return;

  auto enclosing = searchOp->getParentOfType<cir::FuncOp>();
  auto enclosingGlobal = searchOp->getParentOfType<cir::GlobalOp>();
  if (!enclosing && !enclosingGlobal)
    return;

  constexpr llvm::StringLiteral libcallName = "memmem";

  if (isNoBuiltin(searchOp, libcallName) ||
      (enclosing && noBuiltinListDisables(enclosing, libcallName)))
    return;

  if (!searchOp->getAttrOfType<mlir::UnitAttr>(
          cir::CIRDialect::getNarrowCharParamsAttrName()))
    return;

  auto moduleOp = searchOp->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return;
  auto tripleAttr = moduleOp->getAttrOfType<mlir::StringAttr>(
      cir::CIRDialect::getTripleAttrName());
  if (!tripleAttr)
    return;
  llvm::TargetLibraryInfoImpl tliImpl(
      llvm::Triple(tripleAttr.getValue().str()));
  if (!llvm::TargetLibraryInfo(tliImpl).has(llvm::LibFunc_memmem))
    return;

  std::optional<unsigned> sizeWidth = cir::getRecordedIntegerWidth(
      moduleOp, cir::CIRDialect::getSizeTypeWidthAttrName());
  if (!sizeWidth)
    return;

  if (enclosing && enclosing.getName() == libcallName)
    return;

  CIRBaseBuilderTy builder(*searchOp.getContext());
  auto voidPtrTy = builder.getVoidPtrTy();
  auto sizeTy = builder.getUIntNTy(*sizeWidth);
  if (mlir::Operation *existing = symbolTables.lookupSymbolIn(
          moduleOp,
          mlir::StringAttr::get(searchOp.getContext(), libcallName))) {
    auto existingFn = mlir::dyn_cast<cir::FuncOp>(existing);
    if (!existingFn)
      return;
    auto libcallTy =
        cir::FuncType::get({voidPtrTy, sizeTy, voidPtrTy, sizeTy}, voidPtrTy);
    if (existingFn.getAliasee() || existingFn.getFunctionType() != libcallTy ||
        existingFn.getCallingConv() != cir::CallingConv::C)
      return;
  }

  mlir::Location loc = searchOp.getLoc();
  builder.setInsertionPointAfter(searchOp);
  mlir::Value first1 = searchOp.getFirst1();
  mlir::Value last1 = searchOp.getLast1();
  mlir::Value first2 = searchOp.getFirst2();
  mlir::Value last2 = searchOp.getLast2();
  if (haystackWrapperTy) {
    first1 = cir::ExtractMemberOp::create(builder, loc, first1, 0);
    last1 = cir::ExtractMemberOp::create(builder, loc, last1, 0);
    first2 = cir::ExtractMemberOp::create(builder, loc, first2, 0);
    last2 = cir::ExtractMemberOp::create(builder, loc, last2, 0);
  }

  // The C++ empty needle rule returns first1.
  mlir::Value needleIsEmpty =
      builder.createCompare(loc, cir::CmpOpKind::eq, first2, last2);
  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, needleIsEmpty,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, first1);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            // These pointer differences are byte counts because both element
            // types have width eight.
            mlir::Value haystackLen =
                cir::PtrDiffOp::create(builder, loc, sizeTy, last1, first1);
            mlir::Value needleLen =
                cir::PtrDiffOp::create(builder, loc, sizeTy, last2, first2);
            // The C++ short haystack rule returns last1. Together these guards
            // ensure memmem receives a nonempty needle and a haystack at least
            // as long, so every pointer and length passed to C is valid.
            mlir::Value needleIsLonger = builder.createCompare(
                loc, cir::CmpOpKind::gt, needleLen, haystackLen);
            mlir::Value nonEmptyResult =
                cir::TernaryOp::create(
                    builder, loc, needleIsLonger,
                    [&](mlir::OpBuilder &, mlir::Location) {
                      builder.createYield(loc, last1);
                    },
                    [&](mlir::OpBuilder &, mlir::Location) {
                      mlir::Value haystack = builder.createBitcast(
                          loc, first1, builder.getVoidPtrTy());
                      mlir::Value needle = builder.createBitcast(
                          loc, first2, builder.getVoidPtrTy());
                      mlir::Value found =
                          cir::MemMemOp::create(builder, loc, haystack,
                                                haystackLen, needle, needleLen);
                      found = builder.createBitcast(loc, found, haystackIterTy);
                      builder.createYield(
                          loc, builder.createSelect(
                                   loc, builder.createPtrIsNull(found), last1,
                                   found));
                    })
                    .getResult();
            builder.createYield(loc, nonEmptyResult);
          })
          .getResult();

  if (haystackWrapperTy) {
    result = cir::InsertMemberOp::create(builder, loc, searchOp.getFirst1(),
                                         /*index=*/0, result);
  }
  searchOp.getResult().replaceAllUsesWith(result);
  searchOp.erase();
}

void LibOptPass::runOnOperation() {
  mlir::SymbolTableCollection symbolTables;
  getOperation()->walk([&](mlir::Operation *op) {
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case<StdFindOp, StdFindIfOp, StdFindIfNotOp, StdRangesFindOp,
              StdRangesFindRangeOp>(
            [&](auto find) { rewriteFindLikeToMemchr(find, symbolTables); })
        .Case<StdSearchOp>(
            [&](auto search) { rewriteSearchToMemmem(search, symbolTables); });
  });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
