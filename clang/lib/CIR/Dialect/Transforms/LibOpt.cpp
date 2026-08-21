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

static void rewriteStdFindToMemchr(StdFindOp findOp,
                                   mlir::SymbolTableCollection &symbolTables) {
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
  auto patternPtrTy =
      mlir::dyn_cast<cir::PointerType>(findOp.getPattern().getType());
  if (!patternPtrTy || patternPtrTy.getPointee() != elemTy)
    return;

  // LibOpt runs before LoweringPrepare, so a global initializer is still a
  // cir.global here. Anything else is not a shape CIRGen produces.
  auto enclosing = findOp->getParentOfType<cir::FuncOp>();
  auto enclosingGlobal = findOp->getParentOfType<cir::GlobalOp>();
  if (!enclosing && !enclosingGlobal)
    return;

  // No builtin state rides on the raised call and on the enclosing function.
  // A global initializer has no function to carry the list.
  if (isNoBuiltin(findOp, "memchr") ||
      (enclosing && noBuiltinListDisables(enclosing, "memchr")))
    return;

  // Only the unit attribute CIRGen recorded licenses the rewrite, since an
  // enum or atomic element also lowers to a byte-wide integer.
  if (!findOp->getAttrOfType<mlir::UnitAttr>(
          cir::CIRDialect::getNarrowCharParamsAttrName()))
    return;

  auto moduleOp = findOp->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return;

  // The rewrite introduces a libcall the program never named, so the target
  // has to provide it. TargetLibraryInfo is the availability oracle LLVM
  // transforms consult before creating a libcall, and it answers from the
  // triple. The ABI attributes of the introduced call stay with the general
  // call ABI work.
  auto tripleAttr = moduleOp->getAttrOfType<mlir::StringAttr>(
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
  mlir::Value first = findOp.getFirst();
  mlir::Value last = findOp.getLast();
  builder.setInsertionPointAfter(findOp);
  if (wrapperTy) {
    first = cir::ExtractMemberOp::create(builder, loc, first, 0);
    last = cir::ExtractMemberOp::create(builder, loc, last, 0);
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
            mlir::Value pattern = builder.createIntCast(
                builder.createLoad(loc, findOp.getPattern()),
                builder.getSIntNTy(*intWidth));
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
  if (wrapperTy)
    result = cir::InsertMemberOp::create(builder, loc, findOp.getFirst(),
                                         /*index=*/0, result);
  findOp.getResult().replaceAllUsesWith(result);
  findOp.erase();
}

void LibOptPass::runOnOperation() {
  mlir::SymbolTableCollection symbolTables;
  getOperation()->walk(
      [&](StdFindOp findOp) { rewriteStdFindToMemchr(findOp, symbolTables); });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
