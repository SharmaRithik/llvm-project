//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers CIR operations that stand for a C library call to cir.call.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LIBCALLLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct LibCallLoweringPass
    : public impl::LibCallLoweringBase<LibCallLoweringPass> {
  void runOnOperation() override;
};

} // namespace

// Replaces the op with a call to memchr typed from the operands, which
// already carry the target's int and size_t widths. The call and a
// declaration created here get noundef and nothrow, as classic gives the
// builtin's call by default, plus the no builtin list the op recorded. The
// call is also marked nobuiltin when that list disables memchr, as CIRGen
// marks a call the user writes.
static mlir::LogicalResult lowerMemChr(cir::MemChrOp op,
                                       mlir::ModuleOp module) {
  cir::CIRBaseBuilderTy builder(*op.getContext());
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();

  auto fnType = cir::FuncType::get(
      {op.getSrc().getType(), op.getPattern().getType(), op.getLen().getType()},
      op.getType());
  mlir::NamedAttrList noundefAttrs;
  noundefAttrs.set("llvm.noundef", builder.getUnitAttr());
  llvm::SmallVector<mlir::NamedAttrList, 3> argAttrs(op->getNumOperands(),
                                                     noundefAttrs);
  mlir::ArrayAttr nbFuncs = op.getNobuiltinsAttr();

  // Only a function or a variable can be called through.
  mlir::Operation *existing = module.lookupSymbol("memchr");
  auto libFunc = mlir::dyn_cast_if_present<cir::FuncOp>(existing);
  auto libVar = mlir::dyn_cast_if_present<cir::GlobalOp>(existing);
  if (existing && !libFunc && !libVar)
    return op.emitError("memchr names neither a function nor a variable");

  if (!existing) {
    libFunc = cir::getOrCreateRuntimeFuncDecl(module, loc, "memchr", fnType);
    for (auto [index, attrs] : llvm::enumerate(argAttrs))
      libFunc.setArgAttrs(index, attrs);
    libFunc->setAttr(cir::CIRDialect::getNoThrowAttrName(),
                     builder.getUnitAttr());
    if (nbFuncs)
      libFunc->setAttr(cir::CIRDialect::getNoBuiltinsAttrName(), nbFuncs);
  }

  mlir::ValueRange operands = op->getOperands();
  cir::CallOp call;
  if (libFunc && libFunc.getFunctionType() == fnType) {
    call = builder.createCallOp(loc, libFunc, operands, /*attrs=*/{}, argAttrs);
  } else {
    // Another type, or a variable. Call through the address cast to the call
    // type, as classic calls the existing symbol.
    mlir::Type symType =
        libFunc ? mlir::Type(libFunc.getFunctionType()) : libVar.getSymType();
    mlir::Value addr = cir::GetGlobalOp::create(
        builder, loc, cir::PointerType::get(symType), "memchr");
    mlir::Value target =
        builder.createBitcast(addr, cir::PointerType::get(fnType));
    call = builder.createIndirectCallOp(loc, target, fnType, operands,
                                        /*attrs=*/{}, argAttrs);
  }
  call.setNothrowAttr(builder.getUnitAttr());
  if (nbFuncs) {
    call->setAttr(cir::CIRDialect::getNoBuiltinsAttrName(), nbFuncs);
    if (cir::noBuiltinListDisables(nbFuncs, "memchr"))
      call->setAttr(cir::CIRDialect::getNoBuiltinAttrName(),
                    builder.getUnitAttr());
  }
  op.replaceAllUsesWith(call.getResult());
  op.erase();
  return mlir::success();
}

void LibCallLoweringPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  llvm::SmallVector<cir::MemChrOp> ops;
  module.walk([&](cir::MemChrOp op) { ops.push_back(op); });
  for (cir::MemChrOp op : ops) {
    if (failed(lowerMemChr(op, module)))
      return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createLibCallLoweringPass() {
  return std::make_unique<LibCallLoweringPass>();
}
