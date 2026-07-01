//===- IdiomRecognizer.cpp - recognizing and raising idioms to CIR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for recognizing idioms (such as uses of functions
// and types to the C/C++ standard library) and replacing them with Clang IR
// operators for later optimization.
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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

#include <utility>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_IDIOMRECOGNIZER
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// The raised operation must satisfy its own type constraints, so a call whose
// types would not verify is left alone rather than turned into an invalid op.
bool operandTypesMatch(StdFindOp *, CallOp call) {
  mlir::Type iterTy = call.getOperand(0).getType();
  return iterTy == call.getOperand(1).getType() &&
         iterTy == call->getResult(0).getType();
}

bool operandTypesMatch(StrLenOp *, CallOp call) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(call.getOperand(0).getType());
  auto charTy =
      ptrTy ? mlir::dyn_cast<cir::IntType>(ptrTy.getPointee()) : nullptr;
  return charTy && charTy.getWidth() == 8 && charTy.isSigned() &&
         !charTy.getIsBitInt() &&
         cir::isFundamentalIntType(call->getResult(0).getType());
}

// Recognizes a direct cir.call to the standard library function represented
// by `TargetOp` and raises it to that operation. C++ entities are matched
// semantically through the call's AST attribute; C library functions are
// matched by callee name.
template <typename TargetOp, bool IsCXX = true> class StdRecognizer {
  template <size_t... Indices>
  static TargetOp buildCall(cir::CIRBaseBuilderTy &builder, CallOp call,
                            std::index_sequence<Indices...>) {
    return TargetOp::create(builder, call.getLoc(),
                            call->getResult(0).getType(),
                            call.getOperand(Indices)..., call.getCalleeAttr());
  }

public:
  static bool raise(CallOp call, mlir::MLIRContext &context) {
    constexpr unsigned numArgs = TargetOp::getNumArgs();
    if (!call.getCallee() || call.getNumOperands() != numArgs ||
        call->getNumResults() != 1 ||
        !operandTypesMatch(static_cast<TargetOp *>(nullptr), call))
      return false;

    llvm::StringRef funcName = TargetOp::getFunctionName();
    if constexpr (IsCXX) {
      cir::ASTCallExprInterface astAttr = call.getAstAttr();
      if (!astAttr || !astAttr.isStdFunctionCall(funcName))
        return false;
    } else {
      if (*call.getCallee() != funcName)
        return false;
    }

    cir::CIRBaseBuilderTy builder(context);
    builder.setInsertionPointAfter(call.getOperation());
    TargetOp op = buildCall(builder, call, std::make_index_sequence<numArgs>());
    // Preserve call attributes across the raise so that lowering the op back
    // to a call does not weaken the original IR.
    for (llvm::StringRef name :
         {"nothrow", "side_effect", "arg_attrs", "res_attrs"})
      if (mlir::Attribute attr = call->getAttr(name))
        op->setAttr(name, attr);
    call.replaceAllUsesWith(op);
    call.erase();
    return true;
  }
};

struct IdiomRecognizerPass
    : public impl::IdiomRecognizerBase<IdiomRecognizerPass> {
  IdiomRecognizerPass() = default;

  void runOnOperation() override;

  void recognizeStandardLibraryCall(CallOp call);

  clang::ASTContext *astCtx = nullptr;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

void IdiomRecognizerPass::recognizeStandardLibraryCall(CallOp call) {
  if (StdRecognizer<StdFindOp>::raise(call, getContext()))
    return;
  StdRecognizer<StrLenOp, /*IsCXX=*/false>::raise(call, getContext());
}

void IdiomRecognizerPass::runOnOperation() {
  // The AST context will be used to provide additional information such as
  // namespaces and template parameter lists that are lost after lowering to
  // CIR. This information is necessary to recognize many idioms, such as calls
  // to standard library functions.

  // For now, the AST will be required to allow for faster prototyping and
  // exploring of new optimizations. In the future, it may be preferable to
  // make it optional to reduce memory pressure and allow this pass to run
  // on standalone CIR assembly (Possibly generated from non-Clang front ends).

  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  theModule = getOperation();

  // Process call operations
  theModule->walk([&](CallOp callOp) {
    // Skip indirect calls.
    std::optional<llvm::StringRef> callee = callOp.getCallee();
    if (!callee)
      return;

    recognizeStandardLibraryCall(callOp);
  });
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}

std::unique_ptr<Pass>
mlir::createIdiomRecognizerPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<IdiomRecognizerPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
