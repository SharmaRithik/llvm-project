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

// The raised operation requires matching operand types. The searched value
// arrives by reference and must share the iterator type.
bool operandTypesMatch(StdFindOp *, CallOp call) {
  mlir::Type iterTy = call.getOperand(0).getType();
  return iterTy == call.getOperand(1).getType() &&
         iterTy == call.getOperand(2).getType() &&
         iterTy == call->getResult(0).getType();
}

// Raises a direct cir.call to `TargetOp` when the callee carries the
// matching identity tag.
template <typename TargetOp> class StdRecognizer {
  template <size_t... Indices>
  static TargetOp buildCall(cir::CIRBaseBuilderTy &builder, CallOp call,
                            std::index_sequence<Indices...>) {
    return TargetOp::create(builder, call.getLoc(),
                            call->getResult(0).getType(),
                            call.getOperand(Indices)..., call.getCalleeAttr());
  }

public:
  static bool raise(CallOp call, mlir::MLIRContext &context,
                    mlir::SymbolTableCollection &symbolTables) {
    // A musttail call must stay a call, so it is never raised.
    constexpr unsigned numArgs = TargetOp::getNumArgs();
    if (!call.getCallee() || call.getNumOperands() != numArgs ||
        call->getNumResults() != 1 || call.getMusttail() ||
        !operandTypesMatch(static_cast<TargetOp *>(nullptr), call))
      return false;

    // CIRGen tags a declaration by name, freeness, and std membership, so
    // members, static members, and operators never match. The shape of the
    // call is checked here, so a variadic callee never matches.
    cir::FuncOp callee = call.resolveCalleeInTable(&symbolTables);
    if (!callee || callee.getFunctionType().isVarArg())
      return false;
    auto funcIdentity = mlir::dyn_cast_if_present<cir::FuncIdentityAttr>(
        callee.getFuncInfoAttr());
    if (!funcIdentity || funcIdentity.getKind() != TargetOp::getFuncKind())
      return false;

    cir::CIRBaseBuilderTy builder(context);
    builder.setInsertionPointAfter(call.getOperation());
    TargetOp op = buildCall(builder, call, std::make_index_sequence<numArgs>());
    // Preserve call attributes so lowering back does not weaken the IR.
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

  void recognizeStandardLibraryCall(CallOp call,
                                    mlir::SymbolTableCollection &symbolTables);

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

void IdiomRecognizerPass::recognizeStandardLibraryCall(
    CallOp call, mlir::SymbolTableCollection &symbolTables) {
  StdRecognizer<StdFindOp>::raise(call, getContext(), symbolTables);
}

void IdiomRecognizerPass::runOnOperation() {
  // All facts are precomputed on the operations, so the pass needs no AST
  // and also works on parsed CIR assembly.
  theModule = getOperation();
  mlir::SymbolTableCollection symbolTables;

  theModule->walk([&](CallOp callOp) {
    // Skip indirect calls.
    std::optional<llvm::StringRef> callee = callOp.getCallee();
    if (!callee)
      return;

    recognizeStandardLibraryCall(callOp, symbolTables);
  });
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}
