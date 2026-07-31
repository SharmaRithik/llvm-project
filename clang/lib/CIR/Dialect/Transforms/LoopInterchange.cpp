//===- LoopInterchange.cpp - CIR loop interchange ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/Analysis/CIRLoopAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRLOOPINTERCHANGE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CIRLoopInterchangePass
    : public impl::CIRLoopInterchangeBase<CIRLoopInterchangePass> {
  using CIRLoopInterchangeBase::CIRLoopInterchangeBase;

  void runOnOperation() override {
    SmallVector<cir::ForOp, 8> outerLoops;
    getOperation()->walk([&](cir::ForOp loop) {
      if (!loop->getParentOfType<cir::ForOp>())
        outerLoops.push_back(loop);
    });

    for (cir::ForOp loop : outerLoops) {
      FailureOr<cir::TwoLevelLoopNest> nest =
          cir::analyzeTwoLevelLoopNest(loop);
      if (failed(nest) || !emitAnalysisRemarks)
        continue;

      cir::LoopMemoryAnalysis memory = cir::analyzeLoopMemory(*nest);

      std::string message;
      llvm::raw_string_ostream os(message);
      os << "recognized loop nest";
      if (auto function = loop->getParentOfType<cir::FuncOp>())
        os << " in @" << function.getSymName();
      os << " outer init ";
      nest->outer.initial.print(os);
      os << " condition ";
      nest->outer.conditionLHS.print(os);
      os << ' ' << cir::stringifyCmpOpKind(nest->outer.comparison.getKind())
         << ' ';
      nest->outer.conditionRHS.print(os);
      os << " inner init ";
      nest->inner.initial.print(os);
      os << " condition ";
      nest->inner.conditionLHS.print(os);
      os << ' ' << cir::stringifyCmpOpKind(nest->inner.comparison.getKind())
         << ' ';
      nest->inner.conditionRHS.print(os);
      os << " memory " << cir::stringifyLoopMemoryLegality(memory.result);
      loop.emitRemark(os.str());
    }

    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRLoopInterchangePass() {
  return std::make_unique<CIRLoopInterchangePass>();
}
