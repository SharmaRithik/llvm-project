//===- LoopInterchange.cpp - CIR loop interchange ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "clang/CIR/Dialect/Analysis/CIRLoopAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CIRLOOPINTERCHANGE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

static cir::IntAttr getIntegerConstant(const cir::LoopDomainExpr &expression) {
  if (expression.getKind() != cir::LoopDomainExpr::Kind::Constant)
    return {};
  auto constant = expression.getSource().getDefiningOp<cir::ConstantOp>();
  if (!constant)
    return {};
  return dyn_cast<cir::IntAttr>(constant.getValue());
}

static bool isConstantZero(const cir::LoopDomainExpr &expression) {
  cir::IntAttr constant = getIntegerConstant(expression);
  return constant && constant.getValue().isZero();
}

static bool isConstantOne(const cir::LoopDomainExpr &expression) {
  cir::IntAttr constant = getIntegerConstant(expression);
  return constant && constant.getValue().isOne();
}

static bool isProfitableInterchange(const cir::TwoLevelLoopNest &nest,
                                    const cir::LoopMemoryAnalysis &memory) {
  if (memory.accesses.empty())
    return false;

  return llvm::all_of(
      memory.accesses, [&](const cir::LoopMemoryAccess &access) {
        if (access.subscripts.empty())
          return false;
        const cir::LoopDomainExpr &innermost = access.subscripts.back();
        return innermost.getKind() == cir::LoopDomainExpr::Kind::Induction &&
               innermost.getInduction() == nest.outer.induction;
      });
}

static bool isCanonicalUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantOne(nest.outer.initial) || !isConstantZero(nest.inner.initial))
    return false;

  if (nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.outer.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Constant ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction ||
      nest.inner.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionRHS.getInduction() != nest.outer.induction)
    return false;

  cir::IntAttr bound = getIntegerConstant(nest.outer.conditionRHS);
  if (!bound)
    return false;
  auto boundType = dyn_cast<cir::IntType>(bound.getType());
  if (!boundType || nest.outer.stepLoad.getResult().getType() !=
                        nest.inner.stepLoad.getResult().getType())
    return false;

  llvm::APInt one(bound.getValue().getBitWidth(), 1);
  return boundType.isSigned() ? bound.getValue().sgt(one)
                              : bound.getValue().ugt(one);
}

static bool hasCanonicalConditionLayout(cir::LoopDomain &domain) {
  Block &condition = domain.loop.getCond().front();
  if (condition.getOperations().size() != 4)
    return false;

  Operation *lhs = domain.conditionLHS.getSource().getDefiningOp();
  Operation *rhs = domain.conditionRHS.getSource().getDefiningOp();
  if (!lhs || !rhs)
    return false;

  auto operation = condition.begin();
  return &*operation++ == lhs && &*operation++ == rhs &&
         &*operation++ == domain.comparison.getOperation() &&
         isa<cir::ConditionOp>(&*operation);
}

static cir::ScopeOp getPerfectNestScope(cir::TwoLevelLoopNest &nest) {
  cir::ForOp outerLoop = nest.outer.loop;
  cir::ForOp innerLoop = nest.inner.loop;
  if (!outerLoop.getBody().hasOneBlock() || !innerLoop.getBody().hasOneBlock())
    return {};
  if (!hasCanonicalConditionLayout(nest.outer) ||
      !hasCanonicalConditionLayout(nest.inner))
    return {};

  Block &outerBody = outerLoop.getBody().front();
  if (outerBody.getOperations().size() != 2 ||
      !isa<cir::YieldOp>(outerBody.back()))
    return {};

  auto scope = dyn_cast<cir::ScopeOp>(outerBody.front());
  if (!scope || scope.getNumResults() != 0 ||
      !scope.getScopeRegion().hasOneBlock())
    return {};

  Block &scopeBody = scope.getScopeRegion().front();
  if (scopeBody.getOperations().size() != 5)
    return {};

  auto operation = scopeBody.begin();
  if (&*operation++ != nest.inner.induction.getOperation())
    return {};
  auto innerInitialConstant =
      nest.inner.initial.getSource().getDefiningOp<cir::ConstantOp>();
  if (!innerInitialConstant ||
      &*operation++ != innerInitialConstant.getOperation() ||
      &*operation++ != nest.inner.initialization.getOperation() ||
      &*operation++ != innerLoop.getOperation() ||
      !isa<cir::YieldOp>(&*operation))
    return {};

  if (nest.inner.induction->getNumOperands() != 0 ||
      nest.outer.initialization->getBlock() != outerLoop->getBlock())
    return {};
  return scope;
}

static LogicalResult
interchangeCanonicalUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (!isCanonicalUpperTriangle(nest))
    return failure();

  cir::ScopeOp scope = getPerfectNestScope(nest);
  if (!scope)
    return failure();

  auto outerInitialConstant =
      nest.outer.initial.getSource().getDefiningOp<cir::ConstantOp>();
  auto innerInitialConstant =
      nest.inner.initial.getSource().getDefiningOp<cir::ConstantOp>();
  Operation *oldInnerBound =
      nest.inner.conditionRHS.getSource().getDefiningOp();
  cir::IntAttr bound = getIntegerConstant(nest.outer.conditionRHS);
  auto boundType = cast<cir::IntType>(bound.getType());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  llvm::APInt one(bound.getValue().getBitWidth(), 1);
  auto newOuterBound = cir::ConstantOp::create(
      builder, nest.inner.comparison.getLoc(),
      cir::IntAttr::get(boundType, bound.getValue() - one));
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  if (oldInnerBound->use_empty())
    oldInnerBound->erase();

  cir::ForOp outerLoop = nest.outer.loop;
  cir::ForOp innerLoop = nest.inner.loop;
  Block *parent = outerLoop->getBlock();
  Block &outerBody = outerLoop.getBody().front();
  Block &innerBody = innerLoop.getBody().front();

  nest.inner.induction->moveBefore(outerLoop);
  innerInitialConstant->moveBefore(outerLoop);
  nest.inner.initialization->moveBefore(outerLoop);
  innerLoop->moveBefore(outerLoop);
  scope.erase();

  outerBody.getOperations().splice(outerBody.begin(), innerBody.getOperations(),
                                   innerBody.begin(),
                                   std::prev(innerBody.end()));
  innerBody.getOperations().splice(innerBody.begin(), parent->getOperations(),
                                   Block::iterator(outerLoop));

  nest.outer.initialization->moveBefore(outerLoop);
  builder.setInsertionPoint(nest.outer.initialization);
  auto newInnerLoad =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newInnerLoad->setLoc(nest.outer.initialization.getLoc());
  auto newInnerInitial = cir::IncOp::create(
      builder, nest.outer.initialization.getLoc(), newInnerLoad.getResult(),
      nest.inner.increment.getNoSignedWrap());
  nest.outer.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        newInnerInitial);

  if (outerInitialConstant->use_empty())
    outerInitialConstant.erase();
  return success();
}

struct CIRLoopInterchangePass
    : public impl::CIRLoopInterchangeBase<CIRLoopInterchangePass> {
  using CIRLoopInterchangeBase::CIRLoopInterchangeBase;

  void runOnOperation() override {
    bool changed = false;
    SmallVector<cir::ForOp, 8> outerLoops;
    getOperation()->walk([&](cir::ForOp loop) {
      if (!loop->getParentOfType<cir::ForOp>())
        outerLoops.push_back(loop);
    });

    for (cir::ForOp loop : outerLoops) {
      FailureOr<cir::TwoLevelLoopNest> nest =
          cir::analyzeTwoLevelLoopNest(loop);
      if (failed(nest))
        continue;

      cir::LoopMemoryAnalysis memory = cir::analyzeLoopMemory(*nest);
      bool profitable = isProfitableInterchange(*nest, memory);

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
      os << " profitability " << (profitable ? "profitable" : "not profitable");
      if (emitAnalysisRemarks)
        loop.emitRemark(os.str());

      if (!memory.isSafe() || !profitable ||
          failed(interchangeCanonicalUpperTriangle(*nest)))
        continue;

      changed = true;
      if (emitAnalysisRemarks)
        nest->inner.loop.emitRemark("interchanged canonical upper triangle");
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRLoopInterchangePass() {
  return std::make_unique<CIRLoopInterchangePass>();
}
