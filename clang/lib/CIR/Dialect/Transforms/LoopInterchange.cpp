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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <optional>
#include <variant>

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

struct CanonicalUpperPlan {
  cir::IntAttr bound;
  Operation *oldOuterInitial;
  Operation *oldInnerBound;
};

struct CanonicalLowerPlan {
  Operation *oldInnerInitial;
  Operation *oldOuterBound;
};

struct AffineOffsetPlan {
  cir::IntAttr extent;
  cir::IntAttr offset;
};

struct ScaledUpperPlan {
  cir::IntAttr newOuterBound;
  cir::IntAttr coefficient;
  Operation *oldOuterInitial;
};

struct ProductBoundPlan {
  cir::IntAttr extent;
  Value innerInduction;
  Operation *oldOuterBound;
};

struct RectangularSymbolPlan {};

using DomainInterchangePlan =
    std::variant<CanonicalUpperPlan, CanonicalLowerPlan, AffineOffsetPlan,
                 ScaledUpperPlan, ProductBoundPlan, RectangularSymbolPlan>;

struct LoopNestRewritePlan {
  Operation *scope;
  Operation *innerInitialOperation;
};

struct LoopInterchangePlan {
  LoopNestRewritePlan structure;
  DomainInterchangePlan domain;
};

static bool isProfitableInterchange(const cir::TwoLevelLoopNest &nest,
                                    const cir::LoopMemoryAnalysis &memory) {
  if (memory.accesses.empty())
    return false;

  unsigned improved = 0;
  unsigned regressed = 0;
  for (const cir::LoopMemoryAccess &access : memory.accesses) {
    if (access.subscripts.empty())
      return false;
    const cir::LoopDomainExpr &innermost = access.subscripts.back();
    if (innermost.getKind() != cir::LoopDomainExpr::Kind::Induction)
      return false;
    if (innermost.getInduction() == nest.outer.induction) {
      ++improved;
      continue;
    }
    if (innermost.getInduction() == nest.inner.induction) {
      ++regressed;
      continue;
    }
    return false;
  }
  return improved > regressed;
}

static std::optional<CanonicalUpperPlan>
matchCanonicalUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantOne(nest.outer.initial) || !isConstantZero(nest.inner.initial))
    return std::nullopt;

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
    return std::nullopt;

  cir::IntAttr bound = getIntegerConstant(nest.outer.conditionRHS);
  if (!bound)
    return std::nullopt;
  auto boundType = dyn_cast<cir::IntType>(bound.getType());
  if (!boundType || nest.outer.stepLoad.getResult().getType() !=
                        nest.inner.stepLoad.getResult().getType())
    return std::nullopt;

  llvm::APInt one(bound.getValue().getBitWidth(), 1);
  bool hasIterations = boundType.isSigned() ? bound.getValue().sgt(one)
                                            : bound.getValue().ugt(one);
  if (!hasIterations)
    return std::nullopt;

  auto oldOuterInitial =
      nest.outer.initial.getSource().getDefiningOp<cir::ConstantOp>();
  Operation *oldInnerBound =
      nest.inner.conditionRHS.getSource().getDefiningOp();
  if (!oldOuterInitial || !oldInnerBound)
    return std::nullopt;
  return CanonicalUpperPlan{bound, oldOuterInitial.getOperation(),
                            oldInnerBound};
}

static std::optional<CanonicalLowerPlan>
matchCanonicalLowerTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantZero(nest.outer.initial) ||
      nest.inner.initial.getKind() != cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.initial.getInduction() != nest.outer.induction)
    return std::nullopt;

  if (nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.outer.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Constant ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction ||
      nest.inner.conditionRHS.getKind() != cir::LoopDomainExpr::Kind::Constant)
    return std::nullopt;

  cir::IntAttr outerBound = getIntegerConstant(nest.outer.conditionRHS);
  cir::IntAttr innerBound = getIntegerConstant(nest.inner.conditionRHS);
  if (!outerBound || !innerBound || outerBound != innerBound ||
      nest.outer.stepLoad.getResult().getType() !=
          nest.inner.stepLoad.getResult().getType())
    return std::nullopt;

  Operation *oldInnerInitial = nest.inner.initial.getSource().getDefiningOp();
  Operation *oldOuterBound =
      nest.outer.conditionRHS.getSource().getDefiningOp();
  if (!oldInnerInitial || !oldOuterBound)
    return std::nullopt;
  return CanonicalLowerPlan{oldInnerInitial, oldOuterBound};
}

static cir::IntAttr getAddedConstant(const cir::LoopDomainExpr &expression,
                                     cir::AllocaOp induction) {
  if (expression.getKind() != cir::LoopDomainExpr::Kind::Add)
    return {};

  const cir::LoopDomainExpr *lhs = expression.getLHS();
  const cir::LoopDomainExpr *rhs = expression.getRHS();
  if (lhs->getKind() == cir::LoopDomainExpr::Kind::Induction &&
      lhs->getInduction() == induction)
    return getIntegerConstant(*rhs);
  if (rhs->getKind() == cir::LoopDomainExpr::Kind::Induction &&
      rhs->getInduction() == induction)
    return getIntegerConstant(*lhs);
  return {};
}

static cir::IntAttr getMultipliedConstant(const cir::LoopDomainExpr &expression,
                                          cir::AllocaOp induction) {
  if (expression.getKind() != cir::LoopDomainExpr::Kind::Mul)
    return {};

  const cir::LoopDomainExpr *lhs = expression.getLHS();
  const cir::LoopDomainExpr *rhs = expression.getRHS();
  if (lhs->getKind() == cir::LoopDomainExpr::Kind::Induction &&
      lhs->getInduction() == induction)
    return getIntegerConstant(*rhs);
  if (rhs->getKind() == cir::LoopDomainExpr::Kind::Induction &&
      rhs->getInduction() == induction)
    return getIntegerConstant(*lhs);
  return {};
}

static std::optional<AffineOffsetPlan>
matchAffineOffsetUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantZero(nest.outer.initial) ||
      !isConstantZero(nest.inner.initial) ||
      nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction)
    return std::nullopt;

  const cir::LoopDomainExpr &outerBound = nest.outer.conditionRHS;
  if (outerBound.getKind() != cir::LoopDomainExpr::Kind::Sub)
    return std::nullopt;
  cir::IntAttr extent = getIntegerConstant(*outerBound.getLHS());
  cir::IntAttr outerOffset = getIntegerConstant(*outerBound.getRHS());
  cir::IntAttr innerOffset =
      getAddedConstant(nest.inner.conditionRHS, nest.outer.induction);
  if (!extent || !outerOffset || !innerOffset || outerOffset != innerOffset ||
      extent.getType() != outerOffset.getType())
    return std::nullopt;

  auto type = dyn_cast<cir::IntType>(extent.getType());
  if (!type || !type.isSigned() ||
      nest.outer.stepLoad.getResult().getType() != type ||
      nest.inner.stepLoad.getResult().getType() != type)
    return std::nullopt;

  const llvm::APInt &extentValue = extent.getValue();
  const llvm::APInt &offsetValue = outerOffset.getValue();
  if (!offsetValue.isStrictlyPositive() || !extentValue.sgt(offsetValue))
    return std::nullopt;
  return AffineOffsetPlan{extent, outerOffset};
}

static std::optional<ScaledUpperPlan>
matchScaledUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantOne(nest.outer.initial) ||
      !isConstantZero(nest.inner.initial) ||
      nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction)
    return std::nullopt;

  const cir::LoopDomainExpr &outerBound = nest.outer.conditionRHS;
  if (outerBound.getKind() != cir::LoopDomainExpr::Kind::Div)
    return std::nullopt;
  cir::IntAttr numerator = getIntegerConstant(*outerBound.getLHS());
  cir::IntAttr divisor = getIntegerConstant(*outerBound.getRHS());
  cir::IntAttr coefficient =
      getMultipliedConstant(nest.inner.conditionRHS, nest.outer.induction);
  if (!numerator || !divisor || !coefficient ||
      numerator.getType() != divisor.getType() ||
      numerator.getType() != coefficient.getType())
    return std::nullopt;

  auto type = dyn_cast<cir::IntType>(numerator.getType());
  if (!type || !type.isSigned() ||
      nest.outer.stepLoad.getResult().getType() != type ||
      nest.inner.stepLoad.getResult().getType() != type)
    return std::nullopt;

  const llvm::APInt &numeratorValue = numerator.getValue();
  const llvm::APInt &divisorValue = divisor.getValue();
  const llvm::APInt &coefficientValue = coefficient.getValue();
  if (!numeratorValue.isStrictlyPositive() ||
      !divisorValue.isStrictlyPositive() ||
      !coefficientValue.isStrictlyPositive())
    return std::nullopt;

  llvm::APInt upperBound = numeratorValue.sdiv(divisorValue);
  llvm::APInt one(upperBound.getBitWidth(), 1);
  if (!upperBound.sgt(one))
    return std::nullopt;
  bool overflow = false;
  llvm::APInt newOuterBoundValue =
      (upperBound - one).smul_ov(coefficientValue, overflow);
  if (overflow)
    return std::nullopt;

  auto oldOuterInitial =
      nest.outer.initial.getSource().getDefiningOp<cir::ConstantOp>();
  if (!oldOuterInitial)
    return std::nullopt;
  auto newOuterBound = cir::IntAttr::get(type, newOuterBoundValue);
  return ScaledUpperPlan{newOuterBound, coefficient,
                         oldOuterInitial.getOperation()};
}

static bool isInductionProduct(const cir::LoopDomainExpr &expression,
                               cir::AllocaOp lhsInduction,
                               cir::AllocaOp rhsInduction) {
  if (expression.getKind() != cir::LoopDomainExpr::Kind::Mul)
    return false;
  const cir::LoopDomainExpr *lhs = expression.getLHS();
  const cir::LoopDomainExpr *rhs = expression.getRHS();
  if (lhs->getKind() != cir::LoopDomainExpr::Kind::Induction ||
      rhs->getKind() != cir::LoopDomainExpr::Kind::Induction)
    return false;
  return (lhs->getInduction() == lhsInduction &&
          rhs->getInduction() == rhsInduction) ||
         (lhs->getInduction() == rhsInduction &&
          rhs->getInduction() == lhsInduction);
}

static std::optional<ProductBoundPlan>
matchProductBoundTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantOne(nest.outer.initial) ||
      !isConstantZero(nest.inner.initial) ||
      nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      !isInductionProduct(nest.inner.conditionLHS, nest.outer.induction,
                          nest.inner.induction) ||
      nest.outer.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Constant ||
      nest.inner.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Constant ||
      !nest.outer.conditionRHS.isStructurallyEqual(nest.inner.conditionRHS))
    return std::nullopt;

  cir::IntAttr extent = getIntegerConstant(nest.outer.conditionRHS);
  if (!extent)
    return std::nullopt;
  auto type = dyn_cast<cir::IntType>(extent.getType());
  if (!type || !type.isSigned() ||
      nest.outer.stepLoad.getResult().getType() != type ||
      nest.inner.stepLoad.getResult().getType() != type)
    return std::nullopt;

  llvm::APInt one(extent.getValue().getBitWidth(), 1);
  if (!extent.getValue().sgt(one))
    return std::nullopt;
  llvm::APInt two(extent.getValue().getBitWidth(), 2);
  bool overflow = false;
  (void)(extent.getValue() - one).smul_ov(two, overflow);
  if (overflow)
    return std::nullopt;

  const cir::LoopDomainExpr *productLHS = nest.inner.conditionLHS.getLHS();
  const cir::LoopDomainExpr *productRHS = nest.inner.conditionLHS.getRHS();
  const cir::LoopDomainExpr *innerInduction =
      productLHS->getInduction() == nest.inner.induction ? productLHS
                                                         : productRHS;
  Operation *oldOuterBound =
      nest.outer.conditionRHS.getSource().getDefiningOp();
  if (!oldOuterBound)
    return std::nullopt;
  return ProductBoundPlan{extent, innerInduction->getSource(), oldOuterBound};
}

static bool isInvariantSymbol(const cir::LoopDomainExpr &expression,
                              cir::ForOp outerLoop) {
  if (expression.getKind() != cir::LoopDomainExpr::Kind::Symbol)
    return false;
  if (isa<BlockArgument>(expression.getSource()))
    return true;

  auto load = expression.getSource().getDefiningOp<cir::LoadOp>();
  if (!load || load.getIsVolatile() || load.getMemOrder())
    return false;
  auto variable = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!variable || outerLoop->isAncestor(variable.getOperation()))
    return false;

  for (OpOperand &use : variable.getAddr().getUses()) {
    Operation *user = use.getOwner();
    if (auto candidate = dyn_cast<cir::LoadOp>(user)) {
      if (use.getOperandNumber() != cir::LoadOp::odsIndex_addr ||
          candidate.getIsVolatile() || candidate.getMemOrder())
        return false;
      continue;
    }
    if (auto candidate = dyn_cast<cir::StoreOp>(user)) {
      if (use.getOperandNumber() != cir::StoreOp::odsIndex_addr ||
          candidate.getIsVolatile() || candidate.getMemOrder() ||
          outerLoop->isAncestor(user))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static std::optional<RectangularSymbolPlan>
matchRectangularSymbolStart(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantZero(nest.outer.initial) ||
      !isInvariantSymbol(nest.inner.initial, nest.outer.loop) ||
      nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction ||
      nest.outer.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Constant ||
      !nest.outer.conditionRHS.isStructurallyEqual(nest.inner.conditionRHS))
    return std::nullopt;

  if (nest.outer.stepLoad.getResult().getType() !=
          nest.inner.stepLoad.getResult().getType() ||
      nest.inner.initial.getSource().getType() !=
          nest.inner.stepLoad.getResult().getType())
    return std::nullopt;
  return RectangularSymbolPlan{};
}

static std::optional<DomainInterchangePlan>
matchLoopInterchangeDomain(cir::TwoLevelLoopNest &nest) {
  if (auto plan = matchCanonicalUpperTriangle(nest))
    return DomainInterchangePlan(std::move(*plan));
  if (auto plan = matchCanonicalLowerTriangle(nest))
    return DomainInterchangePlan(std::move(*plan));
  if (auto plan = matchAffineOffsetUpperTriangle(nest))
    return DomainInterchangePlan(std::move(*plan));
  if (auto plan = matchScaledUpperTriangle(nest))
    return DomainInterchangePlan(std::move(*plan));
  if (auto plan = matchProductBoundTriangle(nest))
    return DomainInterchangePlan(std::move(*plan));
  if (auto plan = matchRectangularSymbolStart(nest))
    return DomainInterchangePlan(std::move(*plan));
  return std::nullopt;
}

static bool
collectConditionOperations(const cir::LoopDomainExpr &expression,
                           Block &condition,
                           llvm::SmallPtrSetImpl<Operation *> &operations) {
  Operation *operation = expression.getSource().getDefiningOp();
  if (!operation || operation->getBlock() != &condition)
    return false;
  operations.insert(operation);

  if (expression.getLHS() &&
      !collectConditionOperations(*expression.getLHS(), condition, operations))
    return false;
  return !expression.getRHS() ||
         collectConditionOperations(*expression.getRHS(), condition,
                                    operations);
}

static bool hasCanonicalConditionLayout(cir::LoopDomain &domain) {
  Block &condition = domain.loop.getCond().front();
  if (domain.comparison->getBlock() != &condition ||
      domain.comparison->getNextNode() != condition.getTerminator() ||
      !isa<cir::ConditionOp>(condition.getTerminator()))
    return false;

  llvm::SmallPtrSet<Operation *, 8> expressionOperations;
  if (!collectConditionOperations(domain.conditionLHS, condition,
                                  expressionOperations) ||
      !collectConditionOperations(domain.conditionRHS, condition,
                                  expressionOperations))
    return false;

  for (Operation &operation : condition.without_terminator()) {
    if (&operation == domain.comparison.getOperation())
      break;
    if (!expressionOperations.erase(&operation))
      return false;
  }
  return expressionOperations.empty();
}

static std::optional<LoopNestRewritePlan>
matchPerfectLoopNest(cir::TwoLevelLoopNest &nest) {
  cir::ForOp outerLoop = nest.outer.loop;
  cir::ForOp innerLoop = nest.inner.loop;
  if (!outerLoop.getBody().hasOneBlock() || !innerLoop.getBody().hasOneBlock())
    return std::nullopt;
  if (!hasCanonicalConditionLayout(nest.outer) ||
      !hasCanonicalConditionLayout(nest.inner))
    return std::nullopt;

  Block &outerBody = outerLoop.getBody().front();
  if (outerBody.getOperations().size() != 2 ||
      !isa<cir::YieldOp>(outerBody.back()))
    return std::nullopt;

  auto scope = dyn_cast<cir::ScopeOp>(outerBody.front());
  if (!scope || scope.getNumResults() != 0 ||
      !scope.getScopeRegion().hasOneBlock())
    return std::nullopt;

  Block &scopeBody = scope.getScopeRegion().front();
  Operation *innerInitialOperation =
      nest.inner.initial.getSource().getDefiningOp();
  if (scopeBody.getOperations().size() != (innerInitialOperation ? 5u : 4u))
    return std::nullopt;

  auto operation = scopeBody.begin();
  if (&*operation++ != nest.inner.induction.getOperation() ||
      (innerInitialOperation && &*operation++ != innerInitialOperation) ||
      &*operation++ != nest.inner.initialization.getOperation() ||
      &*operation++ != innerLoop.getOperation() ||
      !isa<cir::YieldOp>(&*operation))
    return std::nullopt;

  if (nest.inner.induction->getNumOperands() != 0 ||
      nest.outer.initialization->getBlock() != outerLoop->getBlock())
    return std::nullopt;
  return LoopNestRewritePlan{scope.getOperation(), innerInitialOperation};
}

static std::optional<LoopInterchangePlan>
buildLoopInterchangePlan(cir::TwoLevelLoopNest &nest) {
  std::optional<DomainInterchangePlan> domain =
      matchLoopInterchangeDomain(nest);
  if (!domain)
    return std::nullopt;
  std::optional<LoopNestRewritePlan> structure = matchPerfectLoopNest(nest);
  if (!structure)
    return std::nullopt;
  return LoopInterchangePlan{*structure, std::move(*domain)};
}

static void interchangeLoopStructure(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &plan) {
  cir::ForOp outerLoop = nest.outer.loop;
  cir::ForOp innerLoop = nest.inner.loop;
  Operation *innerInitialOperation = plan.innerInitialOperation;
  Block *parent = outerLoop->getBlock();
  Block &outerBody = outerLoop.getBody().front();
  Block &innerBody = innerLoop.getBody().front();

  nest.inner.induction->moveBefore(outerLoop);
  if (innerInitialOperation)
    innerInitialOperation->moveBefore(outerLoop);
  nest.inner.initialization->moveBefore(outerLoop);
  innerLoop->moveBefore(outerLoop);
  plan.scope->erase();

  outerBody.getOperations().splice(outerBody.begin(), innerBody.getOperations(),
                                   innerBody.begin(),
                                   std::prev(innerBody.end()));
  innerBody.getOperations().splice(innerBody.begin(), parent->getOperations(),
                                   Block::iterator(outerLoop));
  nest.outer.initialization->moveBefore(outerLoop);
}

static void eraseDeadDomainExpression(const cir::LoopDomainExpr &expression) {
  Operation *operation = expression.getSource().getDefiningOp();
  if (!operation || !operation->use_empty())
    return;

  operation->erase();
  if (expression.getLHS())
    eraseDeadDomainExpression(*expression.getLHS());
  if (expression.getRHS())
    eraseDeadDomainExpression(*expression.getRHS());
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &structure,
                                     const CanonicalUpperPlan &plan) {
  cir::IntAttr bound = plan.bound;
  auto boundType = cast<cir::IntType>(bound.getType());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  llvm::APInt one(bound.getValue().getBitWidth(), 1);
  auto newOuterBound = cir::ConstantOp::create(
      builder, nest.inner.comparison.getLoc(),
      cir::IntAttr::get(boundType, bound.getValue() - one));
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  if (plan.oldInnerBound->use_empty())
    plan.oldInnerBound->erase();

  interchangeLoopStructure(nest, structure);
  builder.setInsertionPoint(nest.outer.initialization);
  auto newInnerLoad =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newInnerLoad->setLoc(nest.outer.initialization.getLoc());
  auto newInnerInitial = cir::IncOp::create(
      builder, nest.outer.initialization.getLoc(), newInnerLoad.getResult(),
      nest.inner.increment.getNoSignedWrap());
  nest.outer.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        newInnerInitial);

  if (plan.oldOuterInitial->use_empty())
    plan.oldOuterInitial->erase();
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &structure,
                                     const CanonicalLowerPlan &plan) {
  nest.inner.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        nest.outer.initial.getSource());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.outer.comparison);
  auto newInnerBound =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newInnerBound->setLoc(nest.outer.comparison.getLoc());
  nest.outer.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newInnerBound);
  nest.outer.comparison.setKind(cir::CmpOpKind::le);

  interchangeLoopStructure(nest, structure);
  if (plan.oldInnerInitial->use_empty())
    plan.oldInnerInitial->erase();
  if (plan.oldOuterBound->use_empty())
    plan.oldOuterBound->erase();
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &structure,
                                     const AffineOffsetPlan &plan) {
  cir::IntAttr extent = plan.extent;
  cir::IntAttr offset = plan.offset;
  auto type = cast<cir::IntType>(extent.getType());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  llvm::APInt one(extent.getValue().getBitWidth(), 1);
  auto newOuterBound =
      cir::ConstantOp::create(builder, nest.inner.comparison.getLoc(),
                              cir::IntAttr::get(type, extent.getValue() - one));
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  eraseDeadDomainExpression(nest.inner.conditionRHS);

  interchangeLoopStructure(nest, structure);
  builder.setInsertionPoint(nest.outer.initialization);
  Location location = nest.outer.initialization.getLoc();
  auto newOuterValue =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newOuterValue->setLoc(location);
  auto offsetValue = cir::ConstantOp::create(builder, location, offset);
  auto beforeOffset = cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                         newOuterValue, offsetValue);
  auto difference =
      cir::SubOp::create(builder, location, type, newOuterValue, offsetValue);
  auto adjusted = cir::IncOp::create(builder, location, difference,
                                     /*noSignedWrap=*/false);
  auto newInnerInitial =
      cir::SelectOp::create(builder, location, type, beforeOffset,
                            nest.outer.initial.getSource(), adjusted);
  nest.outer.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        newInnerInitial);
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &structure,
                                     const ScaledUpperPlan &plan) {
  auto type = cast<cir::IntType>(plan.newOuterBound.getType());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  auto newOuterBound = cir::ConstantOp::create(
      builder, nest.inner.comparison.getLoc(), plan.newOuterBound);
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  eraseDeadDomainExpression(nest.inner.conditionRHS);

  interchangeLoopStructure(nest, structure);
  builder.setInsertionPoint(nest.outer.initialization);
  Location location = nest.outer.initialization.getLoc();
  auto newOuterValue =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newOuterValue->setLoc(location);
  auto coefficientValue =
      cir::ConstantOp::create(builder, location, plan.coefficient);
  auto quotient = cir::DivOp::create(builder, location, type, newOuterValue,
                                     coefficientValue);
  auto newInnerInitial = cir::IncOp::create(builder, location, quotient,
                                            /*noSignedWrap=*/false);
  nest.outer.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        newInnerInitial);
  if (plan.oldOuterInitial->use_empty())
    plan.oldOuterInitial->erase();
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &structure,
                                     const ProductBoundPlan &plan) {
  cir::IntAttr extent = plan.extent;
  auto type = cast<cir::IntType>(extent.getType());

  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_lhs,
                                    plan.innerInduction);
  eraseDeadDomainExpression(nest.inner.conditionLHS);

  interchangeLoopStructure(nest, structure);
  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.outer.comparison);
  Location location = nest.outer.comparison.getLoc();
  auto newOuterValue =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newOuterValue->setLoc(location);
  llvm::APInt zeroValue(extent.getValue().getBitWidth(), 0);
  llvm::APInt oneValue(extent.getValue().getBitWidth(), 1);
  auto zero = cir::ConstantOp::create(builder, location,
                                      cir::IntAttr::get(type, zeroValue));
  auto isZero = cir::CmpOp::create(builder, location, cir::CmpOpKind::eq,
                                   newOuterValue, zero);
  auto one = cir::ConstantOp::create(builder, location,
                                     cir::IntAttr::get(type, oneValue));
  auto safeDivisor = cir::SelectOp::create(builder, location, type, isZero, one,
                                           newOuterValue);
  auto reducedExtent = cir::ConstantOp::create(
      builder, location, cir::IntAttr::get(type, extent.getValue() - oneValue));
  auto quotient =
      cir::DivOp::create(builder, location, type, reducedExtent, safeDivisor);
  auto newInnerBound = cir::IncOp::create(builder, location, quotient,
                                          /*noSignedWrap=*/false);
  nest.outer.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newInnerBound);
  if (plan.oldOuterBound->use_empty())
    plan.oldOuterBound->erase();
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopNestRewritePlan &structure,
                                     const RectangularSymbolPlan &) {
  interchangeLoopStructure(nest, structure);
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopInterchangePlan &plan) {
  std::visit(
      [&](const auto &domain) {
        applyLoopInterchangePlan(nest, plan.structure, domain);
      },
      plan.domain);
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
      FailureOr<cir::ThreeLevelLoopBand> band =
          cir::analyzeThreeLevelLoopBand(loop);
      if (succeeded(band) && emitAnalysisRemarks) {
        cir::LoopBandMemoryAnalysis bandMemory =
            cir::analyzeLoopBandMemory(*band);
        std::string message;
        llvm::raw_string_ostream os(message);
        os << "recognized anchored loop band";
        if (auto function = loop->getParentOfType<cir::FuncOp>())
          os << " in @" << function.getSymName();
        os << " outer init ";
        band->outer.initial.print(os);
        os << " inner candidates " << band->innerCandidates.size();
        os << " floating recurrences " << bandMemory.recurrences.size();
        os << " band memory "
           << cir::stringifyLoopMemoryLegality(bandMemory.result);
        loop.emitRemark(os.str());
      }

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

      if (!memory.isSafe() || !profitable)
        continue;

      std::optional<LoopInterchangePlan> plan = buildLoopInterchangePlan(*nest);
      if (!plan)
        continue;
      applyLoopInterchangePlan(*nest, *plan);

      for (cir::LoopReduction &reduction : memory.reductions) {
        reduction.operation.setNoSignedWrap(false);
        reduction.operation.setNoUnsignedWrap(false);
      }

      changed = true;
      if (emitAnalysisRemarks)
        nest->inner.loop.emitRemark("interchanged loop nest");
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createCIRLoopInterchangePass() {
  return std::make_unique<CIRLoopInterchangePass>();
}
