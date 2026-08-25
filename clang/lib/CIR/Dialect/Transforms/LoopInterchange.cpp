//===- LoopInterchange.cpp - CIR loop interchange ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "clang/CIR/Dialect/Analysis/CIRAliasAnalysis.h"
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

struct RectangularPlan {};

using DomainInterchangePlan =
    std::variant<CanonicalUpperPlan, CanonicalLowerPlan, AffineOffsetPlan,
                 ScaledUpperPlan, ProductBoundPlan, RectangularPlan>;

struct LoopNestRewritePlan {
  Operation *scope;
  Operation *innerInitialOperation;
  bool moveInnerInduction;
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

struct InterchangeLocality {
  unsigned improved = 0;
  unsigned regressed = 0;
  bool analyzable = true;

  bool isProfitable() const { return analyzable && improved > regressed; }
};

static InterchangeLocality
scoreBandCandidateLocality(const cir::ThreeLevelLoopBand &band,
                           const cir::LoopBandMemoryAnalysis &memory,
                           const cir::LoopDomain &inner) {
  InterchangeLocality locality;
  for (const cir::LoopMemoryAccess &access : memory.accesses) {
    if (!inner.loop->isAncestor(access.operation))
      continue;
    if (access.subscripts.empty()) {
      locality.analyzable = false;
      return locality;
    }

    const cir::LoopDomainExpr &innermost = access.subscripts.back();
    if (innermost.getKind() == cir::LoopDomainExpr::Kind::Induction) {
      if (innermost.getInduction() == band.outer.induction) {
        ++locality.improved;
        continue;
      }
      if (innermost.getInduction() == inner.induction) {
        ++locality.regressed;
        continue;
      }
    }
    if (innermost.dependsOn(band.outer.induction) ||
        innermost.dependsOn(inner.induction)) {
      locality.analyzable = false;
      return locality;
    }
  }
  return locality;
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

static bool isInvariantRectangularValue(const cir::LoopDomainExpr &expression,
                                        cir::ForOp outerLoop) {
  return expression.getKind() == cir::LoopDomainExpr::Kind::Constant ||
         (expression.getKind() == cir::LoopDomainExpr::Kind::Symbol &&
          isInvariantSymbol(expression, outerLoop));
}

static std::optional<RectangularPlan>
matchRectangularDomain(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isInvariantRectangularValue(nest.outer.initial, nest.outer.loop) ||
      !isInvariantRectangularValue(nest.inner.initial, nest.outer.loop) ||
      !isInvariantRectangularValue(nest.outer.conditionRHS, nest.outer.loop) ||
      !isInvariantRectangularValue(nest.inner.conditionRHS, nest.outer.loop) ||
      nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction)
    return std::nullopt;

  if (nest.outer.stepLoad.getResult().getType() !=
          nest.inner.stepLoad.getResult().getType() ||
      nest.outer.initial.getSource().getType() !=
          nest.outer.stepLoad.getResult().getType() ||
      nest.inner.initial.getSource().getType() !=
          nest.inner.stepLoad.getResult().getType())
    return std::nullopt;
  return RectangularPlan{};
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
  if (auto plan = matchRectangularDomain(nest))
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
  bool moveInnerInduction = nest.inner.induction->getBlock() == &scopeBody;
  unsigned expectedOperations =
      3u + (innerInitialOperation ? 1u : 0u) + (moveInnerInduction ? 1u : 0u);
  if (scopeBody.getOperations().size() != expectedOperations)
    return std::nullopt;

  auto operation = scopeBody.begin();
  if ((moveInnerInduction &&
       &*operation++ != nest.inner.induction.getOperation()) ||
      (innerInitialOperation && &*operation++ != innerInitialOperation) ||
      &*operation++ != nest.inner.initialization.getOperation() ||
      &*operation++ != innerLoop.getOperation() ||
      !isa<cir::YieldOp>(&*operation))
    return std::nullopt;

  if (nest.inner.induction->getNumOperands() != 0 ||
      nest.outer.initialization->getBlock() != outerLoop->getBlock())
    return std::nullopt;
  return LoopNestRewritePlan{scope.getOperation(), innerInitialOperation,
                             moveInnerInduction};
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

  if (plan.moveInnerInduction)
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
                                     const RectangularPlan &) {
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

struct BandRewritePhase {
  cir::ScopeOp innerSetup;
  cir::ForOp innerLoop;
  SmallVector<Operation *, 8> operations;
  SmallVector<Operation *, 8> innerSetupOperations;

  bool isNested() const { return innerLoop != nullptr; }
};

struct BandRewritePlan {
  cir::ScopeOp outerSetup;
  cir::ForOp outerLoop;
  SmallVector<Operation *, 8> outerSetupOperations;
  SmallVector<BandRewritePhase, 4> phases;
};

static bool areBandDomainSymbolsInvariant(const cir::LoopDomainExpr &expression,
                                          cir::ForOp anchorLoop) {
  if (expression.getKind() == cir::LoopDomainExpr::Kind::Symbol) {
    if (isa<BlockArgument>(expression.getSource()))
      return true;
    return isInvariantSymbol(expression, anchorLoop);
  }
  return (!expression.getLHS() ||
          areBandDomainSymbolsInvariant(*expression.getLHS(), anchorLoop)) &&
         (!expression.getRHS() ||
          areBandDomainSymbolsInvariant(*expression.getRHS(), anchorLoop));
}

static void collectSetupExpressionOperations(
    const cir::LoopDomainExpr &expression, Block &block,
    llvm::SmallPtrSetImpl<Operation *> &operations) {
  Operation *operation = expression.getSource().getDefiningOp();
  if (operation && operation->getBlock() == &block)
    operations.insert(operation);
  if (expression.getLHS())
    collectSetupExpressionOperations(*expression.getLHS(), block, operations);
  if (expression.getRHS())
    collectSetupExpressionOperations(*expression.getRHS(), block, operations);
}

static bool
matchLoopSetupInBlock(cir::LoopDomain &domain, Block &block,
                      SmallVectorImpl<Operation *> &setupOperations) {
  Operation *terminator = block.getTerminator();
  if (domain.loop->getBlock() != &block ||
      domain.loop->getNextNode() != terminator ||
      !isa<cir::YieldOp>(terminator) ||
      domain.initialization->getBlock() != &block)
    return false;

  llvm::SmallPtrSet<Operation *, 8> expected;
  collectSetupExpressionOperations(domain.initial, block, expected);
  expected.insert(domain.initialization.getOperation());
  if (domain.induction->getBlock() == &block)
    expected.insert(domain.induction.getOperation());

  for (Operation &operation :
       llvm::make_range(block.begin(), Block::iterator(domain.loop))) {
    if (!expected.erase(&operation))
      return false;
    setupOperations.push_back(&operation);
  }
  return expected.empty();
}

static bool matchLoopSetup(cir::LoopDomain &domain, cir::ScopeOp scope,
                           SmallVectorImpl<Operation *> &setupOperations) {
  if (scope.getNumResults() != 0 || !scope.getScopeRegion().hasOneBlock())
    return false;

  return matchLoopSetupInBlock(domain, scope.getScopeRegion().front(),
                               setupOperations);
}

static Operation *getIterationRoot(Operation *operation, Block &iteration) {
  while (operation && operation->getBlock() != &iteration)
    operation = operation->getParentOp();
  return operation;
}

static int getBandPhase(Operation *operation, Block &iteration,
                        ArrayRef<BandRewritePhase> phases) {
  Operation *root = getIterationRoot(operation, iteration);
  if (!root)
    return -1;
  for (auto [index, phase] : llvm::enumerate(phases))
    if (llvm::is_contained(phase.operations, root))
      return static_cast<int>(index);
  return -1;
}

static bool hasPhaseLocalSSAUses(Block &iteration,
                                 ArrayRef<BandRewritePhase> phases) {
  for (Operation &root : iteration.without_terminator()) {
    int phase = getBandPhase(&root, iteration, phases);
    WalkResult walkResult = root.walk([&](Operation *operation) {
      for (Value result : operation->getResults())
        for (Operation *user : result.getUsers())
          if (getBandPhase(user, iteration, phases) != phase)
            return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return false;
  }
  return true;
}

static std::optional<BandRewritePlan>
matchBandRewrite(cir::ThreeLevelLoopBand &band) {
  cir::ForOp anchorLoop = band.anchor.loop;
  if (!areBandDomainSymbolsInvariant(band.outer.initial, anchorLoop) ||
      !areBandDomainSymbolsInvariant(band.outer.conditionLHS, anchorLoop) ||
      !areBandDomainSymbolsInvariant(band.outer.conditionRHS, anchorLoop))
    return std::nullopt;

  auto outerSetup =
      dyn_cast_or_null<cir::ScopeOp>(band.outer.loop->getParentOp());
  if (!outerSetup || !band.outer.loop.getBody().hasOneBlock())
    return std::nullopt;

  Block &outerBody = band.outer.loop.getBody().front();
  if (outerBody.getOperations().size() != 2 ||
      !isa<cir::YieldOp>(outerBody.back()))
    return std::nullopt;
  auto iteration = dyn_cast<cir::ScopeOp>(outerBody.front());
  if (!iteration || iteration.getNumResults() != 0 ||
      !iteration.getScopeRegion().hasOneBlock())
    return std::nullopt;

  SmallVector<Operation *, 8> outerSetupOperations;
  if (!matchLoopSetup(band.outer, outerSetup, outerSetupOperations))
    return std::nullopt;

  Block &iterationBlock = iteration.getScopeRegion().front();
  if (!isa<cir::YieldOp>(iterationBlock.back()))
    return std::nullopt;

  SmallVector<cir::ScopeOp, 2> innerSetups;
  SmallVector<SmallVector<Operation *, 8>, 2> innerSetupOperations;
  bool hasInlineInnerSetup = false;
  for (cir::LoopDomain &inner : band.innerCandidates) {
    if (inner.initial.dependsOn(band.outer.induction) ||
        inner.conditionLHS.dependsOn(band.outer.induction) ||
        inner.conditionRHS.dependsOn(band.outer.induction) ||
        !areBandDomainSymbolsInvariant(inner.initial, anchorLoop) ||
        !areBandDomainSymbolsInvariant(inner.conditionLHS, anchorLoop) ||
        !areBandDomainSymbolsInvariant(inner.conditionRHS, anchorLoop) ||
        !inner.loop.getBody().hasOneBlock())
      return std::nullopt;

    auto innerSetup = dyn_cast_or_null<cir::ScopeOp>(inner.loop->getParentOp());
    SmallVector<Operation *, 8> setupOperations;
    if (innerSetup == iteration) {
      if (band.innerCandidates.size() != 1 ||
          !matchLoopSetupInBlock(inner, iterationBlock, setupOperations))
        return std::nullopt;
      hasInlineInnerSetup = true;
      innerSetup = {};
    } else {
      if (!innerSetup || innerSetup == outerSetup ||
          innerSetup->getBlock() != &iterationBlock)
        return std::nullopt;
      if (!matchLoopSetup(inner, innerSetup, setupOperations))
        return std::nullopt;
    }
    innerSetups.push_back(innerSetup);
    innerSetupOperations.push_back(std::move(setupOperations));
  }

  SmallVector<BandRewritePhase, 4> phases;
  SmallVector<Operation *, 8> statements;
  auto flushStatements = [&] {
    if (statements.empty())
      return;
    BandRewritePhase phase;
    phase.operations = std::move(statements);
    phases.push_back(std::move(phase));
    statements.clear();
  };

  unsigned innerIndex = 0;
  if (hasInlineInnerSetup) {
    BandRewritePhase phase;
    phase.innerLoop = band.innerCandidates.front().loop;
    phase.operations.append(innerSetupOperations.front());
    phase.operations.push_back(phase.innerLoop.getOperation());
    phase.innerSetupOperations = std::move(innerSetupOperations.front());
    phases.push_back(std::move(phase));
    innerIndex = 1;
  } else {
    for (Operation &operation : iterationBlock.without_terminator()) {
      if (innerIndex == innerSetups.size() ||
          &operation != innerSetups[innerIndex].getOperation()) {
        statements.push_back(&operation);
        continue;
      }

      flushStatements();
      BandRewritePhase phase;
      phase.innerSetup = innerSetups[innerIndex];
      phase.innerLoop = band.innerCandidates[innerIndex].loop;
      phase.operations.push_back(&operation);
      phase.innerSetupOperations = std::move(innerSetupOperations[innerIndex]);
      phases.push_back(std::move(phase));
      ++innerIndex;
    }
    flushStatements();
  }

  if (innerIndex != band.innerCandidates.size() ||
      !hasPhaseLocalSSAUses(iterationBlock, phases))
    return std::nullopt;

  return BandRewritePlan{outerSetup, band.outer.loop,
                         std::move(outerSetupOperations), std::move(phases)};
}

static void cloneBandSetup(BandRewritePlan &plan, IRMapping &mapping) {
  Operation *clone = plan.outerSetup->clone(mapping);
  plan.outerSetup->getBlock()->getOperations().insert(
      plan.outerSetup->getIterator(), clone);
}

static void eraseMappedOperations(ArrayRef<Operation *> operations,
                                  const IRMapping &mapping) {
  for (Operation *operation : llvm::reverse(operations))
    mapping.lookup(operation)->erase();
}

static void eraseOtherPhases(BandRewritePlan &plan, BandRewritePhase &kept,
                             const IRMapping &mapping) {
  for (BandRewritePhase &phase : llvm::reverse(plan.phases)) {
    if (&phase == &kept)
      continue;
    eraseMappedOperations(phase.operations, mapping);
  }
}

static void interchangeClonedBand(BandRewritePlan &plan,
                                  BandRewritePhase &phase,
                                  const IRMapping &mapping) {
  auto outerLoop =
      cast<cir::ForOp>(mapping.lookup(plan.outerLoop.getOperation()));
  auto innerLoop =
      cast<cir::ForOp>(mapping.lookup(phase.innerLoop.getOperation()));
  cir::ScopeOp innerSetup;
  Operation *innerSetupAnchor;
  if (phase.innerSetup) {
    innerSetup =
        cast<cir::ScopeOp>(mapping.lookup(phase.innerSetup.getOperation()));
    innerSetupAnchor = innerSetup;
  } else {
    innerSetupAnchor = mapping.lookup(phase.innerSetupOperations.front());
  }

  Block &innerBody = innerLoop.getBody().front();
  for (Operation &operation :
       llvm::make_early_inc_range(innerBody.without_terminator()))
    operation.moveBefore(innerSetupAnchor);

  for (Operation *operation : phase.innerSetupOperations)
    mapping.lookup(operation)->moveBefore(outerLoop);
  innerLoop->moveBefore(outerLoop);
  if (innerSetup)
    innerSetup->erase();

  for (Operation *operation : plan.outerSetupOperations)
    mapping.lookup(operation)->moveBefore(innerBody.getTerminator());
  outerLoop->moveBefore(innerBody.getTerminator());
}

static void applyBandRewrite(BandRewritePlan &plan) {
  for (BandRewritePhase &phase : plan.phases) {
    IRMapping mapping;
    cloneBandSetup(plan, mapping);
    eraseOtherPhases(plan, phase, mapping);
    if (phase.isNested())
      interchangeClonedBand(plan, phase, mapping);
  }
  plan.outerSetup->erase();
}

struct CIRLoopInterchangePass
    : public impl::CIRLoopInterchangeBase<CIRLoopInterchangePass> {
  using CIRLoopInterchangeBase::CIRLoopInterchangeBase;

  void runOnOperation() override {
    AliasAnalysis aliasAnalysis(getOperation());
    cir::registerCIRAliasAnalyses(aliasAnalysis);

    bool changed = false;
    SmallVector<cir::ForOp, 8> outerLoops;
    getOperation()->walk([&](cir::ForOp loop) {
      if (!loop->getParentOfType<cir::ForOp>())
        outerLoops.push_back(loop);
    });

    for (cir::ForOp loop : outerLoops) {
      FailureOr<cir::ThreeLevelLoopBand> band =
          cir::analyzeThreeLevelLoopBand(loop);
      if (succeeded(band)) {
        cir::LoopBandMemoryAnalysis bandMemory =
            cir::analyzeLoopBandMemory(*band, aliasAnalysis);
        SmallVector<InterchangeLocality, 2> localities;
        for (const cir::LoopDomain &inner : band->innerCandidates)
          localities.push_back(
              scoreBandCandidateLocality(*band, bandMemory, inner));

        if (emitAnalysisRemarks) {
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
          for (auto [index, locality] : llvm::enumerate(localities)) {
            os << " candidate " << index << " locality ";
            if (!locality.analyzable) {
              os << "unknown";
              continue;
            }
            os << "improved " << locality.improved << " regressed "
               << locality.regressed << ' '
               << (locality.isProfitable() ? "profitable" : "not profitable");
          }
          loop.emitRemark(os.str());
        }

        bool allProfitable =
            !localities.empty() &&
            llvm::all_of(localities, [](const InterchangeLocality &locality) {
              return locality.isProfitable();
            });
        if (bandMemory.isSafe() && allProfitable) {
          std::optional<BandRewritePlan> plan = matchBandRewrite(*band);
          if (plan) {
            unsigned nestedPhases = band->innerCandidates.size();
            applyBandRewrite(*plan);
            changed = true;
            if (emitAnalysisRemarks) {
              std::string message;
              llvm::raw_string_ostream os(message);
              os << "distributed and interchanged " << nestedPhases
                 << " nested loop " << (nestedPhases == 1 ? "phase" : "phases");
              loop.emitRemark(os.str());
            }
            continue;
          }
        }
      }

      FailureOr<cir::TwoLevelLoopNest> nest =
          cir::analyzeTwoLevelLoopNest(loop);
      if (failed(nest))
        continue;

      cir::LoopMemoryAnalysis memory =
          cir::analyzeLoopMemory(*nest, aliasAnalysis);
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
      os << " floating recurrences " << memory.recurrences.size();
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
