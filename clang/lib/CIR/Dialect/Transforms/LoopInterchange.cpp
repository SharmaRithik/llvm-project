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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

struct LoopDomainPair {
  cir::LoopDomain &outer;
  cir::LoopDomain &inner;
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

template <typename LoopNest>
static std::optional<CanonicalUpperPlan>
matchCanonicalUpperTriangle(LoopNest &nest) {
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
      nest.outer.initial.getSource().template getDefiningOp<cir::ConstantOp>();
  Operation *oldInnerBound =
      nest.inner.conditionRHS.getSource().getDefiningOp();
  if (!oldOuterInitial || !oldInnerBound)
    return std::nullopt;
  return CanonicalUpperPlan{bound, oldOuterInitial.getOperation(),
                            oldInnerBound};
}

template <typename LoopNest>
static std::optional<CanonicalLowerPlan>
matchCanonicalLowerTriangle(LoopNest &nest) {
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

template <typename LoopNest>
static std::optional<AffineOffsetPlan>
matchAffineOffsetUpperTriangle(LoopNest &nest) {
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

template <typename LoopNest>
static std::optional<ScaledUpperPlan> matchScaledUpperTriangle(LoopNest &nest) {
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
      nest.outer.initial.getSource().template getDefiningOp<cir::ConstantOp>();
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

template <typename LoopNest>
static std::optional<ProductBoundPlan>
matchProductBoundTriangle(LoopNest &nest) {
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

template <typename LoopNest>
static std::optional<RectangularPlan> matchRectangularDomain(LoopNest &nest) {
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

template <typename LoopNest>
static std::optional<DomainInterchangePlan>
matchLoopInterchangeDomain(LoopNest &nest) {
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

static bool isForwardBandComparison(cir::CmpOpKind comparison) {
  return comparison == cir::CmpOpKind::lt || comparison == cir::CmpOpKind::le;
}

template <typename LoopNest>
static std::optional<DomainInterchangePlan>
matchBandInterchangeDomain(LoopNest &nest) {
  if (std::optional<DomainInterchangePlan> plan =
          matchLoopInterchangeDomain(nest))
    return plan;
  if (!isForwardBandComparison(nest.outer.comparison.getKind()) ||
      !isForwardBandComparison(nest.inner.comparison.getKind()) ||
      nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction ||
      nest.inner.initial.dependsOn(nest.outer.induction) ||
      nest.inner.conditionRHS.dependsOn(nest.outer.induction) ||
      nest.outer.stepLoad.getResult().getType() !=
          nest.inner.stepLoad.getResult().getType() ||
      nest.outer.initial.getSource().getType() !=
          nest.outer.stepLoad.getResult().getType() ||
      nest.inner.initial.getSource().getType() !=
          nest.inner.stepLoad.getResult().getType())
    return std::nullopt;
  return DomainInterchangePlan(RectangularPlan{});
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

template <typename InterchangeStructure>
static void applyLoopInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const CanonicalUpperPlan &plan,
                                       InterchangeStructure interchange) {
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

  interchange();
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

template <typename InterchangeStructure>
static void applyLoopInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const CanonicalLowerPlan &plan,
                                       InterchangeStructure interchange,
                                       bool cloneOuterInitial = false) {
  Operation *oldOuterInitial = nest.outer.initial.getSource().getDefiningOp();
  if (!cloneOuterInitial)
    nest.inner.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                          nest.outer.initial.getSource());
  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.outer.comparison);
  auto newInnerBound =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newInnerBound->setLoc(nest.outer.comparison.getLoc());
  nest.outer.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newInnerBound);
  nest.outer.comparison.setKind(cir::CmpOpKind::le);

  interchange();
  if (cloneOuterInitial) {
    builder.setInsertionPoint(nest.inner.initialization);
    auto newOuterInitial =
        cast<cir::ConstantOp>(builder.clone(*oldOuterInitial));
    newOuterInitial->setLoc(nest.inner.initialization.getLoc());
    nest.inner.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                          newOuterInitial);
  }
  if (plan.oldInnerInitial->use_empty())
    plan.oldInnerInitial->erase();
  if (plan.oldOuterBound->use_empty())
    plan.oldOuterBound->erase();
}

template <typename InterchangeStructure>
static void applyLoopInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const AffineOffsetPlan &plan,
                                       InterchangeStructure interchange) {
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

  interchange();
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

template <typename InterchangeStructure>
static void applyLoopInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const ScaledUpperPlan &plan,
                                       InterchangeStructure interchange) {
  auto type = cast<cir::IntType>(plan.newOuterBound.getType());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  auto newOuterBound = cir::ConstantOp::create(
      builder, nest.inner.comparison.getLoc(), plan.newOuterBound);
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  eraseDeadDomainExpression(nest.inner.conditionRHS);

  interchange();
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

template <typename InterchangeStructure>
static void applyLoopInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const ProductBoundPlan &plan,
                                       InterchangeStructure interchange) {
  cir::IntAttr extent = plan.extent;
  auto type = cast<cir::IntType>(extent.getType());

  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_lhs,
                                    plan.innerInduction);
  eraseDeadDomainExpression(nest.inner.conditionLHS);

  interchange();
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

template <typename InterchangeStructure>
static void applyLoopInterchangeDomain(cir::TwoLevelLoopNest &,
                                       const RectangularPlan &,
                                       InterchangeStructure interchange) {
  interchange();
}

static void applyLoopInterchangePlan(cir::TwoLevelLoopNest &nest,
                                     const LoopInterchangePlan &plan) {
  auto interchange = [&] { interchangeLoopStructure(nest, plan.structure); };
  std::visit(
      [&](const auto &domain) {
        applyLoopInterchangeDomain(nest, domain, interchange);
      },
      plan.domain);
}

template <typename Domain, typename InterchangeStructure>
static void applyBandInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const Domain &domain,
                                       InterchangeStructure interchange) {
  applyLoopInterchangeDomain(nest, domain, interchange);
}

template <typename InterchangeStructure>
static void applyBandInterchangeDomain(cir::TwoLevelLoopNest &nest,
                                       const CanonicalLowerPlan &domain,
                                       InterchangeStructure interchange) {
  applyLoopInterchangeDomain(nest, domain, interchange,
                             /*cloneOuterInitial=*/true);
}

struct BandRewritePhase {
  cir::ScopeOp innerSetup;
  cir::ForOp innerLoop;
  SmallVector<Operation *, 8> operations;
  SmallVector<Operation *, 8> innerSetupOperations;
  bool interchange = false;
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
matchBandRewrite(cir::ThreeLevelLoopBand &band,
                 ArrayRef<InterchangeLocality> localities) {
  cir::ForOp anchorLoop = band.anchor.loop;
  if (localities.size() != band.innerCandidates.size() ||
      !areBandDomainSymbolsInvariant(band.outer.initial, anchorLoop) ||
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
  for (auto [index, inner] : llvm::enumerate(band.innerCandidates)) {
    if (localities[index].isProfitable()) {
      LoopDomainPair nest{band.outer, inner};
      if (!matchBandInterchangeDomain(nest))
        return std::nullopt;
    }
    if (!areBandDomainSymbolsInvariant(inner.initial, anchorLoop) ||
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
    phase.interchange = localities.front().isProfitable();
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
      phase.interchange = localities[innerIndex].isProfitable();
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

static void interchangeClonedBandStructure(BandRewritePlan &plan,
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

static bool interchangeClonedBand(BandRewritePlan &plan,
                                  BandRewritePhase &phase,
                                  const IRMapping &mapping) {
  auto outerLoop =
      cast<cir::ForOp>(mapping.lookup(plan.outerLoop.getOperation()));
  auto innerLoop =
      cast<cir::ForOp>(mapping.lookup(phase.innerLoop.getOperation()));
  FailureOr<cir::LoopDomain> outer = cir::analyzeLoopDomain(outerLoop);
  if (failed(outer))
    return false;
  FailureOr<cir::LoopDomain> inner =
      cir::analyzeLoopDomain(innerLoop, {outer->induction});
  if (failed(inner))
    return false;

  cir::TwoLevelLoopNest nest{std::move(*outer), std::move(*inner)};
  std::optional<DomainInterchangePlan> domain =
      matchBandInterchangeDomain(nest);
  if (!domain)
    return false;
  auto interchange = [&] {
    interchangeClonedBandStructure(plan, phase, mapping);
  };
  std::visit(
      [&](const auto &candidate) {
        applyBandInterchangeDomain(nest, candidate, interchange);
      },
      *domain);

  return true;
}

static unsigned applyBandRewrite(BandRewritePlan &plan) {
  unsigned interchanged = 0;
  for (BandRewritePhase &phase : plan.phases) {
    IRMapping mapping;
    cloneBandSetup(plan, mapping);
    eraseOtherPhases(plan, phase, mapping);
    if (phase.interchange && interchangeClonedBand(plan, phase, mapping))
      ++interchanged;
  }
  plan.outerSetup->erase();
  return interchanged;
}

struct TileAddress {
  Value base;
  SmallVector<cir::LoopDomainExpr, 2> subscripts;
};

static FailureOr<Value> stripTileIndexCast(Value value) {
  while (auto cast = value.getDefiningOp<cir::CastOp>()) {
    if (cast.getKind() != cir::CastKind::integral)
      return failure();

    auto sourceType = dyn_cast<cir::IntType>(cast.getSrc().getType());
    auto resultType = dyn_cast<cir::IntType>(cast.getResult().getType());
    if (!sourceType || !resultType ||
        sourceType.isSigned() != resultType.isSigned() ||
        sourceType.getWidth() > resultType.getWidth())
      return failure();
    value = cast.getSrc();
  }
  return value;
}

static FailureOr<TileAddress>
analyzeTileAddress(Value address, ArrayRef<cir::AllocaOp> inductions) {
  SmallVector<cir::LoopDomainExpr, 2> subscripts;
  Value current = address;
  while (true) {
    Value index;
    if (auto element = current.getDefiningOp<cir::GetElementOp>()) {
      index = element.getIndex();
      current = element.getBase();
    } else if (auto stride = current.getDefiningOp<cir::PtrStrideOp>()) {
      index = stride.getStride();
      current = stride.getBase();
    } else {
      break;
    }

    FailureOr<Value> stripped = stripTileIndexCast(index);
    if (failed(stripped))
      return failure();
    FailureOr<cir::LoopDomainExpr> expression =
        cir::buildLoopDomainExpr(*stripped, inductions);
    if (failed(expression))
      return failure();
    subscripts.insert(subscripts.begin(), std::move(*expression));
  }

  if (failed(cir::resolveCIRPointerArgument(current)) &&
      failed(cir::resolveCIRConstantGlobalPointer(current)))
    return failure();

  return TileAddress{current, std::move(subscripts)};
}

static bool isSimpleInduction(const cir::LoopDomainExpr &expression,
                              cir::AllocaOp induction) {
  return expression.getKind() == cir::LoopDomainExpr::Kind::Induction &&
         expression.getInduction() == induction;
}

static bool hasInvariantLeaf(const cir::LoopDomainExpr &expression,
                             cir::ForOp loop) {
  if (expression.getKind() == cir::LoopDomainExpr::Kind::Constant)
    return true;
  if (expression.getKind() != cir::LoopDomainExpr::Kind::Symbol)
    return false;
  if (isa<BlockArgument>(expression.getSource()))
    return true;
  return isInvariantSymbol(expression, loop);
}

static bool sameAddress(const TileAddress &lhs,
                        const cir::LoopMemoryAccess &rhs,
                        AliasAnalysis &aliasAnalysis) {
  if (!aliasAnalysis.alias(lhs.base, rhs.base.pointer).isMust() ||
      lhs.subscripts.size() != rhs.subscripts.size())
    return false;
  return llvm::all_of(
      llvm::zip(lhs.subscripts, rhs.subscripts), [](const auto &pair) {
        return std::get<0>(pair).isStructurallyEqual(std::get<1>(pair));
      });
}

static bool dependsOn(Value value, Value target,
                      llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (value == target)
    return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || !visited.insert(operation).second)
    return false;
  return llvm::any_of(operation->getOperands(), [&](Value operand) {
    return dependsOn(operand, target, visited);
  });
}

static bool isInvariantTileLoad(cir::LoadOp load, cir::ForOp loop) {
  auto alloca = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!alloca || loop->isAncestor(alloca.getOperation()))
    return false;
  for (OpOperand &use : alloca.getAddr().getUses()) {
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
          loop->isAncestor(user))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static bool matchScaleBody(cir::ForOp scaleLoop, cir::ForOp enclosingLoop,
                           const cir::LoopElementRecurrence &recurrence,
                           ArrayRef<cir::AllocaOp> inductions,
                           AliasAnalysis &aliasAnalysis,
                           SmallVectorImpl<Operation *> &bodyOperations) {
  SmallVector<cir::LoadOp, 8> loads;
  SmallVector<cir::StoreOp, 2> stores;
  bool supported = true;
  scaleLoop.getBody().walk([&](Operation *operation) {
    if (isa<cir::ForOp, cir::AllocaOp>(operation)) {
      supported = false;
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<cir::LoadOp>(operation)) {
      if (load.getIsVolatile() || load.getMemOrder()) {
        supported = false;
        return WalkResult::interrupt();
      }
      loads.push_back(load);
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<cir::StoreOp>(operation)) {
      if (store.getIsVolatile() || store.getMemOrder()) {
        supported = false;
        return WalkResult::interrupt();
      }
      stores.push_back(store);
      return WalkResult::advance();
    }
    if (isa<cir::YieldOp, cir::ScopeOp, cir::FMulOp, cir::GetElementOp>(
            operation) ||
        mlir::isMemoryEffectFree(operation))
      return WalkResult::advance();
    supported = false;
    return WalkResult::interrupt();
  });
  if (!supported || stores.size() != 1)
    return false;

  cir::StoreOp targetStore = stores.front();
  cir::LoadOp targetLoad;
  for (cir::LoadOp load : loads) {
    if (load.getAddr() == targetStore.getAddr()) {
      if (targetLoad)
        return false;
      targetLoad = load;
      continue;
    }
    if (llvm::any_of(inductions, [&](cir::AllocaOp induction) {
          return load.getAddr() == induction.getAddr();
        }))
      continue;
    if (!isInvariantTileLoad(load, enclosingLoop))
      return false;
  }
  if (!targetLoad)
    return false;

  llvm::SmallPtrSet<Operation *, 16> visited;
  if (!dependsOn(targetStore.getValue(), targetLoad.getResult(), visited))
    return false;

  FailureOr<TileAddress> address =
      analyzeTileAddress(targetStore.getAddr(), inductions);
  if (failed(address) ||
      !sameAddress(*address, recurrence.target, aliasAnalysis))
    return false;

  for (Operation &operation : scaleLoop.getBody().front().without_terminator())
    bodyOperations.push_back(&operation);
  return true;
}

static Value cloneInvariantLeaf(OpBuilder &builder,
                                const cir::LoopDomainExpr &expression) {
  Value source = expression.getSource();
  if (isa<BlockArgument>(source))
    return source;
  Operation *operation = source.getDefiningOp();
  assert(operation && operation->getNumResults() == 1);
  return builder.clone(*operation)->getResult(0);
}

static Value createTileConstant(OpBuilder &builder, Location location,
                                cir::IntType type, uint64_t value) {
  return cir::ConstantOp::create(
      builder, location,
      cir::IntAttr::get(type, llvm::APInt(type.getWidth(), value)));
}

static Value createTileLoad(OpBuilder &builder, Location location,
                            cir::AllocaOp variable) {
  return cir::LoadOp::create(builder, location, variable.getAllocaType(),
                             variable.getAddr());
}

using TileValueBuilder = llvm::function_ref<Value(OpBuilder &, Location)>;
using TileBodyBuilder = llvm::function_ref<void(OpBuilder &, Location)>;

static void emitLoopScope(OpBuilder &builder, Location location,
                          cir::AllocaOp induction,
                          TileValueBuilder initialBuilder,
                          TileValueBuilder conditionBuilder,
                          TileValueBuilder stepBuilder,
                          TileBodyBuilder bodyBuilder) {
  cir::ScopeOp::create(
      builder, location, [&](OpBuilder &builder, Location location) {
        Value initial = initialBuilder(builder, location);
        cir::StoreOp::create(builder, location, initial, induction.getAddr());
        cir::ForOp::create(
            builder, location,
            [&](OpBuilder &builder, Location location) {
              cir::ConditionOp::create(builder, location,
                                       conditionBuilder(builder, location));
            },
            [&](OpBuilder &builder, Location location) {
              bodyBuilder(builder, location);
              cir::YieldOp::create(builder, location);
            },
            [&](OpBuilder &builder, Location location) {
              cir::StoreOp::create(builder, location,
                                   stepBuilder(builder, location),
                                   induction.getAddr());
              cir::YieldOp::create(builder, location);
            });
        cir::YieldOp::create(builder, location);
      });
}

static void cloneTileBody(OpBuilder &builder,
                          ArrayRef<Operation *> operations) {
  IRMapping mapping;
  for (Operation *operation : operations)
    builder.clone(*operation, mapping);
}

class TriangularTileEmitter {
public:
  TriangularTileEmitter(cir::IntType type, cir::IntType wideType,
                        cir::AllocaOp tileI, cir::AllocaOp tileJ,
                        cir::AllocaOp tileK, cir::AllocaOp pointI,
                        cir::AllocaOp pointJ, cir::AllocaOp pointK,
                        const cir::LoopDomainExpr &iBound,
                        const cir::LoopDomainExpr &kBound,
                        ArrayRef<Operation *> scaleBody,
                        ArrayRef<Operation *> updateBody)
      : type(type), wideType(wideType), tileI(tileI), tileJ(tileJ),
        tileK(tileK), pointI(pointI), pointJ(pointJ), pointK(pointK),
        iBound(iBound), kBound(kBound), scaleBody(scaleBody),
        updateBody(updateBody) {}

  void emitScale(OpBuilder &builder, Location location) {
    emitTileI(builder, location, [&](OpBuilder &builder, Location location) {
      emitTileJ(builder, location, [&](OpBuilder &builder, Location location) {
        emitPointI(builder, location,
                   [&](OpBuilder &builder, Location location) {
                     emitRectangularJ(builder, location,
                                      [&](OpBuilder &builder, Location) {
                                        cloneTileBody(builder, scaleBody);
                                      });
                   });
      });
      emitPointI(builder, location, [&](OpBuilder &builder, Location location) {
        emitDiagonalJ(builder, location, [&](OpBuilder &builder, Location) {
          cloneTileBody(builder, scaleBody);
        });
      });
    });
  }

  void emitUpdate(OpBuilder &builder, Location location) {
    emitTileI(builder, location, [&](OpBuilder &builder, Location location) {
      emitTileJ(builder, location, [&](OpBuilder &builder, Location location) {
        emitKBand(builder, location,
                  [&](OpBuilder &builder, Location location, bool fullTile) {
                    emitPointI(
                        builder, location,
                        [&](OpBuilder &builder, Location location) {
                          emitRectangularJ(
                              builder, location,
                              [&](OpBuilder &builder, Location location) {
                                emitPointK(builder, location, fullTile);
                              });
                        });
                  });
      });
      emitKBand(builder, location,
                [&](OpBuilder &builder, Location location, bool fullTile) {
                  emitPointI(builder, location,
                             [&](OpBuilder &builder, Location location) {
                               emitDiagonalJ(
                                   builder, location,
                                   [&](OpBuilder &builder, Location location) {
                                     emitPointK(builder, location, fullTile);
                                   });
                             });
                });
    });
  }

private:
  using KBandBodyBuilder =
      llvm::function_ref<void(OpBuilder &, Location, bool)>;

  static constexpr uint64_t rowTileSize = 32;
  static constexpr uint64_t reductionTileSize = 4;

  Value constant(OpBuilder &builder, Location location, uint64_t value) {
    return createTileConstant(builder, location, type, value);
  }

  Value wideConstant(OpBuilder &builder, Location location, uint64_t value) {
    return createTileConstant(builder, location, wideType, value);
  }

  Value wideLoad(OpBuilder &builder, Location location) {
    return createTileLoad(builder, location, tileI);
  }

  Value pointTileI(OpBuilder &builder, Location location) {
    return cir::CastOp::create(builder, location, type, cir::CastKind::integral,
                               wideLoad(builder, location));
  }

  Value wideBound(OpBuilder &builder, Location location) {
    return cir::CastOp::create(builder, location, wideType,
                               cir::CastKind::integral,
                               cloneInvariantLeaf(builder, iBound));
  }

  Value load(OpBuilder &builder, Location location, cir::AllocaOp variable) {
    return createTileLoad(builder, location, variable);
  }

  Value add(OpBuilder &builder, Location location, Value lhs, Value rhs) {
    return cir::AddOp::create(builder, location, type, lhs, rhs);
  }

  Value subtract(OpBuilder &builder, Location location, Value lhs, Value rhs) {
    return cir::SubOp::create(builder, location, type, lhs, rhs);
  }

  Value increment(OpBuilder &builder, Location location,
                  cir::AllocaOp variable) {
    return cir::IncOp::create(builder, location,
                              load(builder, location, variable),
                              /*noSignedWrap=*/true);
  }

  Value cappedUpper(OpBuilder &builder, Location location, Value start,
                    Value bound, uint64_t width) {
    Value distance = subtract(builder, location, bound, start);
    Value tileWidth = constant(builder, location, width);
    Value shortTile = cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                         distance, tileWidth);
    Value advance = cir::SelectOp::create(builder, location, type, shortTile,
                                          distance, tileWidth);
    return add(builder, location, start, advance);
  }

  void emitTileI(OpBuilder &builder, Location location,
                 TileBodyBuilder bodyBuilder) {
    emitLoopScope(
        builder, location, tileI,
        [&](OpBuilder &builder, Location location) {
          return wideConstant(builder, location, 0);
        },
        [&](OpBuilder &builder, Location location) {
          return cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                    wideLoad(builder, location),
                                    wideBound(builder, location));
        },
        [&](OpBuilder &builder, Location location) {
          return cir::AddOp::create(
              builder, location, wideType, wideLoad(builder, location),
              wideConstant(builder, location, rowTileSize));
        },
        bodyBuilder);
  }

  void emitTileJ(OpBuilder &builder, Location location,
                 TileBodyBuilder bodyBuilder) {
    emitLoopScope(
        builder, location, tileJ,
        [&](OpBuilder &builder, Location location) {
          return constant(builder, location, 0);
        },
        [&](OpBuilder &builder, Location location) {
          Value remaining =
              subtract(builder, location, pointTileI(builder, location),
                       load(builder, location, tileJ));
          return cir::CmpOp::create(builder, location, cir::CmpOpKind::ge,
                                    remaining,
                                    constant(builder, location, rowTileSize));
        },
        [&](OpBuilder &builder, Location location) {
          return add(builder, location, load(builder, location, tileJ),
                     constant(builder, location, rowTileSize));
        },
        bodyBuilder);
  }

  void emitPointI(OpBuilder &builder, Location location,
                  TileBodyBuilder bodyBuilder) {
    emitLoopScope(
        builder, location, pointI,
        [&](OpBuilder &builder, Location location) {
          return pointTileI(builder, location);
        },
        [&](OpBuilder &builder, Location location) {
          Value start = pointTileI(builder, location);
          Value upper =
              cappedUpper(builder, location, start,
                          cloneInvariantLeaf(builder, iBound), rowTileSize);
          return cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                    load(builder, location, pointI), upper);
        },
        [&](OpBuilder &builder, Location location) {
          return increment(builder, location, pointI);
        },
        bodyBuilder);
  }

  void emitRectangularJ(OpBuilder &builder, Location location,
                        TileBodyBuilder bodyBuilder) {
    emitLoopScope(
        builder, location, pointJ,
        [&](OpBuilder &builder, Location location) {
          return load(builder, location, tileJ);
        },
        [&](OpBuilder &builder, Location location) {
          Value upper = add(builder, location, load(builder, location, tileJ),
                            constant(builder, location, rowTileSize));
          return cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                    load(builder, location, pointJ), upper);
        },
        [&](OpBuilder &builder, Location location) {
          return increment(builder, location, pointJ);
        },
        bodyBuilder);
  }

  void emitDiagonalJ(OpBuilder &builder, Location location,
                     TileBodyBuilder bodyBuilder) {
    emitLoopScope(
        builder, location, pointJ,
        [&](OpBuilder &builder, Location location) {
          return pointTileI(builder, location);
        },
        [&](OpBuilder &builder, Location location) {
          return cir::CmpOp::create(builder, location, cir::CmpOpKind::le,
                                    load(builder, location, pointJ),
                                    load(builder, location, pointI));
        },
        [&](OpBuilder &builder, Location location) {
          return increment(builder, location, pointJ);
        },
        bodyBuilder);
  }

  void emitKBand(OpBuilder &builder, Location location,
                 KBandBodyBuilder bodyBuilder) {
    emitLoopScope(
        builder, location, tileK,
        [&](OpBuilder &builder, Location location) {
          return constant(builder, location, 0);
        },
        [&](OpBuilder &builder, Location location) {
          Value remaining =
              subtract(builder, location, cloneInvariantLeaf(builder, kBound),
                       load(builder, location, tileK));
          return cir::CmpOp::create(
              builder, location, cir::CmpOpKind::ge, remaining,
              constant(builder, location, reductionTileSize));
        },
        [&](OpBuilder &builder, Location location) {
          return add(builder, location, load(builder, location, tileK),
                     constant(builder, location, reductionTileSize));
        },
        [&](OpBuilder &builder, Location location) {
          bodyBuilder(builder, location, true);
        });
    bodyBuilder(builder, location, false);
  }

  void emitPointK(OpBuilder &builder, Location location, bool fullTile) {
    emitLoopScope(
        builder, location, pointK,
        [&](OpBuilder &builder, Location location) {
          return load(builder, location, tileK);
        },
        [&](OpBuilder &builder, Location location) {
          Value upper =
              fullTile ? add(builder, location, load(builder, location, tileK),
                             constant(builder, location, reductionTileSize))
                       : cloneInvariantLeaf(builder, kBound);
          return cir::CmpOp::create(builder, location, cir::CmpOpKind::lt,
                                    load(builder, location, pointK), upper);
        },
        [&](OpBuilder &builder, Location location) {
          return increment(builder, location, pointK);
        },
        [&](OpBuilder &builder, Location) {
          cloneTileBody(builder, updateBody);
        });
  }

  cir::IntType type;
  cir::IntType wideType;
  cir::AllocaOp tileI;
  cir::AllocaOp tileJ;
  cir::AllocaOp tileK;
  cir::AllocaOp pointI;
  cir::AllocaOp pointJ;
  cir::AllocaOp pointK;
  const cir::LoopDomainExpr &iBound;
  const cir::LoopDomainExpr &kBound;
  ArrayRef<Operation *> scaleBody;
  ArrayRef<Operation *> updateBody;
};

static cir::AllocaOp cloneTileInduction(OpBuilder &builder,
                                        cir::AllocaOp source, StringRef name) {
  auto clone = cast<cir::AllocaOp>(builder.clone(*source.getOperation()));
  clone->setAttr("name", builder.getStringAttr(name));
  return clone;
}

static cir::AllocaOp createWideTileInduction(OpBuilder &builder,
                                             cir::AllocaOp source,
                                             cir::IntType type) {
  return cir::AllocaOp::create(builder, source.getLoc(),
                               cir::PointerType::get(type), "i.tile",
                               source.getAlignmentAttr());
}

static bool tileTriangularRecurrenceBand(cir::ForOp loop,
                                         AliasAnalysis &aliasAnalysis) {
  FailureOr<cir::ThreeLevelLoopBand> band =
      cir::analyzeThreeLevelLoopBand(loop);
  if (failed(band) || band->innerCandidates.size() != 1)
    return false;

  cir::LoopDomain &i = band->anchor;
  cir::LoopDomain &j = band->outer;
  cir::LoopDomain &k = band->innerCandidates.front();
  auto type = dyn_cast<cir::IntType>(i.stepLoad.getResult().getType());
  if (!type || !type.isSigned() || j.stepLoad.getResult().getType() != type ||
      k.stepLoad.getResult().getType() != type ||
      i.comparison.getKind() != cir::CmpOpKind::lt ||
      j.comparison.getKind() != cir::CmpOpKind::le ||
      k.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantZero(i.initial) || !isConstantZero(j.initial) ||
      !isConstantZero(k.initial) ||
      !isSimpleInduction(i.conditionLHS, i.induction) ||
      !isSimpleInduction(j.conditionLHS, j.induction) ||
      !isSimpleInduction(j.conditionRHS, i.induction) ||
      !isSimpleInduction(k.conditionLHS, k.induction) ||
      !hasInvariantLeaf(i.conditionRHS, i.loop) ||
      !hasInvariantLeaf(k.conditionRHS, i.loop) ||
      !i.increment.getNoSignedWrap() || !j.increment.getNoSignedWrap() ||
      !k.increment.getNoSignedWrap())
    return false;

  cir::LoopBandMemoryAnalysis memory =
      cir::analyzeLoopBandMemory(*band, aliasAnalysis);
  if (!memory.isSafe() || memory.recurrences.size() != 1)
    return false;
  const cir::LoopElementRecurrence &recurrence = memory.recurrences.front();
  if (recurrence.recurrenceInduction != k.induction ||
      recurrence.laneInductions.size() != 1 ||
      recurrence.laneInductions.front() != j.induction ||
      recurrence.target.subscripts.size() != 2 ||
      !isSimpleInduction(recurrence.target.subscripts[0], i.induction) ||
      !isSimpleInduction(recurrence.target.subscripts[1], j.induction))
    return false;

  auto iSetup = dyn_cast_or_null<cir::ScopeOp>(i.loop->getParentOp());
  if (!iSetup || !i.loop.getBody().hasOneBlock())
    return false;
  Block &iBody = i.loop.getBody().front();
  if (iBody.getOperations().size() != 2 || !isa<cir::YieldOp>(iBody.back()))
    return false;
  auto iteration = dyn_cast<cir::ScopeOp>(iBody.front());
  if (!iteration || !iteration.getScopeRegion().hasOneBlock())
    return false;

  Block &iterationBlock = iteration.getScopeRegion().front();
  SmallVector<Operation *, 2> phases;
  for (Operation &operation : iterationBlock.without_terminator())
    phases.push_back(&operation);
  Operation *updatePhase =
      getIterationRoot(j.loop.getOperation(), iterationBlock);
  if (phases.size() != 2 || !updatePhase ||
      !llvm::is_contained(phases, updatePhase))
    return false;
  Operation *scalePhase =
      phases.front() == updatePhase ? phases.back() : phases.front();
  auto scaleScope = dyn_cast<cir::ScopeOp>(scalePhase);
  if (!scaleScope)
    return false;

  SmallVector<cir::ForOp, 2> scaleLoops;
  scaleScope.walk(
      [&](cir::ForOp candidate) { scaleLoops.push_back(candidate); });
  if (scaleLoops.size() != 1)
    return false;
  FailureOr<cir::LoopDomain> scale =
      cir::analyzeLoopDomain(scaleLoops.front(), {i.induction});
  if (failed(scale) || scale->induction != j.induction ||
      scale->comparison.getKind() != cir::CmpOpKind::le ||
      !isConstantZero(scale->initial) ||
      !isSimpleInduction(scale->conditionLHS, j.induction) ||
      !isSimpleInduction(scale->conditionRHS, i.induction))
    return false;

  SmallVector<Operation *, 8> scaleBody;
  SmallVector<cir::AllocaOp, 3> inductions = {i.induction, j.induction,
                                              k.induction};
  if (!matchScaleBody(scale->loop, i.loop, recurrence, inductions,
                      aliasAnalysis, scaleBody))
    return false;

  SmallVector<Operation *, 8> updateBody;
  for (Operation &operation : k.loop.getBody().front().without_terminator())
    updateBody.push_back(&operation);

  OpBuilder builder(iSetup);
  cir::IntType wideType =
      cir::IntType::get(type.getContext(), 64, /*isSigned=*/true);
  cir::AllocaOp tileI = createWideTileInduction(builder, i.induction, wideType);
  cir::AllocaOp tileJ = cloneTileInduction(builder, j.induction, "j.tile");
  cir::AllocaOp tileK = cloneTileInduction(builder, k.induction, "k.tile");

  TriangularTileEmitter emitter(
      type, wideType, tileI, tileJ, tileK, i.induction, j.induction,
      k.induction, i.conditionRHS, k.conditionRHS, scaleBody, updateBody);
  emitter.emitScale(builder, i.loop.getLoc());
  emitter.emitUpdate(builder, i.loop.getLoc());
  iSetup.erase();
  return true;
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

        bool anyProfitable =
            !localities.empty() &&
            llvm::any_of(localities, [](const InterchangeLocality &locality) {
              return locality.isProfitable();
            });
        if (bandMemory.isSafe() && anyProfitable) {
          std::optional<BandRewritePlan> plan =
              matchBandRewrite(*band, localities);
          if (plan) {
            unsigned nestedPhases = applyBandRewrite(*plan);
            changed = true;
            if (emitAnalysisRemarks) {
              std::string message;
              llvm::raw_string_ostream os(message);
              os << "distributed and interchanged " << nestedPhases
                 << " nested loop " << (nestedPhases == 1 ? "phase" : "phases");
              loop.emitRemark(os.str());
            }
            tileTriangularRecurrenceBand(loop, aliasAnalysis);
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
