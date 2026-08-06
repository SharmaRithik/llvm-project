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

static bool isCanonicalLowerTriangle(cir::TwoLevelLoopNest &nest) {
  if (nest.outer.comparison.getKind() != cir::CmpOpKind::lt ||
      nest.inner.comparison.getKind() != cir::CmpOpKind::lt ||
      !isConstantZero(nest.outer.initial) ||
      nest.inner.initial.getKind() != cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.initial.getInduction() != nest.outer.induction)
    return false;

  if (nest.outer.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.outer.conditionLHS.getInduction() != nest.outer.induction ||
      nest.outer.conditionRHS.getKind() !=
          cir::LoopDomainExpr::Kind::Constant ||
      nest.inner.conditionLHS.getKind() !=
          cir::LoopDomainExpr::Kind::Induction ||
      nest.inner.conditionLHS.getInduction() != nest.inner.induction ||
      nest.inner.conditionRHS.getKind() != cir::LoopDomainExpr::Kind::Constant)
    return false;

  cir::IntAttr outerBound = getIntegerConstant(nest.outer.conditionRHS);
  cir::IntAttr innerBound = getIntegerConstant(nest.inner.conditionRHS);
  return outerBound && innerBound && outerBound == innerBound &&
         nest.outer.stepLoad.getResult().getType() ==
             nest.inner.stepLoad.getResult().getType();
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

static bool isAffineOffsetUpperTriangle(cir::TwoLevelLoopNest &nest) {
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
    return false;

  const cir::LoopDomainExpr &outerBound = nest.outer.conditionRHS;
  if (outerBound.getKind() != cir::LoopDomainExpr::Kind::Sub)
    return false;
  cir::IntAttr extent = getIntegerConstant(*outerBound.getLHS());
  cir::IntAttr outerOffset = getIntegerConstant(*outerBound.getRHS());
  cir::IntAttr innerOffset =
      getAddedConstant(nest.inner.conditionRHS, nest.outer.induction);
  if (!extent || !outerOffset || !innerOffset || outerOffset != innerOffset ||
      extent.getType() != outerOffset.getType())
    return false;

  auto type = dyn_cast<cir::IntType>(extent.getType());
  if (!type || !type.isSigned() ||
      nest.outer.stepLoad.getResult().getType() != type ||
      nest.inner.stepLoad.getResult().getType() != type)
    return false;

  const llvm::APInt &extentValue = extent.getValue();
  const llvm::APInt &offsetValue = outerOffset.getValue();
  return offsetValue.isStrictlyPositive() && extentValue.sgt(offsetValue);
}

static bool isScaledUpperTriangle(cir::TwoLevelLoopNest &nest) {
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
    return false;

  const cir::LoopDomainExpr &outerBound = nest.outer.conditionRHS;
  if (outerBound.getKind() != cir::LoopDomainExpr::Kind::Div)
    return false;
  cir::IntAttr numerator = getIntegerConstant(*outerBound.getLHS());
  cir::IntAttr divisor = getIntegerConstant(*outerBound.getRHS());
  cir::IntAttr coefficient =
      getMultipliedConstant(nest.inner.conditionRHS, nest.outer.induction);
  if (!numerator || !divisor || !coefficient ||
      numerator.getType() != divisor.getType() ||
      numerator.getType() != coefficient.getType())
    return false;

  auto type = dyn_cast<cir::IntType>(numerator.getType());
  if (!type || !type.isSigned() ||
      nest.outer.stepLoad.getResult().getType() != type ||
      nest.inner.stepLoad.getResult().getType() != type)
    return false;

  const llvm::APInt &numeratorValue = numerator.getValue();
  const llvm::APInt &divisorValue = divisor.getValue();
  const llvm::APInt &coefficientValue = coefficient.getValue();
  if (!numeratorValue.isStrictlyPositive() ||
      !divisorValue.isStrictlyPositive() ||
      !coefficientValue.isStrictlyPositive())
    return false;

  llvm::APInt upperBound = numeratorValue.sdiv(divisorValue);
  llvm::APInt one(upperBound.getBitWidth(), 1);
  if (!upperBound.sgt(one))
    return false;
  bool overflow = false;
  (void)(upperBound - one).smul_ov(coefficientValue, overflow);
  return !overflow;
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

static bool isProductBoundTriangle(cir::TwoLevelLoopNest &nest) {
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
    return false;

  cir::IntAttr extent = getIntegerConstant(nest.outer.conditionRHS);
  if (!extent)
    return false;
  auto type = dyn_cast<cir::IntType>(extent.getType());
  if (!type || !type.isSigned() ||
      nest.outer.stepLoad.getResult().getType() != type ||
      nest.inner.stepLoad.getResult().getType() != type)
    return false;

  llvm::APInt one(extent.getValue().getBitWidth(), 1);
  if (!extent.getValue().sgt(one))
    return false;
  llvm::APInt two(extent.getValue().getBitWidth(), 2);
  bool overflow = false;
  (void)(extent.getValue() - one).smul_ov(two, overflow);
  return !overflow;
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
  Operation *innerInitialOperation =
      nest.inner.initial.getSource().getDefiningOp();
  if (!innerInitialOperation || &*operation++ != innerInitialOperation ||
      &*operation++ != nest.inner.initialization.getOperation() ||
      &*operation++ != innerLoop.getOperation() ||
      !isa<cir::YieldOp>(&*operation))
    return {};

  if (nest.inner.induction->getNumOperands() != 0 ||
      nest.outer.initialization->getBlock() != outerLoop->getBlock())
    return {};
  return scope;
}

static void interchangeLoopStructure(cir::TwoLevelLoopNest &nest,
                                     cir::ScopeOp scope) {
  cir::ForOp outerLoop = nest.outer.loop;
  cir::ForOp innerLoop = nest.inner.loop;
  Operation *innerInitialOperation =
      nest.inner.initial.getSource().getDefiningOp();
  Block *parent = outerLoop->getBlock();
  Block &outerBody = outerLoop.getBody().front();
  Block &innerBody = innerLoop.getBody().front();

  nest.inner.induction->moveBefore(outerLoop);
  innerInitialOperation->moveBefore(outerLoop);
  nest.inner.initialization->moveBefore(outerLoop);
  innerLoop->moveBefore(outerLoop);
  scope.erase();

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

static LogicalResult
interchangeCanonicalUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (!isCanonicalUpperTriangle(nest))
    return failure();

  cir::ScopeOp scope = getPerfectNestScope(nest);
  if (!scope)
    return failure();

  auto outerInitialConstant =
      nest.outer.initial.getSource().getDefiningOp<cir::ConstantOp>();
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

  interchangeLoopStructure(nest, scope);
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

static LogicalResult
interchangeCanonicalLowerTriangle(cir::TwoLevelLoopNest &nest) {
  if (!isCanonicalLowerTriangle(nest))
    return failure();

  cir::ScopeOp scope = getPerfectNestScope(nest);
  if (!scope)
    return failure();

  Operation *oldInnerInitial = nest.inner.initial.getSource().getDefiningOp();
  Operation *oldOuterBound =
      nest.outer.conditionRHS.getSource().getDefiningOp();

  nest.inner.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        nest.outer.initial.getSource());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.outer.comparison);
  auto newInnerBound =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newInnerBound->setLoc(nest.outer.comparison.getLoc());
  nest.outer.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newInnerBound);
  nest.outer.comparison.setKind(cir::CmpOpKind::le);

  interchangeLoopStructure(nest, scope);
  if (oldInnerInitial->use_empty())
    oldInnerInitial->erase();
  if (oldOuterBound->use_empty())
    oldOuterBound->erase();
  return success();
}

static LogicalResult
interchangeAffineOffsetUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (!isAffineOffsetUpperTriangle(nest))
    return failure();

  cir::ScopeOp scope = getPerfectNestScope(nest);
  if (!scope)
    return failure();

  cir::IntAttr extent = getIntegerConstant(*nest.outer.conditionRHS.getLHS());
  cir::IntAttr offset = getIntegerConstant(*nest.outer.conditionRHS.getRHS());
  auto type = cast<cir::IntType>(extent.getType());

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  llvm::APInt one(extent.getValue().getBitWidth(), 1);
  auto newOuterBound =
      cir::ConstantOp::create(builder, nest.inner.comparison.getLoc(),
                              cir::IntAttr::get(type, extent.getValue() - one));
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  eraseDeadDomainExpression(nest.inner.conditionRHS);

  interchangeLoopStructure(nest, scope);
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
  return success();
}

static LogicalResult
interchangeScaledUpperTriangle(cir::TwoLevelLoopNest &nest) {
  if (!isScaledUpperTriangle(nest))
    return failure();

  cir::ScopeOp scope = getPerfectNestScope(nest);
  if (!scope)
    return failure();

  auto oldOuterInitial =
      nest.outer.initial.getSource().getDefiningOp<cir::ConstantOp>();
  const cir::LoopDomainExpr &outerBound = nest.outer.conditionRHS;
  cir::IntAttr numerator = getIntegerConstant(*outerBound.getLHS());
  cir::IntAttr divisor = getIntegerConstant(*outerBound.getRHS());
  cir::IntAttr coefficient =
      getMultipliedConstant(nest.inner.conditionRHS, nest.outer.induction);
  auto type = cast<cir::IntType>(numerator.getType());
  llvm::APInt one(numerator.getValue().getBitWidth(), 1);
  llvm::APInt upperBound = numerator.getValue().sdiv(divisor.getValue());
  llvm::APInt newOuterBoundValue = (upperBound - one) * coefficient.getValue();

  OpBuilder builder(nest.outer.loop.getContext());
  builder.setInsertionPoint(nest.inner.comparison);
  auto newOuterBound =
      cir::ConstantOp::create(builder, nest.inner.comparison.getLoc(),
                              cir::IntAttr::get(type, newOuterBoundValue));
  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_rhs, newOuterBound);
  eraseDeadDomainExpression(nest.inner.conditionRHS);

  interchangeLoopStructure(nest, scope);
  builder.setInsertionPoint(nest.outer.initialization);
  Location location = nest.outer.initialization.getLoc();
  auto newOuterValue =
      cast<cir::LoadOp>(builder.clone(*nest.inner.stepLoad.getOperation()));
  newOuterValue->setLoc(location);
  auto coefficientValue =
      cir::ConstantOp::create(builder, location, coefficient);
  auto quotient = cir::DivOp::create(builder, location, type, newOuterValue,
                                     coefficientValue);
  auto newInnerInitial = cir::IncOp::create(builder, location, quotient,
                                            /*noSignedWrap=*/false);
  nest.outer.initialization->setOperand(cir::StoreOp::odsIndex_value,
                                        newInnerInitial);
  if (oldOuterInitial->use_empty())
    oldOuterInitial.erase();
  return success();
}

static LogicalResult
interchangeProductBoundTriangle(cir::TwoLevelLoopNest &nest) {
  if (!isProductBoundTriangle(nest))
    return failure();

  cir::ScopeOp scope = getPerfectNestScope(nest);
  if (!scope)
    return failure();

  cir::IntAttr extent = getIntegerConstant(nest.outer.conditionRHS);
  auto type = cast<cir::IntType>(extent.getType());
  const cir::LoopDomainExpr *productLHS = nest.inner.conditionLHS.getLHS();
  const cir::LoopDomainExpr *productRHS = nest.inner.conditionLHS.getRHS();
  const cir::LoopDomainExpr *innerInduction =
      productLHS->getInduction() == nest.inner.induction ? productLHS
                                                         : productRHS;
  Operation *oldOuterBound =
      nest.outer.conditionRHS.getSource().getDefiningOp();

  nest.inner.comparison->setOperand(cir::CmpOp::odsIndex_lhs,
                                    innerInduction->getSource());
  eraseDeadDomainExpression(nest.inner.conditionLHS);

  interchangeLoopStructure(nest, scope);
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
  if (oldOuterBound->use_empty())
    oldOuterBound->erase();
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

      if (!memory.isSafe() || !profitable)
        continue;

      LogicalResult transformed = interchangeCanonicalUpperTriangle(*nest);
      if (failed(transformed))
        transformed = interchangeCanonicalLowerTriangle(*nest);
      if (failed(transformed))
        transformed = interchangeAffineOffsetUpperTriangle(*nest);
      if (failed(transformed))
        transformed = interchangeScaledUpperTriangle(*nest);
      if (failed(transformed))
        transformed = interchangeProductBoundTriangle(*nest);
      if (failed(transformed))
        continue;

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
