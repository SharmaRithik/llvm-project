//===- CIRLoopAnalysis.cpp - CIR loop analysis ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRLoopAnalysis.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace cir {

FailureOr<LoopDomainExpr>
buildLoopDomainExpr(Value value, ArrayRef<cir::AllocaOp> inductions) {
  if (auto constant = value.getDefiningOp<cir::ConstantOp>()) {
    if (!isa<cir::IntAttr>(constant.getValue()))
      return failure();
    return LoopDomainExpr(LoopDomainExpr::Kind::Constant, value);
  }

  if (auto load = value.getDefiningOp<cir::LoadOp>()) {
    if (load.getIsVolatile() || load.getMemOrder())
      return failure();
    auto alloca = load.getAddr().getDefiningOp<cir::AllocaOp>();
    if (alloca && llvm::is_contained(inductions, alloca))
      return LoopDomainExpr(LoopDomainExpr::Kind::Induction, value, alloca);
    return LoopDomainExpr(LoopDomainExpr::Kind::Symbol, value);
  }

  if (isa<BlockArgument>(value))
    return LoopDomainExpr(LoopDomainExpr::Kind::Symbol, value);

  LoopDomainExpr::Kind kind;
  Operation *operation = value.getDefiningOp();
  if (isa_and_nonnull<cir::AddOp>(operation))
    kind = LoopDomainExpr::Kind::Add;
  else if (isa_and_nonnull<cir::SubOp>(operation))
    kind = LoopDomainExpr::Kind::Sub;
  else if (isa_and_nonnull<cir::MulOp>(operation))
    kind = LoopDomainExpr::Kind::Mul;
  else if (isa_and_nonnull<cir::DivOp>(operation))
    kind = LoopDomainExpr::Kind::Div;
  else
    return failure();

  if (operation->getNumOperands() != 2)
    return failure();
  FailureOr<LoopDomainExpr> lhs =
      buildLoopDomainExpr(operation->getOperand(0), inductions);
  FailureOr<LoopDomainExpr> rhs =
      buildLoopDomainExpr(operation->getOperand(1), inductions);
  if (failed(lhs) || failed(rhs))
    return failure();

  return LoopDomainExpr(kind, value,
                        std::make_unique<LoopDomainExpr>(std::move(*lhs)),
                        std::make_unique<LoopDomainExpr>(std::move(*rhs)));
}

bool LoopDomainExpr::dependsOn(cir::AllocaOp variable) const {
  if (kind == Kind::Induction)
    return induction == variable;
  return (lhs && lhs->dependsOn(variable)) || (rhs && rhs->dependsOn(variable));
}

void LoopDomainExpr::print(llvm::raw_ostream &os) const {
  switch (kind) {
  case Kind::Constant:
    os << "constant";
    return;
  case Kind::Symbol:
    os << "symbol";
    return;
  case Kind::Induction:
    os << "induction";
    return;
  case Kind::Add:
    os << "add";
    break;
  case Kind::Sub:
    os << "sub";
    break;
  case Kind::Mul:
    os << "mul";
    break;
  case Kind::Div:
    os << "div";
    break;
  }
  os << '(';
  lhs->print(os);
  os << ',';
  rhs->print(os);
  os << ')';
}

static FailureOr<LoopDomain>
analyzeLoopDomainImpl(cir::ForOp loop,
                      ArrayRef<cir::AllocaOp> enclosingInductions) {
  if (loop.maybeGetCleanup() || !loop.getCond().hasOneBlock() ||
      !loop.getStep().hasOneBlock())
    return failure();

  Block &step = loop.getStep().front();
  if (step.getOperations().size() != 4)
    return failure();

  auto stepIt = step.begin();
  auto stepLoad = dyn_cast<cir::LoadOp>(&*stepIt++);
  auto increment = dyn_cast<cir::IncOp>(&*stepIt++);
  auto stepStore = dyn_cast<cir::StoreOp>(&*stepIt++);
  if (!stepLoad || !increment || !stepStore || !isa<cir::YieldOp>(&*stepIt) ||
      stepLoad.getIsVolatile() || stepLoad.getMemOrder() ||
      stepStore.getIsVolatile() || stepStore.getMemOrder())
    return failure();

  auto induction = stepLoad.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!induction || increment.getInput() != stepLoad.getResult() ||
      stepStore.getAddr() != induction.getAddr() ||
      stepStore.getValue() != increment.getResult())
    return failure();

  auto initialization = dyn_cast_or_null<cir::StoreOp>(loop->getPrevNode());
  if (!initialization || initialization.getAddr() != induction.getAddr() ||
      initialization.getIsVolatile() || initialization.getMemOrder())
    return failure();

  for (OpOperand &use : induction.getAddr().getUses()) {
    Operation *user = use.getOwner();
    if (auto load = dyn_cast<cir::LoadOp>(user)) {
      if (use.getOperandNumber() != cir::LoadOp::odsIndex_addr ||
          load.getIsVolatile() || load.getMemOrder())
        return failure();
      continue;
    }
    if (auto store = dyn_cast<cir::StoreOp>(user)) {
      if (use.getOperandNumber() != cir::StoreOp::odsIndex_addr ||
          (store != initialization && store != stepStore))
        return failure();
      continue;
    }
    return failure();
  }

  Block &conditionBlock = loop.getCond().front();
  auto condition = dyn_cast<cir::ConditionOp>(conditionBlock.getTerminator());
  if (!condition)
    return failure();
  auto comparison = condition.getCondition().getDefiningOp<cir::CmpOp>();
  if (!comparison || !isa<cir::IntType>(comparison.getLhs().getType()))
    return failure();

  SmallVector<cir::AllocaOp, 3> inductions(enclosingInductions);
  inductions.push_back(induction);
  FailureOr<LoopDomainExpr> initial =
      buildLoopDomainExpr(initialization.getValue(), inductions);
  FailureOr<LoopDomainExpr> conditionLHS =
      buildLoopDomainExpr(comparison.getLhs(), inductions);
  FailureOr<LoopDomainExpr> conditionRHS =
      buildLoopDomainExpr(comparison.getRhs(), inductions);
  if (failed(initial) || failed(conditionLHS) || failed(conditionRHS) ||
      (!conditionLHS->dependsOn(induction) &&
       !conditionRHS->dependsOn(induction)))
    return failure();

  return LoopDomain{loop,
                    induction,
                    initialization,
                    stepLoad,
                    increment,
                    stepStore,
                    comparison,
                    std::move(*initial),
                    std::move(*conditionLHS),
                    std::move(*conditionRHS)};
}

FailureOr<LoopDomain>
analyzeLoopDomain(cir::ForOp loop,
                  ArrayRef<cir::AllocaOp> enclosingInductions) {
  return analyzeLoopDomainImpl(loop, enclosingInductions);
}

FailureOr<TwoLevelLoopNest> analyzeTwoLevelLoopNest(cir::ForOp outerLoop) {
  FailureOr<LoopDomain> outer = analyzeLoopDomain(outerLoop);
  if (failed(outer))
    return failure();

  SmallVector<cir::ForOp, 2> immediateNestedLoops;
  outerLoop.getBody().walk([&](cir::ForOp candidate) {
    if (candidate->getParentOfType<cir::ForOp>() == outerLoop)
      immediateNestedLoops.push_back(candidate);
  });
  if (immediateNestedLoops.size() != 1)
    return failure();

  cir::AllocaOp outerInduction = outer->induction;
  FailureOr<LoopDomain> inner =
      analyzeLoopDomain(immediateNestedLoops.front(), outerInduction);
  if (failed(inner))
    return failure();

  return TwoLevelLoopNest{std::move(*outer), std::move(*inner)};
}

} // namespace cir
