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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

bool LoopDomainExpr::isStructurallyEqual(const LoopDomainExpr &other) const {
  if (kind != other.kind)
    return false;

  switch (kind) {
  case Kind::Constant: {
    auto lhsConstant = source.getDefiningOp<cir::ConstantOp>();
    auto rhsConstant = other.source.getDefiningOp<cir::ConstantOp>();
    return lhsConstant && rhsConstant &&
           lhsConstant.getValue() == rhsConstant.getValue();
  }
  case Kind::Symbol:
    return source == other.source;
  case Kind::Induction:
    return induction == other.induction;
  case Kind::Add:
  case Kind::Sub:
  case Kind::Mul:
  case Kind::Div:
    return lhs->isStructurallyEqual(*other.lhs) &&
           rhs->isStructurallyEqual(*other.rhs);
  }
  llvm_unreachable("unknown loop domain expression kind");
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
          (store != initialization && store != stepStore &&
           loop->isAncestor(store)))
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

StringRef stringifyLoopMemoryLegality(LoopMemoryLegality result) {
  switch (result) {
  case LoopMemoryLegality::Safe:
    return "safe";
  case LoopMemoryLegality::UnsupportedOperation:
    return "unsupported operation";
  case LoopMemoryLegality::UnsupportedAddress:
    return "unsupported address";
  case LoopMemoryLegality::PotentialDependence:
    return "potential dependence";
  }
  llvm_unreachable("unknown loop memory legality result");
}

static FailureOr<Value> stripIndexCast(Value value) {
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

static FailureOr<LoopMemoryAccess>
analyzeMemoryAccess(Operation *operation, Value address, bool isWrite,
                    ArrayRef<cir::AllocaOp> inductions) {
  SmallVector<Value, 2> indices;
  Value current = address;
  while (auto element = current.getDefiningOp<cir::GetElementOp>()) {
    indices.insert(indices.begin(), element.getIndex());
    current = element.getBase();
  }

  auto getGlobal = current.getDefiningOp<cir::GetGlobalOp>();
  if (!getGlobal || getGlobal.getTls())
    return failure();

  auto base = SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
      getGlobal, getGlobal.getNameAttr());
  if (!base || base.isDeclaration() || base.getAliasee() ||
      !cir::isLocalLinkage(base.getLinkage()))
    return failure();

  SmallVector<LoopDomainExpr, 2> subscripts;
  for (Value index : indices) {
    FailureOr<Value> stripped = stripIndexCast(index);
    if (failed(stripped))
      return failure();
    FailureOr<LoopDomainExpr> expression =
        buildLoopDomainExpr(*stripped, inductions);
    if (failed(expression))
      return failure();
    subscripts.push_back(std::move(*expression));
  }

  return LoopMemoryAccess{operation, base, std::move(subscripts), isWrite};
}

static bool hasIdenticalSubscripts(const LoopMemoryAccess &lhs,
                                   const LoopMemoryAccess &rhs) {
  if (lhs.subscripts.size() != rhs.subscripts.size())
    return false;
  for (auto [lhsSubscript, rhsSubscript] :
       llvm::zip(lhs.subscripts, rhs.subscripts))
    if (!lhsSubscript.isStructurallyEqual(rhsSubscript))
      return false;
  return true;
}

static bool isInjectiveOverNest(const LoopMemoryAccess &access,
                                const TwoLevelLoopNest &nest) {
  if (access.subscripts.size() != 2)
    return false;

  bool foundOuter = false;
  bool foundInner = false;
  for (const LoopDomainExpr &subscript : access.subscripts) {
    if (subscript.getKind() != LoopDomainExpr::Kind::Induction)
      return false;
    if (subscript.getInduction() == nest.outer.induction) {
      if (foundOuter)
        return false;
      foundOuter = true;
      continue;
    }
    if (subscript.getInduction() == nest.inner.induction) {
      if (foundInner)
        return false;
      foundInner = true;
      continue;
    }
    return false;
  }
  return foundOuter && foundInner;
}

static FailureOr<LoopReduction> analyzePrivateAddReduction(
    cir::AllocaOp variable, ArrayRef<cir::LoadOp> privateLoads,
    ArrayRef<cir::StoreOp> privateStores, const TwoLevelLoopNest &nest) {
  if (nest.outer.loop->isAncestor(variable.getOperation()))
    return failure();

  SmallVector<cir::LoadOp, 2> loads;
  SmallVector<cir::StoreOp, 2> stores;
  for (cir::LoadOp load : privateLoads)
    if (load.getAddr() == variable.getAddr())
      loads.push_back(load);
  for (cir::StoreOp store : privateStores)
    if (store.getAddr() == variable.getAddr())
      stores.push_back(store);
  if (loads.size() != 1 || stores.size() != 1)
    return failure();

  cir::LoadOp load = loads.front();
  cir::StoreOp store = stores.front();
  if (load->getBlock() != store->getBlock() || !load.getResult().hasOneUse())
    return failure();

  auto addition = store.getValue().getDefiningOp<cir::AddOp>();
  if (!addition || addition->getBlock() != load->getBlock() ||
      !addition.getResult().hasOneUse() || addition.getSaturated() ||
      !isa<cir::IntType>(addition.getResult().getType()) ||
      (addition.getLhs() != load.getResult() &&
       addition.getRhs() != load.getResult()))
    return failure();

  for (OpOperand &use : variable.getAddr().getUses()) {
    Operation *user = use.getOwner();
    if (auto candidate = dyn_cast<cir::LoadOp>(user)) {
      if (use.getOperandNumber() != cir::LoadOp::odsIndex_addr ||
          candidate.getIsVolatile() || candidate.getMemOrder())
        return failure();
    } else if (auto candidate = dyn_cast<cir::StoreOp>(user)) {
      if (use.getOperandNumber() != cir::StoreOp::odsIndex_addr ||
          candidate.getIsVolatile() || candidate.getMemOrder())
        return failure();
    } else {
      return failure();
    }

    if (nest.outer.loop->isAncestor(user) && user != load.getOperation() &&
        user != store.getOperation())
      return failure();
  }

  return LoopReduction{variable, load, addition, store};
}

LoopMemoryAnalysis analyzeLoopMemory(const TwoLevelLoopNest &nest) {
  SmallVector<LoopMemoryAccess, 8> accesses;
  SmallVector<cir::LoadOp, 2> privateLoads;
  SmallVector<cir::StoreOp, 2> privateStores;
  SmallVector<cir::AllocaOp, 2> privateVariables;
  LoopMemoryLegality result = LoopMemoryLegality::Safe;
  cir::ForOp innerLoop = nest.inner.loop;
  cir::AllocaOp outerInduction = nest.outer.induction;
  cir::AllocaOp innerInduction = nest.inner.induction;
  Value outerInductionAddress = outerInduction.getAddr();
  Value innerInductionAddress = innerInduction.getAddr();
  SmallVector<cir::AllocaOp, 2> inductions = {outerInduction, innerInduction};

  WalkResult walkResult = innerLoop.getBody().walk([&](Operation *operation)
                                                       -> WalkResult {
    if (auto load = dyn_cast<cir::LoadOp>(operation)) {
      if (load.getIsVolatile() || load.getMemOrder()) {
        result = LoopMemoryLegality::UnsupportedOperation;
        return WalkResult::interrupt();
      }
      if (load.getAddr() == outerInductionAddress ||
          load.getAddr() == innerInductionAddress)
        return WalkResult::advance();
      if (auto variable = load.getAddr().getDefiningOp<cir::AllocaOp>()) {
        privateLoads.push_back(load);
        if (!llvm::is_contained(privateVariables, variable))
          privateVariables.push_back(variable);
        return WalkResult::advance();
      }
      FailureOr<LoopMemoryAccess> access =
          analyzeMemoryAccess(operation, load.getAddr(), false, inductions);
      if (failed(access)) {
        result = LoopMemoryLegality::UnsupportedAddress;
        return WalkResult::interrupt();
      }
      accesses.push_back(std::move(*access));
      return WalkResult::advance();
    }

    if (auto store = dyn_cast<cir::StoreOp>(operation)) {
      if (store.getIsVolatile() || store.getMemOrder()) {
        result = LoopMemoryLegality::UnsupportedOperation;
        return WalkResult::interrupt();
      }
      if (auto variable = store.getAddr().getDefiningOp<cir::AllocaOp>()) {
        privateStores.push_back(store);
        if (!llvm::is_contained(privateVariables, variable))
          privateVariables.push_back(variable);
        return WalkResult::advance();
      }
      FailureOr<LoopMemoryAccess> access =
          analyzeMemoryAccess(operation, store.getAddr(), true, inductions);
      if (failed(access)) {
        result = LoopMemoryLegality::UnsupportedAddress;
        return WalkResult::interrupt();
      }
      accesses.push_back(std::move(*access));
      return WalkResult::advance();
    }

    if (isa<cir::YieldOp, cir::ScopeOp, cir::GetGlobalOp, cir::GetElementOp,
            cir::PtrStrideOp>(operation))
      return WalkResult::advance();
    if (isa<cir::LoopOpInterface, cir::BreakOp, cir::ContinueOp, cir::ReturnOp>(
            operation) ||
        operation->getNumRegions() != 0 ||
        !mlir::isMemoryEffectFree(operation)) {
      result = LoopMemoryLegality::UnsupportedOperation;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return LoopMemoryAnalysis{result, std::move(accesses), {}};

  SmallVector<LoopReduction, 2> reductions;
  for (cir::AllocaOp variable : privateVariables) {
    FailureOr<LoopReduction> reduction =
        analyzePrivateAddReduction(variable, privateLoads, privateStores, nest);
    if (failed(reduction))
      return LoopMemoryAnalysis{
          LoopMemoryLegality::UnsupportedAddress, std::move(accesses), {}};
    reductions.push_back(*reduction);
  }

  for (const LoopMemoryAccess &access : accesses) {
    if (access.isWrite && !isInjectiveOverNest(access, nest))
      return LoopMemoryAnalysis{
          LoopMemoryLegality::PotentialDependence, std::move(accesses), {}};
  }

  for (auto lhs = accesses.begin(), end = accesses.end(); lhs != end; ++lhs) {
    for (auto rhs = std::next(lhs); rhs != end; ++rhs) {
      if ((!lhs->isWrite && !rhs->isWrite) || lhs->base != rhs->base)
        continue;
      if (!hasIdenticalSubscripts(*lhs, *rhs))
        return LoopMemoryAnalysis{
            LoopMemoryLegality::PotentialDependence, std::move(accesses), {}};
    }
  }

  return LoopMemoryAnalysis{LoopMemoryLegality::Safe, std::move(accesses),
                            std::move(reductions)};
}

} // namespace cir
