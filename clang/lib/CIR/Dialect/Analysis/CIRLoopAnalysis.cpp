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

#include <utility>

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

static SmallVector<cir::ForOp, 2> collectImmediateNestedLoops(cir::ForOp loop) {
  SmallVector<cir::ForOp, 2> nestedLoops;
  loop.getBody().walk([&](cir::ForOp candidate) {
    if (candidate->getParentOfType<cir::ForOp>() == loop)
      nestedLoops.push_back(candidate);
  });
  return nestedLoops;
}

FailureOr<TwoLevelLoopNest> analyzeTwoLevelLoopNest(cir::ForOp outerLoop) {
  FailureOr<LoopDomain> outer = analyzeLoopDomain(outerLoop);
  if (failed(outer))
    return failure();

  SmallVector<cir::ForOp, 2> immediateNestedLoops =
      collectImmediateNestedLoops(outerLoop);
  if (immediateNestedLoops.size() != 1)
    return failure();

  cir::AllocaOp outerInduction = outer->induction;
  FailureOr<LoopDomain> inner =
      analyzeLoopDomain(immediateNestedLoops.front(), outerInduction);
  if (failed(inner))
    return failure();

  return TwoLevelLoopNest{std::move(*outer), std::move(*inner)};
}

FailureOr<ThreeLevelLoopBand> analyzeThreeLevelLoopBand(cir::ForOp anchorLoop) {
  FailureOr<LoopDomain> anchor = analyzeLoopDomain(anchorLoop);
  if (failed(anchor))
    return failure();

  SmallVector<cir::ForOp, 2> outerCandidates;
  for (cir::ForOp loop : collectImmediateNestedLoops(anchorLoop))
    if (!collectImmediateNestedLoops(loop).empty())
      outerCandidates.push_back(loop);
  if (outerCandidates.size() != 1)
    return failure();

  FailureOr<LoopDomain> outer =
      analyzeLoopDomain(outerCandidates.front(), anchor->induction);
  if (failed(outer))
    return failure();

  SmallVector<cir::AllocaOp, 2> enclosingInductions = {anchor->induction,
                                                       outer->induction};
  SmallVector<LoopDomain, 2> innerCandidates;
  for (cir::ForOp loop : collectImmediateNestedLoops(outer->loop)) {
    FailureOr<LoopDomain> inner = analyzeLoopDomain(loop, enclosingInductions);
    if (failed(inner))
      return failure();
    innerCandidates.push_back(std::move(*inner));
  }
  if (innerCandidates.empty())
    return failure();

  return ThreeLevelLoopBand{std::move(*anchor), std::move(*outer),
                            std::move(innerCandidates)};
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

static bool isNoAliasFunctionArgument(BlockArgument argument) {
  auto function =
      dyn_cast_or_null<cir::FuncOp>(argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front() &&
         function.getArgAttr(argument.getArgNumber(), "llvm.noalias");
}

static FailureOr<BlockArgument> resolveNoAliasArgument(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (isNoAliasFunctionArgument(argument))
      return argument;
    return failure();
  }

  auto load = value.getDefiningOp<cir::LoadOp>();
  if (!load || load.getIsVolatile() || load.getMemOrder())
    return failure();
  auto slot = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!slot)
    return failure();

  cir::StoreOp definingStore;
  for (OpOperand &use : slot.getAddr().getUses()) {
    Operation *user = use.getOwner();
    if (auto candidate = dyn_cast<cir::LoadOp>(user)) {
      if (use.getOperandNumber() != cir::LoadOp::odsIndex_addr ||
          candidate.getIsVolatile() || candidate.getMemOrder())
        return failure();
      continue;
    }
    if (auto candidate = dyn_cast<cir::StoreOp>(user)) {
      if (use.getOperandNumber() != cir::StoreOp::odsIndex_addr ||
          candidate.getIsVolatile() || candidate.getMemOrder() || definingStore)
        return failure();
      definingStore = candidate;
      continue;
    }
    return failure();
  }

  if (!definingStore)
    return failure();
  auto argument = dyn_cast<BlockArgument>(definingStore.getValue());
  if (!argument || !isNoAliasFunctionArgument(argument))
    return failure();
  return argument;
}

static FailureOr<LoopMemoryAccess>
analyzeMemoryAccess(Operation *operation, Value address, bool isWrite,
                    ArrayRef<cir::AllocaOp> inductions) {
  SmallVector<Value, 2> indices;
  Value current = address;
  while (true) {
    if (auto element = current.getDefiningOp<cir::GetElementOp>()) {
      indices.insert(indices.begin(), element.getIndex());
      current = element.getBase();
      continue;
    }
    if (auto stride = current.getDefiningOp<cir::PtrStrideOp>()) {
      indices.insert(indices.begin(), stride.getStride());
      current = stride.getBase();
      continue;
    }
    break;
  }

  auto getGlobal = current.getDefiningOp<cir::GetGlobalOp>();
  LoopMemoryBase base;
  if (getGlobal) {
    if (getGlobal.getTls())
      return failure();
    base.global = SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    if (!base.global || base.global.isDeclaration() ||
        base.global.getAliasee() ||
        !cir::isLocalLinkage(base.global.getLinkage()))
      return failure();
  } else {
    FailureOr<BlockArgument> argument = resolveNoAliasArgument(current);
    if (failed(argument))
      return failure();
    base.argument = *argument;
  }

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

static bool isInjectiveOverInductions(const LoopMemoryAccess &access,
                                      ArrayRef<cir::AllocaOp> inductions) {
  if (access.subscripts.size() != inductions.size())
    return false;

  SmallVector<cir::AllocaOp, 3> found;
  for (const LoopDomainExpr &subscript : access.subscripts) {
    if (subscript.getKind() != LoopDomainExpr::Kind::Induction)
      return false;
    cir::AllocaOp induction = subscript.getInduction();
    if (!llvm::is_contained(inductions, induction) ||
        llvm::is_contained(found, induction))
      return false;
    found.push_back(induction);
  }
  return found.size() == inductions.size();
}

static bool isInjectiveOverNest(const LoopMemoryAccess &access,
                                const TwoLevelLoopNest &nest) {
  SmallVector<cir::AllocaOp, 2> inductions = {nest.outer.induction,
                                              nest.inner.induction};
  return isInjectiveOverInductions(access, inductions);
}

static std::pair<cir::LoadOp, Operation *>
matchElementRecurrence(cir::StoreOp store) {
  if (store.getIsVolatile() || store.getMemOrder() ||
      !isa<cir::FPTypeInterface>(store.getValue().getType()))
    return {};

  if (auto addition = store.getValue().getDefiningOp<cir::FAddOp>()) {
    if (addition.getFenv())
      return {};
    cir::LoadOp load;
    auto lhs = addition.getLhs().getDefiningOp<cir::LoadOp>();
    auto rhs = addition.getRhs().getDefiningOp<cir::LoadOp>();
    if (lhs && lhs.getAddr() == store.getAddr())
      load = lhs;
    if (rhs && rhs.getAddr() == store.getAddr()) {
      if (load)
        return {};
      load = rhs;
    }
    if (!load)
      return {};
    return {load, addition.getOperation()};
  }

  if (auto multiplyAdd = store.getValue().getDefiningOp<cir::FMulAddOp>()) {
    if (multiplyAdd.getFenv())
      return {};
    auto load = multiplyAdd.getC().getDefiningOp<cir::LoadOp>();
    if (!load || load.getAddr() != store.getAddr())
      return {};
    return {load, multiplyAdd.getOperation()};
  }

  return {};
}

SmallVector<LoopElementRecurrence, 2>
analyzeLoopElementRecurrences(const ThreeLevelLoopBand &band,
                              const LoopDomain &inner) {
  SmallVector<LoopElementRecurrence, 2> recurrences;
  cir::ForOp innerLoop = inner.loop;
  if (!innerLoop.getBody().hasOneBlock() ||
      llvm::none_of(band.innerCandidates, [&](const LoopDomain &candidate) {
        return candidate.loop == inner.loop;
      }))
    return recurrences;

  Block &body = innerLoop.getBody().front();
  SmallVector<cir::AllocaOp, 3> inductions = {
      band.anchor.induction, band.outer.induction, inner.induction};
  SmallVector<cir::AllocaOp, 2> targetInductions = {band.anchor.induction,
                                                    band.outer.induction};

  for (Operation &operation : body.without_terminator()) {
    auto store = dyn_cast<cir::StoreOp>(operation);
    if (!store)
      continue;
    auto [load, combiner] = matchElementRecurrence(store);
    if (!load || load.getIsVolatile() || load.getMemOrder() ||
        load->getBlock() != &body || combiner->getBlock() != &body ||
        !load.getResult().hasOneUse() || !combiner->getResult(0).hasOneUse())
      continue;

    FailureOr<LoopMemoryAccess> target =
        analyzeMemoryAccess(store, store.getAddr(), true, inductions);
    if (failed(target) || !isInjectiveOverInductions(*target, targetInductions))
      continue;
    recurrences.push_back(
        LoopElementRecurrence{std::move(*target), load, combiner, store});
  }
  return recurrences;
}

static bool isInvariantLoad(cir::LoadOp load, cir::ForOp anchorLoop) {
  auto variable = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!variable || anchorLoop->isAncestor(variable.getOperation()))
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
          anchorLoop->isAncestor(user))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static const LoopDomain *getContainingInner(const ThreeLevelLoopBand &band,
                                            Operation *operation) {
  for (const LoopDomain &inner : band.innerCandidates)
    if (inner.loop->isAncestor(operation))
      return &inner;
  return nullptr;
}

static bool isRecurrenceStore(Operation *operation,
                              ArrayRef<LoopElementRecurrence> recurrences) {
  return llvm::any_of(recurrences,
                      [&](const LoopElementRecurrence &recurrence) {
                        cir::StoreOp store = recurrence.store;
                        return store.getOperation() == operation;
                      });
}

static bool isBandWriteMapSafe(const LoopMemoryAccess &access,
                               const ThreeLevelLoopBand &band,
                               ArrayRef<LoopElementRecurrence> recurrences) {
  if (isRecurrenceStore(access.operation, recurrences))
    return true;

  if (const LoopDomain *inner = getContainingInner(band, access.operation)) {
    SmallVector<cir::AllocaOp, 2> inductions = {inner->induction,
                                                band.outer.induction};
    return isInjectiveOverInductions(access, inductions);
  }

  SmallVector<cir::AllocaOp, 2> inductions = {band.anchor.induction,
                                              band.outer.induction};
  return isInjectiveOverInductions(access, inductions);
}

static bool isPositiveOffsetFromAnchor(const LoopDomain &inner,
                                       cir::AllocaOp anchor) {
  const LoopDomainExpr &initial = inner.initial;
  if (initial.getKind() != LoopDomainExpr::Kind::Add)
    return false;

  const LoopDomainExpr *constant = nullptr;
  if (initial.getLHS()->getKind() == LoopDomainExpr::Kind::Induction &&
      initial.getLHS()->getInduction() == anchor)
    constant = initial.getRHS();
  else if (initial.getRHS()->getKind() == LoopDomainExpr::Kind::Induction &&
           initial.getRHS()->getInduction() == anchor)
    constant = initial.getLHS();
  if (!constant || constant->getKind() != LoopDomainExpr::Kind::Constant)
    return false;

  auto constantOp = constant->getSource().getDefiningOp<cir::ConstantOp>();
  auto value = constantOp ? dyn_cast<cir::IntAttr>(constantOp.getValue())
                          : cir::IntAttr{};
  auto type = value ? dyn_cast<cir::IntType>(value.getType()) : cir::IntType{};
  auto addition = initial.getSource().getDefiningOp<cir::AddOp>();
  cir::IncOp increment = inner.increment;
  return type && type.isSigned() && value.getValue().isStrictlyPositive() &&
         addition && addition.getNoSignedWrap() && increment.getNoSignedWrap();
}

static bool areBandAccessesIndependent(const LoopMemoryAccess &lhs,
                                       const LoopMemoryAccess &rhs,
                                       const ThreeLevelLoopBand &band) {
  if ((!lhs.isWrite && !rhs.isWrite) || lhs.base != rhs.base)
    return true;
  if (hasIdenticalSubscripts(lhs, rhs))
    return true;

  SmallVector<cir::AllocaOp, 2> outerInductions = {band.anchor.induction,
                                                   band.outer.induction};
  if (isInjectiveOverInductions(lhs, outerInductions) &&
      isInjectiveOverInductions(rhs, outerInductions))
    return true;

  if (lhs.subscripts.size() != rhs.subscripts.size())
    return false;

  bool foundDifference = false;
  for (auto [lhsSubscript, rhsSubscript] :
       llvm::zip(lhs.subscripts, rhs.subscripts)) {
    if (lhsSubscript.isStructurallyEqual(rhsSubscript))
      continue;
    if (lhsSubscript.getKind() != LoopDomainExpr::Kind::Induction ||
        rhsSubscript.getKind() != LoopDomainExpr::Kind::Induction)
      return false;

    cir::AllocaOp lhsInduction = lhsSubscript.getInduction();
    cir::AllocaOp rhsInduction = rhsSubscript.getInduction();
    cir::AllocaOp innerInduction =
        lhsInduction == band.anchor.induction ? rhsInduction : lhsInduction;
    if (lhsInduction != band.anchor.induction &&
        rhsInduction != band.anchor.induction)
      return false;

    auto inner =
        llvm::find_if(band.innerCandidates, [&](const LoopDomain &candidate) {
          return candidate.induction == innerInduction;
        });
    if (inner == band.innerCandidates.end() ||
        !isPositiveOffsetFromAnchor(*inner, band.anchor.induction))
      return false;
    foundDifference = true;
  }
  return foundDifference;
}

LoopBandMemoryAnalysis analyzeLoopBandMemory(const ThreeLevelLoopBand &band) {
  SmallVector<LoopMemoryAccess, 16> accesses;
  SmallVector<LoopElementRecurrence, 2> recurrences;
  for (const LoopDomain &inner : band.innerCandidates) {
    SmallVector<LoopElementRecurrence, 2> innerRecurrences =
        analyzeLoopElementRecurrences(band, inner);
    for (LoopElementRecurrence &recurrence : innerRecurrences)
      recurrences.push_back(std::move(recurrence));
  }

  SmallVector<cir::AllocaOp, 4> inductions = {band.anchor.induction,
                                              band.outer.induction};
  for (const LoopDomain &inner : band.innerCandidates)
    if (!llvm::is_contained(inductions, inner.induction))
      inductions.push_back(inner.induction);

  LoopMemoryLegality result = LoopMemoryLegality::Safe;
  cir::ForOp outerLoop = band.outer.loop;
  cir::ForOp anchorLoop = band.anchor.loop;
  WalkResult walkResult = outerLoop.getBody().walk([&](Operation *operation)
                                                       -> WalkResult {
    if (auto load = dyn_cast<cir::LoadOp>(operation)) {
      if (load.getIsVolatile() || load.getMemOrder()) {
        result = LoopMemoryLegality::UnsupportedOperation;
        return WalkResult::interrupt();
      }
      if (llvm::any_of(inductions,
                       [&](cir::AllocaOp induction) {
                         return load.getAddr() == induction.getAddr();
                       }) ||
          succeeded(resolveNoAliasArgument(load.getResult())) ||
          isInvariantLoad(load, anchorLoop))
        return WalkResult::advance();

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
      if (llvm::any_of(inductions, [&](cir::AllocaOp induction) {
            return store.getAddr() == induction.getAddr();
          }))
        return WalkResult::advance();

      FailureOr<LoopMemoryAccess> access =
          analyzeMemoryAccess(operation, store.getAddr(), true, inductions);
      if (failed(access)) {
        result = LoopMemoryLegality::UnsupportedAddress;
        return WalkResult::interrupt();
      }
      accesses.push_back(std::move(*access));
      return WalkResult::advance();
    }

    if (auto alloca = dyn_cast<cir::AllocaOp>(operation)) {
      if (llvm::is_contained(inductions, alloca))
        return WalkResult::advance();
      result = LoopMemoryLegality::UnsupportedAddress;
      return WalkResult::interrupt();
    }

    if (auto loop = dyn_cast<cir::ForOp>(operation)) {
      if (llvm::any_of(band.innerCandidates, [&](const LoopDomain &candidate) {
            return candidate.loop == loop;
          }))
        return WalkResult::advance();
      result = LoopMemoryLegality::UnsupportedOperation;
      return WalkResult::interrupt();
    }

    if (isa<cir::YieldOp, cir::ConditionOp, cir::ScopeOp, cir::GetGlobalOp,
            cir::GetElementOp, cir::PtrStrideOp>(operation))
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
    return LoopBandMemoryAnalysis{result, std::move(accesses),
                                  std::move(recurrences)};

  for (const LoopMemoryAccess &access : accesses)
    if (access.isWrite && !isBandWriteMapSafe(access, band, recurrences))
      return LoopBandMemoryAnalysis{LoopMemoryLegality::PotentialDependence,
                                    std::move(accesses),
                                    std::move(recurrences)};

  for (auto lhs = accesses.begin(), end = accesses.end(); lhs != end; ++lhs)
    for (auto rhs = std::next(lhs); rhs != end; ++rhs)
      if (!areBandAccessesIndependent(*lhs, *rhs, band))
        return LoopBandMemoryAnalysis{LoopMemoryLegality::PotentialDependence,
                                      std::move(accesses),
                                      std::move(recurrences)};

  return LoopBandMemoryAnalysis{LoopMemoryLegality::Safe, std::move(accesses),
                                std::move(recurrences)};
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
      if (succeeded(resolveNoAliasArgument(load.getResult())))
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
