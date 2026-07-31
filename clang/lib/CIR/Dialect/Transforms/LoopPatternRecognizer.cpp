//===- LoopPatternRecognizer.cpp - name known loop nest shapes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOOPPATTERNRECOGNIZER
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// A loop's induction variable is the slot its step region writes back to. The
// exit test cannot serve here, since some shapes never mention it on its own.
static Value inductionSlot(ForOp loop) {
  Value slot;
  if (!loop.getStep().hasOneBlock())
    return slot;
  for (Operation &op : loop.getStep().front())
    if (auto store = dyn_cast<StoreOp>(op))
      slot = store.getAddr();
  return slot;
}

// The value a slot holds on entry, from the store that precedes the loop.
static Value initialValue(ForOp loop, Value slot) {
  for (Operation *op = loop->getPrevNode(); op; op = op->getPrevNode())
    if (auto store = dyn_cast<StoreOp>(op); store && store.getAddr() == slot)
      return store.getValue();
  return {};
}

static Value loadedSlot(Value v) {
  auto load = v.getDefiningOp<LoadOp>();
  return load ? load.getAddr() : Value();
}

// A cond region may hold several blocks and is not verified to end in
// cir.condition, so neither can be assumed here.
static CmpOp exitTest(ForOp loop) {
  if (!loop.getCond().hasOneBlock())
    return {};
  auto condition =
      dyn_cast_or_null<ConditionOp>(loop.getCond().front().getTerminator());
  return condition ? condition.getCondition().getDefiningOp<CmpOp>() : CmpOp();
}

// The global an address is rooted at, through indexing and casts.
static llvm::StringRef rootGlobal(Value addr) {
  while (addr) {
    if (auto global = addr.getDefiningOp<GetGlobalOp>())
      return global.getName();
    if (auto element = addr.getDefiningOp<GetElementOp>())
      addr = element.getBase();
    else if (auto cast = addr.getDefiningOp<CastOp>())
      addr = cast.getSrc();
    else
      return {};
  }
  return {};
}

enum class Start { Invariant, Outer, Parameter, Other };
enum class Bound {
  Invariant,
  Outer,
  OuterPlusK,
  OuterTimesK,
  OuterProduct,
  Other
};

struct Body {
  bool reads = false;
  bool readsWhatItWrites = false;
};

// A parameter reaches its slot through the store the prologue emits.
static bool isParameterSlot(Value slot) {
  return llvm::any_of(slot.getUsers(), [](Operation *user) {
    auto store = dyn_cast<StoreOp>(user);
    return store && isa<BlockArgument>(store.getValue());
  });
}

static Start classifyStart(Value init, Value outer) {
  if (!init)
    return Start::Other;
  if (init.getDefiningOp<ConstantOp>())
    return Start::Invariant;
  Value slot = loadedSlot(init);
  if (slot == outer)
    return Start::Outer;
  if (slot && isParameterSlot(slot))
    return Start::Parameter;
  return Start::Other;
}

static Bound classifyBound(CmpOp test, Value inner, Value outer) {
  auto isConstant = [](Value v) { return v.getDefiningOp<ConstantOp>(); };
  auto isOuterScaled = [&](auto op) {
    return (loadedSlot(op.getLhs()) == outer && isConstant(op.getRhs())) ||
           (loadedSlot(op.getRhs()) == outer && isConstant(op.getLhs()));
  };
  auto isOuterTimesInner = [&](auto op) {
    return (loadedSlot(op.getLhs()) == inner &&
            loadedSlot(op.getRhs()) == outer) ||
           (loadedSlot(op.getLhs()) == outer &&
            loadedSlot(op.getRhs()) == inner);
  };

  if (test.getKind() != CmpOpKind::lt)
    return Bound::Other;

  // With the induction variable buried in a product there is no side to read
  // the limit off, which is the whole difficulty of this shape.
  if (loadedSlot(test.getLhs()) != inner) {
    auto product = test.getLhs().getDefiningOp<MulOp>();
    return product && isOuterTimesInner(product) ? Bound::OuterProduct
                                                 : Bound::Other;
  }

  Value limit = test.getRhs();
  if (isConstant(limit))
    return Bound::Invariant;
  if (loadedSlot(limit) == outer)
    return Bound::Outer;
  if (auto add = limit.getDefiningOp<AddOp>(); add && isOuterScaled(add))
    return Bound::OuterPlusK;
  if (auto mul = limit.getDefiningOp<MulOp>(); mul && isOuterScaled(mul))
    return Bound::OuterTimesK;
  return Bound::Other;
}

static Body classifyBody(ForOp loop) {
  llvm::SmallVector<llvm::StringRef> read, written;
  loop.getBody().walk([&](Operation *op) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (llvm::StringRef global = rootGlobal(load.getAddr()); !global.empty())
        read.push_back(global);
    } else if (auto store = dyn_cast<StoreOp>(op)) {
      if (llvm::StringRef global = rootGlobal(store.getAddr()); !global.empty())
        written.push_back(global);
    }
  });
  return {!read.empty(), llvm::any_of(written, [&](llvm::StringRef global) {
            return llvm::is_contained(read, global);
          })};
}

// The cirBench shapes, each told apart by the one place it differs.
static llvm::StringRef patternName(Start start, Bound bound, Body body) {
  if (bound == Bound::Invariant)
    return start == Start::Outer       ? "tri_lower_start"
           : start == Start::Parameter ? "tri_arg_start"
                                       : llvm::StringRef();
  if (start != Start::Invariant)
    return {};
  switch (bound) {
  case Bound::Outer:
    return !body.reads              ? "tri_fill"
           : body.readsWhatItWrites ? "tri_ldlt_update"
                                    : "tri_upper_bound";
  case Bound::OuterPlusK:
    return "tri_addk_bound";
  case Bound::OuterTimesK:
    return "tri_variant_2i";
  case Bound::OuterProduct:
    return "tri_mul_bound";
  default:
    return {};
  }
}

struct LoopPatternRecognizerPass
    : public impl::LoopPatternRecognizerBase<LoopPatternRecognizerPass> {
  void runOnOperation() override;
};

} // namespace

void LoopPatternRecognizerPass::runOnOperation() {
  getOperation()->walk([](ForOp outer) {
    Value outerSlot = inductionSlot(outer);
    if (!outerSlot)
      return;
    outer.getBody().walk([&](ForOp inner) {
      Value innerSlot = inductionSlot(inner);
      CmpOp test = exitTest(inner);
      if (!innerSlot || !test)
        return;
      llvm::StringRef name = patternName(
          classifyStart(initialValue(inner, innerSlot), outerSlot),
          classifyBound(test, innerSlot, outerSlot), classifyBody(inner));
      if (!name.empty())
        llvm::errs() << "[CIR Debug]: Found " << name << " kernel pattern!\n";
    });
  });
}

std::unique_ptr<Pass> mlir::createLoopPatternRecognizerPass() {
  return std::make_unique<LoopPatternRecognizerPass>();
}
