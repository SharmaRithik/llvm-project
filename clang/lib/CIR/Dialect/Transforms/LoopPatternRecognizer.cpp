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

enum class Start { Constant, Outer, Parameter, Other };

// The inner loop's exit bound, described the way the interchange pass in
// LoopOpt.cpp describes it, so the two agree on what a shape is.
struct Bound {
  enum Kind {
    Constant, // j < N
    Affine,   // j < scale * i + offset
    Product,  // j * i < N
    Other,
  };
  Kind kind = Other;
  int64_t scale = 0;
  int64_t offset = 0;
};

struct Body {
  bool reads = false;
  bool readsWhatItWrites = false;
};

// The loops directly inside this one.
static llvm::SmallVector<ForOp> immediateChildLoops(ForOp loop) {
  llvm::SmallVector<ForOp> children;
  loop.getBody().walk([&](ForOp inner) {
    if (inner != loop && inner->getParentOfType<ForOp>() == loop)
      children.push_back(inner);
  });
  return children;
}

// A parameter reaches its slot through the store the prologue emits.
static bool isParameterSlot(Value slot) {
  return llvm::any_of(slot.getUsers(), [](Operation *user) {
    auto store = dyn_cast<StoreOp>(user);
    return store && isa<BlockArgument>(store.getValue());
  });
}

// The value of an integer constant, or nothing.
static std::optional<int64_t> constantValue(Value v) {
  auto konst = v.getDefiningOp<ConstantOp>();
  auto attr = konst ? dyn_cast<cir::IntAttr>(konst.getValue()) : cir::IntAttr();
  if (!attr)
    return std::nullopt;
  return attr.getValue().getSExtValue();
}

static Start classifyStart(Value init, Value outer) {
  if (!init)
    return Start::Other;
  if (init.getDefiningOp<ConstantOp>())
    return Start::Constant;
  Value slot = loadedSlot(init);
  if (slot == outer)
    return Start::Outer;
  if (slot && isParameterSlot(slot))
    return Start::Parameter;
  return Start::Other;
}

static Bound classifyBound(CmpOp test, Value inner, Value outer) {
  // Split a constant off a binary op, in either operand order, and report the
  // other side. Returns nothing when neither side is a constant.
  auto peel = [&](auto op, Value &rest) -> std::optional<int64_t> {
    if (std::optional<int64_t> k = constantValue(op.getRhs())) {
      rest = op.getLhs();
      return k;
    }
    if (std::optional<int64_t> k = constantValue(op.getLhs())) {
      rest = op.getRhs();
      return k;
    }
    return std::nullopt;
  };

  if (test.getKind() != CmpOpKind::lt)
    return {};

  // With the induction variable buried in a product there is no side to read
  // the limit off, which is the whole difficulty of this shape.
  if (loadedSlot(test.getLhs()) != inner) {
    auto product = test.getLhs().getDefiningOp<MulOp>();
    if (!product)
      return {};
    bool bothIvs = (loadedSlot(product.getLhs()) == inner &&
                    loadedSlot(product.getRhs()) == outer) ||
                   (loadedSlot(product.getLhs()) == outer &&
                    loadedSlot(product.getRhs()) == inner);
    return bothIvs ? Bound{Bound::Product} : Bound{};
  }

  Value limit = test.getRhs();
  if (constantValue(limit))
    return {Bound::Constant};

  // scale * i + offset, with either factor absent.
  int64_t offset = 0;
  if (auto add = limit.getDefiningOp<AddOp>()) {
    Value rest;
    std::optional<int64_t> k = peel(add, rest);
    if (!k)
      return {};
    offset = *k;
    limit = rest;
  }
  int64_t scale = 1;
  if (auto mul = limit.getDefiningOp<MulOp>()) {
    Value rest;
    std::optional<int64_t> c = peel(mul, rest);
    if (!c)
      return {};
    scale = *c;
    limit = rest;
  }
  if (loadedSlot(limit) != outer)
    return {};
  return {Bound::Affine, scale, offset};
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
  if (bound.kind == Bound::Constant)
    return start == Start::Outer       ? "tri_lower_start"
           : start == Start::Parameter ? "tri_arg_start"
                                       : llvm::StringRef();
  if (start != Start::Constant)
    return {};
  if (bound.kind == Bound::Product)
    return "tri_mul_bound";
  if (bound.kind != Bound::Affine)
    return {};
  if (bound.scale != 1)
    return bound.offset == 0 ? "tri_variant_2i" : llvm::StringRef();
  if (bound.offset != 0)
    return "tri_addk_bound";
  return !body.reads              ? "tri_fill"
         : body.readsWhatItWrites ? "tri_ldlt_update"
                                  : "tri_upper_bound";
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
    // Only the loops directly inside this one. Any deeper descendant is not
    // paired with this loop, and reporting it would name a shape the pair
    // does not have.
    for (ForOp inner : immediateChildLoops(outer)) {
      Value innerSlot = inductionSlot(inner);
      CmpOp test = exitTest(inner);
      if (!innerSlot || !test)
        continue;
      llvm::StringRef name = patternName(
          classifyStart(initialValue(inner, innerSlot), outerSlot),
          classifyBound(test, innerSlot, outerSlot), classifyBody(inner));
      if (!name.empty())
        inner.emitRemark() << "Found " << name << " kernel pattern";
    }
  });
}

std::unique_ptr<Pass> mlir::createLoopPatternRecognizerPass() {
  return std::make_unique<LoopPatternRecognizerPass>();
}
