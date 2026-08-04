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

// The object an address is rooted at, through indexing and casts. A global
// resolves to its definition, a stack slot to its alloca, and an array reached
// through a pointer parameter to the slot holding that pointer.
// Returns null unless the address indexes into something, so a plain scalar
// slot such as an induction variable is not mistaken for an array access.
static Operation *rootObject(Value addr) {
  bool indexed = false;
  while (addr) {
    if (!indexed && (addr.getDefiningOp<GetElementOp>() ||
                     addr.getDefiningOp<PtrStrideOp>()))
      indexed = true;
    if (auto global = addr.getDefiningOp<GetGlobalOp>()) {
      if (!indexed)
        return nullptr;
      // Resolve to the definition. Two references to one global are separate
      // operations, so the reference itself is not a stable identity.
      auto mod = global->getParentOfType<mlir::ModuleOp>();
      auto def =
          mod ? mod.lookupSymbol<GlobalOp>(global.getName()) : GlobalOp();
      return def ? def.getOperation() : nullptr;
    }
    if (auto alloca = addr.getDefiningOp<AllocaOp>())
      return indexed ? alloca.getOperation() : nullptr;
    if (auto element = addr.getDefiningOp<GetElementOp>())
      addr = element.getBase();
    else if (auto stride = addr.getDefiningOp<PtrStrideOp>())
      addr = stride.getBase();
    else if (auto cast = addr.getDefiningOp<CastOp>())
      addr = cast.getSrc();
    else if (auto load = addr.getDefiningOp<LoadOp>())
      addr = load.getAddr(); // an array handle held in a slot
    else
      return nullptr;
  }
  return nullptr;
}

// The inner loop's start value.
struct Start {
  enum Kind {
    Constant,  // j = 0
    Outer,     // j = i + offset
    Parameter, // j = lo, a function argument
    Other,
  };
  Kind kind = Other;
  int64_t offset = 0;
};

// The inner loop's exit bound, described the way the interchange pass in
// LoopOpt.cpp describes it, so the two agree on what a shape is.
struct Bound {
  enum Kind {
    Invariant, // j < N, a constant or anything the nest does not write
    Affine,    // j < scale * i + offset
    Product,   // j * i < N
    Other,
  };
  Kind kind = Other;
  int64_t scale = 0;
  int64_t offset = 0;
  bool orEqual = false; // the test is <= rather than <
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

// True when nothing in the nest stores to the slot this value was loaded from,
// so the value is the same on every iteration.
static bool isInvariantInNest(Value v, ForOp nest) {
  if (v.getDefiningOp<ConstantOp>())
    return true;
  Value slot = loadedSlot(v);
  if (!slot || !slot.getDefiningOp<AllocaOp>())
    return false;
  bool written = false;
  nest->walk([&](StoreOp store) {
    if (store.getAddr() == slot)
      written = true;
  });
  return !written;
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
    return {};
  if (init.getDefiningOp<ConstantOp>())
    return {Start::Constant};
  // j = i + offset, in either operand order.
  int64_t offset = 0;
  if (auto add = init.getDefiningOp<AddOp>()) {
    if (std::optional<int64_t> k = constantValue(add.getRhs())) {
      offset = *k;
      init = add.getLhs();
    } else if (std::optional<int64_t> k = constantValue(add.getLhs())) {
      offset = *k;
      init = add.getRhs();
    } else {
      return {};
    }
  }
  Value slot = loadedSlot(init);
  if (slot == outer)
    return {Start::Outer, offset};
  if (slot && offset == 0 && isParameterSlot(slot))
    return {Start::Parameter};
  return {};
}

static Bound classifyBound(CmpOp test, Value inner, Value outer, ForOp nest) {
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

  bool orEqual = test.getKind() == CmpOpKind::le;
  if (test.getKind() != CmpOpKind::lt && !orEqual)
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
    return bothIvs ? Bound{Bound::Product, 0, 0, orEqual} : Bound{};
  }

  Value limit = test.getRhs();

  // scale * i + offset, with either factor absent.
  int64_t offset = 0;
  if (auto add = limit.getDefiningOp<AddOp>()) {
    Value rest;
    std::optional<int64_t> k = peel(add, rest);
    if (k) {
      offset = *k;
      limit = rest;
    }
  }
  int64_t scale = 1;
  if (auto mul = limit.getDefiningOp<MulOp>()) {
    Value rest;
    std::optional<int64_t> c = peel(mul, rest);
    if (c) {
      scale = *c;
      limit = rest;
    }
  }
  if (loadedSlot(limit) == outer)
    return {Bound::Affine, scale, offset, orEqual};

  // Not the outer variable, so the bound is a fixed limit as long as nothing
  // in the nest writes it. A parameter such as n qualifies, not just a literal.
  if (isInvariantInNest(test.getRhs(), nest))
    return {Bound::Invariant, 0, 0, orEqual};
  return {};
}

static Body classifyBody(ForOp loop) {
  llvm::SmallVector<Operation *> read, written;
  loop.getBody().walk([&](Operation *op) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (Operation *base = rootObject(load.getAddr()))
        read.push_back(base);
    } else if (auto store = dyn_cast<StoreOp>(op)) {
      if (Operation *base = rootObject(store.getAddr()))
        written.push_back(base);
    }
  });
  return {!read.empty(), llvm::any_of(written, [&](Operation *base) {
            return llvm::is_contained(read, base);
          })};
}

// The shapes, each told apart by the one place it differs. The eight cirBench
// names are kept as they are, the rest describe the shape.
static llvm::StringRef patternName(Start start, Bound bound, Body body) {
  if (bound.kind == Bound::Product)
    return start.kind == Start::Constant ? "tri_mul_bound" : llvm::StringRef();

  // A bound that does not name the outer variable makes the triangle, if there
  // is one, come from the start.
  if (bound.kind == Bound::Invariant) {
    if (start.kind == Start::Parameter)
      return "tri_arg_start";
    if (start.kind != Start::Outer)
      return {};
    return start.offset == 0 ? "tri_lower_start" : "tri_offset_start";
  }

  if (bound.kind != Bound::Affine)
    return {};
  // A start that also names the outer variable is a band, not a triangle.
  if (start.kind != Start::Constant)
    return {};
  if (bound.orEqual)
    return bound.scale == 1 && bound.offset == 0 ? "tri_upper_bound_le"
                                                 : llvm::StringRef();
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
      llvm::StringRef name =
          patternName(classifyStart(initialValue(inner, innerSlot), outerSlot),
                      classifyBound(test, innerSlot, outerSlot, outer),
                      classifyBody(inner));
      if (!name.empty())
        mlir::emitRemark(inner.getLoc())
            << "Found " << name << " kernel pattern";
    }
  });
}

std::unique_ptr<Pass> mlir::createLoopPatternRecognizerPass() {
  return std::make_unique<LoopPatternRecognizerPass>();
}
