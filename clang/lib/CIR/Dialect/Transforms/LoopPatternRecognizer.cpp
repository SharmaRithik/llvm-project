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

// The step region identifies the induction variable. The exit test cannot,
// since a product test never names the variable on its own.
static Value inductionSlot(ForOp loop) {
  Value slot;
  if (!loop.getStep().hasOneBlock())
    return slot;
  for (Operation &op : loop.getStep().front())
    if (auto store = dyn_cast<StoreOp>(op))
      slot = store.getAddr();
  return slot;
}

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
// cir.condition, so neither can be assumed.
static CmpOp exitTest(ForOp loop) {
  if (!loop.getCond().hasOneBlock())
    return {};
  auto condition =
      dyn_cast_or_null<ConditionOp>(loop.getCond().front().getTerminator());
  return condition ? condition.getCondition().getDefiningOp<CmpOp>() : CmpOp();
}

// The object an array access is rooted at. Null unless the address indexes
// into something, so an induction variable load is not taken for an access.
static Operation *rootObject(Value addr) {
  bool indexed = false;
  while (addr) {
    if (!indexed && (addr.getDefiningOp<GetElementOp>() ||
                     addr.getDefiningOp<PtrStrideOp>()))
      indexed = true;
    if (auto global = addr.getDefiningOp<GetGlobalOp>()) {
      if (!indexed)
        return nullptr;
      // Two references to one global are separate operations, so resolve to
      // the definition to get a stable identity.
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

// Described the way LoopOpt.cpp describes a bound, so the two passes agree on
// what a shape is.
struct Bound {
  enum Kind {
    Invariant, // j < N, anything the nest does not write
    Affine,    // j < scale * i + offset
    Product,   // j * i < N
    Other,
  };
  Kind kind = Other;
  int64_t scale = 0;
  int64_t offset = 0;
  bool orEqual = false;
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
  // Split a constant off, in either operand order, and report the other side.
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

  // A product names neither variable alone, so there is no side to read a
  // limit off.
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

  // A fixed limit, which a parameter such as n satisfies, not only a literal.
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

// The eight cirBench names are kept as they are, the rest describe the shape.
static llvm::StringRef patternName(Start start, Bound bound, Body body) {
  if (bound.kind == Bound::Product)
    return start.kind == Start::Constant ? "tri_mul_bound" : llvm::StringRef();

  // With no outer variable in the bound, any triangle comes from the start.
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
  getOperation()->walk([](ForOp loop) {
    Value slot = inductionSlot(loop);
    CmpOp test = exitTest(loop);
    if (!slot || !test)
      return;
    // Search outward. A triangular bound can be separated from the loop it
    // names by other loops, as in for i / for j / for k < i. Stop at the first
    // enclosing loop that explains the shape, so each loop reports once.
    for (ForOp outer = loop->getParentOfType<ForOp>(); outer;
         outer = outer->getParentOfType<ForOp>()) {
      Value outerSlot = inductionSlot(outer);
      if (!outerSlot)
        continue;
      llvm::StringRef name = patternName(
          classifyStart(initialValue(loop, slot), outerSlot),
          classifyBound(test, slot, outerSlot, outer), classifyBody(loop));
      if (!name.empty()) {
        mlir::emitRemark(loop.getLoc())
            << "Found " << name << " kernel pattern";
        return;
      }
    }
  });
}

std::unique_ptr<Pass> mlir::createLoopPatternRecognizerPass() {
  return std::make_unique<LoopPatternRecognizerPass>();
}
