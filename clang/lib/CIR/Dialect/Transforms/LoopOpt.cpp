//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// High level loop optimizations on structured CIR.
// Runs while loops are still cir.for and array accesses still carry shape.
// The implemented transform is interchange of the innermost perfectly nested
// pair when it is provably legal and profitable.

#include "PassDetail.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TimeProfiler.h"
#include <optional>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOOPOPT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// Loop nest recognition

// A recognized canonical counted loop.
// The induction variable lives in a cir.alloca slot named ivSlot.
struct CountedLoop {
  cir::ForOp forOp;
  mlir::Value ivSlot;
};

// Defined below. The comparison a counted loop exits on, or null.
static cir::CmpOp getExitCmp(cir::ForOp forOp);

// Defined below. True if a region has an op the dependence analysis cannot
// model. allowAnyStore lets the body write any array. When false only a store
// to ivSlot is allowed, which is all a swappable control region may contain.
static bool regionHasUnmodeledEffect(mlir::Region &region, mlir::Value ivSlot,
                                     bool allowAnyStore);

// Recognize the canonical counted loop idiom.
// Returns nullopt for anything non-canonical so the pass leaves it untouched.
static std::optional<CountedLoop> recognizeCountedLoop(cir::ForOp forOp) {
  mlir::Region &cond = forOp.getCond();
  if (!getExitCmp(forOp))
    return std::nullopt;

  // The induction variable is the slot the step region writes back to. Reading
  // it off the exit test instead would miss a test that never names the slot on
  // its own, such as one comparing a product of two variables.
  mlir::Region &step = forOp.getStep();
  if (!step.hasOneBlock())
    return std::nullopt;
  mlir::Value ivSlot;
  for (auto store : step.front().getOps<cir::StoreOp>()) {
    if (ivSlot)
      return std::nullopt;
    ivSlot = store.getAddr();
  }
  if (!ivSlot || !ivSlot.getDefiningOp<cir::AllocaOp>())
    return std::nullopt;

  // interchange swaps the cond and step regions wholesale, so they must carry
  // no side effect beyond the induction compare and update. The cond may only
  // read, so it must not write at all. The step may write only the induction
  // slot. Anything else would be relocated and change behavior.
  if (regionHasUnmodeledEffect(cond, /*ivSlot=*/{}, /*allowAnyStore=*/false) ||
      regionHasUnmodeledEffect(step, ivSlot, /*allowAnyStore=*/false))
    return std::nullopt;

  // A cleanup region runs on every per iteration exit edge. The swap does not
  // move it, so it would end up guarding a loop with a different trip count.
  if (forOp.maybeGetCleanup())
    return std::nullopt;

  bool bodyWritesIv = false;
  forOp.getBody().walk([&](cir::StoreOp store) {
    if (store.getAddr() == ivSlot)
      bodyWritesIv = true;
  });
  if (bodyWritesIv)
    return std::nullopt;

  return CountedLoop{forOp, ivSlot};
}

// Loops whose nearest enclosing cir.for is forOp.
static llvm::SmallVector<cir::ForOp> immediateChildLoops(cir::ForOp forOp) {
  llvm::SmallVector<cir::ForOp> children;
  forOp.getBody().walk([&](cir::ForOp inner) {
    if (inner != forOp && inner->getParentOfType<cir::ForOp>() == forOp)
      children.push_back(inner);
  });
  return children;
}

// The perfect nest chain of counted loops rooted at top, outer to inner.
static llvm::SmallVector<CountedLoop> collectPerfectNest(cir::ForOp top) {
  llvm::SmallVector<CountedLoop> nest;
  cir::ForOp cur = top;
  while (cur) {
    std::optional<CountedLoop> rec = recognizeCountedLoop(cur);
    if (!rec)
      break;
    nest.push_back(*rec);
    llvm::SmallVector<cir::ForOp> children = immediateChildLoops(cur);
    if (children.size() != 1)
      break;
    cur = children.front();
  }
  return nest;
}

// Memory access analysis for interchange legality

// How a single array subscript relates to the two loops being interchanged.
enum class IdxTag { P, Q, Inv, Unknown };

// Peel cir.get_element and cir.ptr_stride off an address.
// Collects subscripts with the contiguous innermost dimension first.
// Returns the root base pointer.
static mlir::Value peelAddress(mlir::Value addr,
                               llvm::SmallVectorImpl<mlir::Value> &indices) {
  while (true) {
    if (auto ge = addr.getDefiningOp<cir::GetElementOp>()) {
      indices.push_back(ge.getIndex());
      addr = ge.getBase();
      continue;
    }
    if (auto ps = addr.getDefiningOp<cir::PtrStrideOp>()) {
      indices.push_back(ps.getStride());
      addr = ps.getBase();
      continue;
    }
    break;
  }
  return addr;
}

// Classify a subscript against the two loop IV slots p and q.
// body is used to check that any other index variable is loop invariant.
static IdxTag classifyIndex(mlir::Value idx, mlir::Value p, mlir::Value q,
                            mlir::Region &body) {
  mlir::Value v = idx;
  while (auto cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  if (v.getDefiningOp<cir::ConstantOp>())
    return IdxTag::Inv;
  auto load = v.getDefiningOp<cir::LoadOp>();
  if (!load)
    return IdxTag::Unknown;
  mlir::Value slot = load.getAddr();
  if (slot == p)
    return IdxTag::P;
  if (slot == q)
    return IdxTag::Q;
  // Only a stack slot can be tracked reliably so anything else is Unknown.
  if (!slot.getDefiningOp<cir::AllocaOp>())
    return IdxTag::Unknown;
  // Some other variable is invariant only if it is never written in body.
  bool written = false;
  body.walk([&](cir::StoreOp store) {
    if (store.getAddr() == slot)
      written = true;
  });
  return written ? IdxTag::Unknown : IdxTag::Inv;
}

// The store that fills a single init parameter slot, when base is the loaded
// value of one. The slot must hold the argument for the whole function and
// never have its own address taken, or the load could see something else.
static cir::StoreOp paramSlotInit(mlir::Value base) {
  auto load = base.getDefiningOp<cir::LoadOp>();
  auto slot =
      load ? load.getAddr().getDefiningOp<cir::AllocaOp>() : cir::AllocaOp();
  if (!slot)
    return {};
  cir::StoreOp init;
  for (mlir::Operation *user : slot->getUsers()) {
    if (mlir::isa<cir::LoadOp>(user))
      continue;
    auto store = mlir::dyn_cast<cir::StoreOp>(user);
    if (!store || init)
      return {};
    init = store;
  }
  auto arg = init ? mlir::dyn_cast<mlir::BlockArgument>(init.getValue())
                  : mlir::BlockArgument();
  auto func = slot->getParentOfType<cir::FuncOp>();
  if (arg && func && arg.getOwner() == &func.getBody().front())
    return init;
  return {};
}

// True when the parameter behind a slot init carries the noalias promise.
static bool isNoAliasParam(cir::StoreOp init) {
  auto arg = mlir::cast<mlir::BlockArgument>(init.getValue());
  auto func = init->getParentOfType<cir::FuncOp>();
  return func && func.getArgAttr(arg.getArgNumber(), "llvm.noalias");
}

// A stable identity for a base object used to reason about aliasing. Distinct
// allocas and distinct globals denote distinct objects. Returns the defining
// alloca or the resolved global definition, or null when the base has unknown
// aliasing such as a pointer parameter. With assumeParamsDistinct the caller
// promises a runtime disjointness test around the nest, so a parameter slot
// init becomes an identity without the restrict qualifier.
static mlir::Operation *baseObjectId(mlir::Value base,
                                     bool assumeParamsDistinct = false) {
  if (auto a = base.getDefiningOp<cir::AllocaOp>())
    return a.getOperation();
  if (auto g = base.getDefiningOp<cir::GetGlobalOp>()) {
    // Resolve alias chains so aliased globals map to one identity.
    llvm::StringRef name = g.getName();
    auto mod = g->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return nullptr;
    cir::GlobalOp gv;
    for (int hop = 0; hop < 16; ++hop) {
      gv = mod.lookupSymbol<cir::GlobalOp>(name);
      if (!gv)
        return nullptr; // unknown definition so be conservative
      mlir::FlatSymbolRefAttr aa = gv.getAliaseeAttr();
      if (!aa)
        break;
      name = aa.getValue();
    }
    return gv ? gv.getOperation() : nullptr;
  }
  // A restrict parameter reaches storage nothing else in the function reaches,
  // so its slot init is a sound identity. What the pointer points at is a
  // different object from the slot holding the pointer, so the init store
  // stands in as the identity of the pointee.
  if (cir::StoreOp init = paramSlotInit(base))
    if (assumeParamsDistinct || isNoAliasParam(init))
      return init.getOperation();
  // Anything else such as a pointer parameter has unknown aliasing.
  return nullptr;
}

// A local scalar the body only accumulates into, under an operation the
// iteration order cannot affect. Returns that operation so the caller can drop
// its wrap flags, which a reordered partial sum can no longer promise.
static mlir::Operation *orderFreeReduction(mlir::Operation *slotOp,
                                           cir::ForOp nest) {
  auto alloca = mlir::dyn_cast<cir::AllocaOp>(slotOp);
  if (!alloca)
    return nullptr;
  mlir::Value slot = alloca.getResult();
  // A slot whose address is taken could be read anywhere.
  for (mlir::Operation *user : slot.getUsers())
    if (!mlir::isa<cir::LoadOp, cir::StoreOp>(user))
      return nullptr;

  // Count over the whole nest, not just the innermost body. A cond or step
  // region that reads the accumulator would make the swap change the iteration
  // space itself.
  cir::StoreOp store;
  unsigned loads = 0, stores = 0;
  nest->walk([&](mlir::Operation *op) {
    if (auto l = mlir::dyn_cast<cir::LoadOp>(op)) {
      if (l.getAddr() == slot)
        ++loads;
    } else if (auto s = mlir::dyn_cast<cir::StoreOp>(op)) {
      if (s.getAddr() != slot)
        return;
      ++stores;
      store = s;
    }
  });
  if (loads != 1 || stores != 1)
    return nullptr;

  // The stored value must combine the loaded value with something else.
  mlir::Operation *combine = store.getValue().getDefiningOp();
  if (!mlir::isa_and_nonnull<cir::AddOp, cir::MulOp>(combine))
    return nullptr;
  // The combined value may not escape. Storing it anywhere else makes that
  // location carry a partial sum, whose value depends on the order.
  if (!combine->hasOneUse() || *combine->user_begin() != store.getOperation())
    return nullptr;
  // Saturating addition is commutative but not associative, so a reordered
  // partial sum can clamp where the original did not.
  if (auto add = mlir::dyn_cast<cir::AddOp>(combine); add && add.getSaturated())
    return nullptr;
  bool accumulates = llvm::any_of(combine->getOperands(), [&](mlir::Value v) {
    auto load = v.getDefiningOp<cir::LoadOp>();
    return load && load.getAddr() == slot;
  });
  return accumulates ? combine : nullptr;
}

// Drop the wrap flags a reordered reduction can no longer promise.
static void relaxWrapFlags(llvm::ArrayRef<mlir::Operation *> reductions) {
  for (mlir::Operation *op : reductions) {
    if (auto add = mlir::dyn_cast<cir::AddOp>(op)) {
      add.setNoSignedWrap(false);
      add.setNoUnsignedWrap(false);
    } else if (auto mul = mlir::dyn_cast<cir::MulOp>(op)) {
      mul.setNoSignedWrap(false);
      mul.setNoUnsignedWrap(false);
    }
  }
}

// True if op is a non-region leaf whose effect the dependence analysis cannot
// model. Escapes, calls, inline asm, memory intrinsics and memcpy are rejected
// by name because some report no queryable memory effect. A store is allowed
// only when the body may write any array, or when it targets the induction slot
// in a control region. Everything else must be provably free of memory effects.
static bool leafHasUnmodeledEffect(mlir::Operation *op, mlir::Value ivSlot,
                                   bool allowAnyStore) {
  if (mlir::isa<cir::BreakOp, cir::ContinueOp, cir::GotoOp, cir::ReturnOp,
                cir::ThrowOp, cir::TryOp, cir::AwaitOp, cir::CallOp,
                cir::InlineAsmOp, cir::LLVMIntrinsicCallOp, cir::MemCpyOp>(op))
    return true;
  if (auto l = mlir::dyn_cast<cir::LoadOp>(op))
    return l.getIsVolatile() || l.getMemOrder();
  if (auto s = mlir::dyn_cast<cir::StoreOp>(op)) {
    if (s.getIsVolatile() || s.getMemOrder())
      return true;
    return !allowAnyStore && s.getAddr() != ivSlot;
  }
  // Loop control terminators, array address arithmetic, and a local stack slot
  // are safe to reorder around but are not annotated memory effect free, so
  // allow them by name. A cir.alloca reserves stack and never aliases the
  // arrays the dependence analysis reasons about.
  if (mlir::isa<cir::GetElementOp, cir::ConditionOp, cir::YieldOp,
                cir::AllocaOp>(op))
    return false;
  return !mlir::isMemoryEffectFree(op);
}

static bool regionHasUnmodeledEffect(mlir::Region &region, mlir::Value ivSlot,
                                     bool allowAnyStore) {
  bool unsafe = false;
  region.walk([&](mlir::Operation *op) {
    if (unsafe || op->getNumRegions() > 0)
      return; // containers are visited through their contents
    if (leafHasUnmodeledEffect(op, ivSlot, allowAnyStore))
      unsafe = true;
  });
  return unsafe;
}

// True if the body has an op the dependence analysis cannot model soundly.
static bool hasUnmodeledEffect(mlir::Region &body) {
  return regionHasUnmodeledEffect(body, /*ivSlot=*/{}, /*allowAnyStore=*/true);
}

// Is interchanging the innermost pair with outer p and inner q provably legal.
// Subscripts are separable and distinct base objects do not alias.
// Every written base must vary with p or q so no dependence direction can be
// reversed by the swap.
// The one decline a runtime disjointness test can lift.
// True when no store in the nest writes the slot, so a load of it before the
// nest sees the value every iteration sees.
static bool slotUnwrittenIn(cir::ForOp nest, mlir::Value slot) {
  bool written = false;
  nest->walk([&](cir::StoreOp store) {
    if (store.getAddr() == slot)
      written = true;
  });
  return !written;
}

// Subscript equality reasoning shared by interchange legality and loop
// distribution. A subscript is seen relative to the two loops of the pair.
struct DimComp {
  enum Kind { J, Inner, Inv, Const, Unknown } kind;
  mlir::Value value; // Inv, the slot the subscript loads
  int64_t konst;     // Const
  bool operator==(const DimComp &o) const {
    return kind == o.kind && value == o.value && konst == o.konst;
  }
};

static DimComp dimCompOf(mlir::Value idx, mlir::Value jSlot, mlir::Value kSlot,
                         cir::ForOp jFor) {
  mlir::Value v = idx;
  while (auto cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  if (auto konst = v.getDefiningOp<cir::ConstantOp>()) {
    if (auto attr = mlir::dyn_cast<cir::IntAttr>(konst.getValue()))
      return {DimComp::Const, {}, attr.getValue().getSExtValue()};
    return {DimComp::Unknown, {}, 0};
  }
  auto load = v.getDefiningOp<cir::LoadOp>();
  mlir::Value slot = load ? load.getAddr() : mlir::Value();
  if (!slot)
    return {DimComp::Unknown, {}, 0};
  if (slot == jSlot)
    return {DimComp::J, {}, 0};
  if (slot == kSlot)
    return {DimComp::Inner, {}, 0};
  if (!slot.getDefiningOp<cir::AllocaOp>() || !slotUnwrittenIn(jFor, slot))
    return {DimComp::Unknown, {}, 0};
  return {DimComp::Inv, slot, 0};
}

// Whether two subscript vectors can name one element, as a little union find
// equality system. With crossIteration the touching iterations must differ in
// the outer variable, which distribution reversal requires. Infeasible when
// the equalities force the two outer iterations together under that demand,
// force two different constants together, or force the inner variable onto a
// value its start keeps it above.
static bool subscriptsCanMeet(llvm::ArrayRef<DimComp> a,
                              llvm::ArrayRef<DimComp> b, bool crossIteration,
                              mlir::Value innerStartSlot,
                              int64_t innerStartOff) {
  if (a.size() != b.size())
    return true; // shapes beyond the solver so assume the worst

  // Nodes 0 and 1 are the two outer iterations, 2 and 3 the inner variable
  // of each side.
  llvm::SmallVector<unsigned, 8> parent{0, 1, 2, 3};
  auto find = [&](unsigned x) {
    while (parent[x] != x)
      x = parent[x] = parent[parent[x]];
    return x;
  };
  auto join = [&](unsigned x, unsigned y) { parent[find(x)] = find(y); };
  llvm::DenseMap<mlir::Value, unsigned> valueNode;
  llvm::DenseMap<int64_t, unsigned> constNode;
  auto nodeOf = [&](const DimComp &c, bool late) -> int {
    switch (c.kind) {
    case DimComp::J:
      return late ? 0 : 1;
    case DimComp::Inner:
      return late ? 2 : 3;
    case DimComp::Inv: {
      auto it = valueNode.try_emplace(c.value, parent.size());
      if (it.second)
        parent.push_back(parent.size());
      return it.first->second;
    }
    case DimComp::Const: {
      auto it = constNode.try_emplace(c.konst, parent.size());
      if (it.second)
        parent.push_back(parent.size());
      return it.first->second;
    }
    case DimComp::Unknown:
      return -1;
    }
    llvm_unreachable("covered switch");
  };
  for (unsigned d = 0; d < a.size(); ++d) {
    int na = nodeOf(a[d], /*late=*/false);
    int nb = nodeOf(b[d], /*late=*/true);
    if (na < 0 || nb < 0)
      return true;
    join(nb, na);
  }
  if (crossIteration && find(0) == find(1))
    return false; // would need the two outer iterations equal
  llvm::DenseMap<unsigned, int64_t> constOf;
  for (auto &kv : constNode) {
    auto it = constOf.try_emplace(find(kv.second), kv.first);
    if (!it.second && it.first->second != kv.first)
      return false; // would need two different constants equal
  }
  if (innerStartSlot && innerStartOff >= 1) {
    auto it = valueNode.find(innerStartSlot);
    if (it != valueNode.end()) {
      unsigned lower = find(it->second);
      if (find(2) == lower || find(3) == lower)
        return false; // would need the inner variable below its start
    }
  }
  return true;
}

static const char kImperfectNest[] =
    "the nest is imperfect, the outer body holds more than the inner loop";

static const char kUnknownAliasing[] =
    "an access goes through a pointer whose aliasing is unknown";

static const char *
interchangeIllegal(mlir::Value p, mlir::Value q, mlir::Region &innerBody,
                   cir::ForOp nest, bool assumeParamsDistinct,
                   mlir::Value innerStartSlot, int64_t innerStartOff,
                   llvm::SmallVectorImpl<mlir::Operation *> &reductions) {
  // Bail on any op the dependence analysis cannot model.
  if (hasUnmodeledEffect(innerBody))
    return "the body holds an operation with effects this pass cannot model";

  // Per base we track the shared dimension tag shape and whether it is written.
  struct BaseInfo {
    llvm::SmallVector<IdxTag, 4> shape;
    bool written = false;
    bool seen = false;
    bool mixed = false; // reached through more than one subscript shape
    llvm::SmallVector<std::pair<llvm::SmallVector<DimComp, 4>, bool>, 8> accs;
  };
  llvm::DenseMap<mlir::Operation *, BaseInfo> bases;
  const char *bail = nullptr;

  auto handle = [&](mlir::Value addr, bool isWrite) {
    if (bail)
      return;
    llvm::SmallVector<mlir::Value, 4> idxs;
    mlir::Value base = peelAddress(addr, idxs);
    mlir::Operation *id = baseObjectId(base, assumeParamsDistinct);
    if (!id) {
      bail = kUnknownAliasing;
      return;
    }
    llvm::SmallVector<IdxTag, 4> shape;
    for (mlir::Value idx : idxs) {
      IdxTag t = classifyIndex(idx, p, q, innerBody);
      if (t == IdxTag::Unknown) {
        bail = "a subscript is not an induction variable, a constant, or "
               "invariant in the nest";
        return;
      }
      shape.push_back(t);
    }
    auto &info = bases[id];
    if (!info.seen) {
      info.seen = true;
      info.shape = shape;
    } else if (info.shape != shape) {
      info.mixed = true;
    }
    info.written |= isWrite;
    if (info.accs.size() < 8) {
      llvm::SmallVector<DimComp, 4> dims;
      for (mlir::Value idx : idxs)
        dims.push_back(dimCompOf(idx, p, q, nest));
      info.accs.push_back({dims, isWrite});
    } else {
      info.mixed = true; // too many shapes to reason about, stay safe
    }
  };

  innerBody.walk([&](mlir::Operation *op) {
    if (auto l = mlir::dyn_cast<cir::LoadOp>(op))
      handle(l.getAddr(), /*isWrite=*/false);
    else if (auto s = mlir::dyn_cast<cir::StoreOp>(op))
      handle(s.getAddr(), /*isWrite=*/true);
  });

  if (bail)
    return bail;

  // A base the nest never writes cannot carry a dependence, whatever its
  // subscripts are, so only a written base has to be examined.
  for (auto &kv : bases) {
    const BaseInfo &info = kv.second;
    if (!info.written)
      continue;
    // Two subscript shapes on one written base carry a dependence unless the
    // differing accesses provably never name one element.
    if (info.mixed) {
      for (unsigned i = 0; i < info.accs.size(); ++i)
        for (unsigned j = i + 1; j < info.accs.size(); ++j) {
          auto &a = info.accs[i], &b = info.accs[j];
          if (!a.second && !b.second)
            continue;
          if (a.first == b.first)
            continue; // same shape, updates stay in their relative order
          if (subscriptsCanMeet(a.first, b.first, /*crossIteration=*/false,
                                innerStartSlot, innerStartOff))
            return "a written object is reached through two different "
                   "subscript shapes";
        }
    }
    bool usesPQ = false;
    for (IdxTag t : info.shape)
      if (t == IdxTag::P || t == IdxTag::Q)
        usesPQ = true;
    if (!usesPQ) {
      mlir::Operation *combine = orderFreeReduction(kv.first, nest);
      if (!combine)
        return "a written object varies with neither induction variable and "
               "is not an order free reduction";
      reductions.push_back(combine);
    }
  }
  return nullptr;
}

// Count array accesses that gather under each candidate inner loop in one walk.
// An access gathers when the inner IV indexes a non-contiguous dimension.
// pGathers is what a p inner loop would gather, qGathers what a q inner loop
// would gather.
static void gatherCounts(mlir::Value p, mlir::Value q, mlir::Region &body,
                         int &pGathers, int &qGathers) {
  pGathers = 0;
  qGathers = 0;
  body.walk([&](mlir::Operation *op) {
    mlir::Value addr;
    if (auto l = mlir::dyn_cast<cir::LoadOp>(op))
      addr = l.getAddr();
    else if (auto s = mlir::dyn_cast<cir::StoreOp>(op))
      addr = s.getAddr();
    else
      return;
    llvm::SmallVector<mlir::Value, 4> idxs;
    peelAddress(addr, idxs);
    // idxs.front is contiguous so only the remaining dimensions can gather.
    for (size_t k = 1; k < idxs.size(); ++k) {
      IdxTag t = classifyIndex(idxs[k], p, q, body);
      if (t == IdxTag::P)
        ++pGathers;
      else if (t == IdxTag::Q)
        ++qGathers;
    }
  });
}

// A store whose cell never moves along the new inner loop becomes a serial
// floating point accumulation there, which cannot vectorize without value
// changing reassociation, while the current inner loop streams it across
// lanes. Removing a gather does not pay for that.
static bool formsSerialFpAccumulation(mlir::Value p, mlir::Value q,
                                      mlir::Region &body) {
  bool forms = false;
  body.walk([&](cir::StoreOp store) {
    llvm::SmallVector<mlir::Value, 4> idxs;
    peelAddress(store.getAddr(), idxs);
    if (idxs.empty() ||
        !mlir::isa<cir::FPTypeInterface>(store.getValue().getType()))
      return;
    for (mlir::Value idx : idxs)
      if (classifyIndex(idx, p, q, body) == IdxTag::P)
        return;
    forms = true;
  });
  return forms;
}

// The current inner loop is q and after interchange it becomes p.
// Profitable when the new inner loop p produces fewer gathers than q does now.
static bool isInterchangeProfitable(mlir::Value p, mlir::Value q,
                                    mlir::Region &innerBody) {
  int pGathers, qGathers;
  gatherCounts(p, q, innerBody, pGathers, qGathers);
  return pGathers < qGathers;
}

// Mechanical interchange

// The parts of a counted loop needed to move it.
// The for op, the IV slot, its alloca, and the init store in the same block.
struct LoopParts {
  cir::ForOp forOp;
  mlir::Value ivSlot;
  cir::AllocaOp alloca;
  cir::StoreOp init;
  mlir::Operation *initDef;
  bool ownsAlloca; // the slot is declared by this loop rather than shared
};

// Interchange leaves each induction variable holding a different final value,
// so a slot declared outside the nest may only be read where something has
// already set it again. A counter of another loop qualifies, a plain read of
// the value the nest left behind does not.
static bool slotConfinedToNest(mlir::Value slot, cir::ForOp nest) {
  for (mlir::Operation *user : slot.getUsers()) {
    if (!mlir::isa<cir::LoadOp, cir::StoreOp>(user))
      return false; // the address escapes so any reader is possible
    if (!mlir::isa<cir::LoadOp>(user) || nest->isAncestor(user))
      continue;
    bool reset = false;
    for (cir::ForOp f = user->getParentOfType<cir::ForOp>(); f && !reset;
         f = f->getParentOfType<cir::ForOp>())
      reset = llvm::any_of(
          f->getBlock()->getOps<cir::StoreOp>(), [&](cir::StoreOp store) {
            return store.getAddr() == slot && store->isBeforeInBlock(f);
          });
    if (!reset)
      return false;
  }
  return true;
}

static std::optional<LoopParts> getLoopParts(const CountedLoop &cl) {
  auto alloca = cl.ivSlot.getDefiningOp<cir::AllocaOp>();
  if (!alloca)
    return std::nullopt;
  mlir::Block *block = cl.forOp->getBlock();
  // An alloca in the loop block belongs to the loop and travels with it. One
  // further out already dominates both positions, so it stays where it is,
  // but then the counter outlives the nest and has to be checked for that.
  bool ownsAlloca = alloca->getBlock() == block;
  if (!ownsAlloca) {
    if (!alloca->getParentRegion()->isProperAncestor(
            cl.forOp->getParentRegion()))
      return std::nullopt;
    if (!slotConfinedToNest(cl.ivSlot, cl.forOp))
      return std::nullopt;
  }

  // The init is the last store before the loop. A shared counter may be
  // reinitialized several times in one block, once per loop that drives it.
  cir::StoreOp init;
  for (auto store : block->getOps<cir::StoreOp>()) {
    if (store.getAddr() != cl.ivSlot)
      continue;
    if (!store->isBeforeInBlock(cl.forOp))
      continue;
    if (!init || init->isBeforeInBlock(store))
      init = store;
  }
  if (!init)
    return std::nullopt;
  mlir::Operation *initDef = init.getValue().getDefiningOp();
  // The init value moves with the store, so what it may be depends on how the
  // caller uses it. initSurvivesMove decides that.
  if (!initDef ||
      !mlir::isa<cir::ConstantOp, cir::LoadOp, cir::AddOp>(initDef) ||
      !initDef->hasOneUse())
    return std::nullopt;
  if (initDef->getBlock() != block)
    return std::nullopt;
  return LoopParts{cl.forOp, cl.ivSlot, alloca, init, initDef, ownsAlloca};
}

// True when the init value reads the same thing from its new position.
// Defined with the versioning code below.
static bool invariantChainOk(mlir::Value v, cir::ForOp nest);
static void innerStartFact(LoopParts inner, cir::ForOp jFor, mlir::Value &slot,
                           int64_t &off);
static mlir::Value recreateInvariant(mlir::OpBuilder &builder, mlir::Value v,
                                     cir::ForOp nest);
static void eraseIfDead(mlir::Value value);

static bool initSurvivesMove(LoopParts lp, cir::ForOp nest) {
  // A constant or an invariant chain of loads and adds reads the same thing
  // from the new position, where it is rebuilt rather than moved.
  return invariantChainOk(lp.init.getValue(), nest);
}

// Swap the contents of two single entry regions.
static void swapRegions(mlir::Region &a, mlir::Region &b) {
  mlir::Region tmp;
  tmp.takeBody(a);
  a.takeBody(b);
  b.takeBody(tmp);
}

// Move a loop IV control just before target.
// Keeps the order alloca then init value then store.
static void moveIvControlBefore(LoopParts lp, mlir::Operation *target,
                                cir::ForOp nest) {
  if (lp.ownsAlloca)
    lp.alloca->moveBefore(target);
  lp.init->moveBefore(target);
  if (mlir::isa<cir::ConstantOp, cir::LoadOp>(lp.initDef)) {
    lp.initDef->moveBefore(lp.init.getOperation());
    return;
  }
  // An expression start such as i + 1 leaves its operands behind, so rebuild
  // it at the new position and drop the original chain.
  mlir::OpBuilder builder(lp.init.getOperation());
  mlir::Value old = lp.init.getValue();
  mlir::Value fresh = recreateInvariant(builder, old, nest);
  if (!fresh) {
    lp.initDef->moveBefore(lp.init.getOperation()); // initSurvivesMove said ok
    return;
  }
  lp.init.getValueMutable().assign(fresh);
  eraseIfDead(old);
}

// Interchange the outer and inner loops of an adjacent pair.
// Legality and profitability and getLoopParts must already have passed.
static void interchange(LoopParts outer, LoopParts inner) {
  cir::ForOp fOuter = outer.forOp;
  cir::ForOp fInner = inner.forOp;
  // Swap loop control so each for now drives the other IV.
  swapRegions(fOuter.getCond(), fInner.getCond());
  swapRegions(fOuter.getStep(), fInner.getStep());
  // Declare each IV control in its new scope.
  // The inner IV control reinitializes on each outer iteration.
  moveIvControlBefore(inner, fOuter, outer.forOp);
  moveIvControlBefore(outer, fInner, outer.forOp);
}

// A genuine perfect nest has nothing in the outer body but the inner loop.
// Any other statement would change its execution count after interchange.
static bool isPerfectPair(cir::ForOp outerFor, LoopParts inner) {
  bool ok = true;
  outerFor.getBody().walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *op) -> mlir::WalkResult {
        if (op == inner.forOp.getOperation())
          return mlir::WalkResult::skip(); // the inner loop subtree is expected
        if (mlir::isa<cir::ScopeOp, cir::YieldOp, cir::ConditionOp>(op))
          return mlir::WalkResult::advance(); // structural only
        if (op->hasOneUse() && *op->user_begin() == inner.initDef)
          return mlir::WalkResult::advance(); // feeds the inner IV setup
        if (op == inner.alloca.getOperation() || op == inner.initDef ||
            op == inner.init.getOperation())
          return mlir::WalkResult::advance(); // inner IV setup
        ok = false; // some other op lives around the loops
        return mlir::WalkResult::interrupt();
      });
  return ok;
}

// Does v transitively read slot. Used to reject non-rectangular bounds. The
// seen set keeps a shared definition from being revisited on a value DAG.
static bool valueDependsOnSlot(mlir::Value v, mlir::Value slot,
                               llvm::SmallPtrSetImpl<mlir::Operation *> &seen) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def || !seen.insert(def).second)
    return false;
  if (auto load = mlir::dyn_cast<cir::LoadOp>(def))
    return load.getAddr() == slot;
  for (mlir::Value operand : def->getOperands())
    if (valueDependsOnSlot(operand, slot, seen))
      return true;
  return false;
}

static bool valueDependsOnSlot(mlir::Value v, mlir::Value slot) {
  llvm::SmallPtrSet<mlir::Operation *, 16> seen;
  return valueDependsOnSlot(v, slot, seen);
}

// The upper bound value tested in a counted loop cond region.
// The comparison a counted loop exits on, or null when the cond region does
// not have that shape. Every consumer of the exit test goes through here.
static cir::CmpOp getExitCmp(cir::ForOp forOp) {
  mlir::Region &cond = forOp.getCond();
  if (!cond.hasOneBlock())
    return {};
  auto condOp =
      mlir::dyn_cast_or_null<cir::ConditionOp>(cond.front().getTerminator());
  if (!condOp)
    return {};
  return condOp.getCondition().getDefiningOp<cir::CmpOp>();
}

static mlir::Value getLoopBoundRHS(cir::ForOp forOp) {
  cir::CmpOp cmp = getExitCmp(forOp);
  return cmp ? cmp.getRhs() : mlir::Value();
}

// True when the loop exits on a strict less than test. The rebuilt triangle
// below is derived from that test, so any other comparison is a different
// iteration space.
// True when the loop exits on <= rather than <.
static bool exitsOnLessOrEqual(cir::ForOp forOp) {
  cir::CmpOp cmp = getExitCmp(forOp);
  return cmp && cmp.getKind() == cir::CmpOpKind::le;
}

static bool exitsOnLessThan(cir::ForOp forOp) {
  cir::CmpOp cmp = getExitCmp(forOp);
  return cmp && cmp.getKind() == cir::CmpOpKind::lt;
}

// True when the value can be emitted as a constant of this type without
// changing it. The 32 bit limit also keeps the arithmetic below in range.
static bool isEmittable(mlir::Type type, int64_t value) {
  auto intType = mlir::dyn_cast<cir::IntType>(type);
  if (!intType || !llvm::isInt<32>(value))
    return false;
  return intType.isSigned()
             ? llvm::isIntN(intType.getWidth(), value)
             : value >= 0 && llvm::isUIntN(intType.getWidth(), value);
}

// Fold an integer expression over constants. A bound written as N minus K
// reaches the pass unfolded, so reading the constant off one op is not enough.
static std::optional<int64_t> foldConstant(mlir::Value value,
                                           unsigned depth = 0) {
  mlir::Operation *op = value.getDefiningOp();
  if (!op || depth > 4)
    return std::nullopt;
  if (auto cst = mlir::dyn_cast<cir::ConstantOp>(op)) {
    auto attr = mlir::dyn_cast<cir::IntAttr>(cst.getValue());
    auto intType =
        attr ? mlir::dyn_cast<cir::IntType>(attr.getType()) : cir::IntType();
    if (!intType)
      return std::nullopt;
    int64_t folded = intType.isSigned()
                         ? attr.getValue().getSExtValue()
                         : static_cast<int64_t>(attr.getValue().getZExtValue());
    return isEmittable(intType, folded) ? std::optional<int64_t>(folded)
                                        : std::nullopt;
  }
  if (auto cast = mlir::dyn_cast<cir::CastOp>(op)) {
    if (cast.getKind() != cir::CastKind::integral)
      return std::nullopt;
    std::optional<int64_t> source = foldConstant(cast.getSrc(), depth + 1);
    // A narrowing cast would not carry the folded value through unchanged.
    if (source && isEmittable(cast.getType(), *source))
      return source;
    return std::nullopt;
  }
  std::optional<int64_t> lhs, rhs;
  if (op->getNumOperands() == 2) {
    lhs = foldConstant(op->getOperand(0), depth + 1);
    rhs = foldConstant(op->getOperand(1), depth + 1);
  }
  if (!lhs || !rhs)
    return std::nullopt;
  int64_t folded;
  if (mlir::isa<cir::AddOp>(op)) {
    if (llvm::AddOverflow(*lhs, *rhs, folded))
      return std::nullopt;
  } else if (mlir::isa<cir::SubOp>(op)) {
    if (llvm::SubOverflow(*lhs, *rhs, folded))
      return std::nullopt;
  } else if (mlir::isa<cir::MulOp>(op)) {
    if (llvm::MulOverflow(*lhs, *rhs, folded))
      return std::nullopt;
  } else if (mlir::isa<cir::DivOp>(op) && *rhs != 0) {
    folded = *lhs / *rhs;
  } else {
    return std::nullopt;
  }
  // Check every step, not just the leaves. Otherwise a wrapping unsigned
  // subtraction reaches the callers as a negative value they compare as
  // signed, and a nested product leaves the range the leaf check assumed.
  return isEmittable(op->getResult(0).getType(), folded)
             ? std::optional<int64_t>(folded)
             : std::nullopt;
}

// Erase a dead value and everything that dies with it. Operations are collected
// before any erase, since a value reachable twice through the operand graph
// would otherwise be visited again after its definition was freed.
static void eraseIfDead(mlir::Value value) {
  llvm::SmallVector<mlir::Operation *> worklist, order;
  llvm::SmallPtrSet<mlir::Operation *, 8> dead;
  if (mlir::Operation *def = value.getDefiningOp())
    worklist.push_back(def);
  while (!worklist.empty()) {
    mlir::Operation *op = worklist.pop_back_val();
    // An operation dies only once every user of it is already dying.
    if (dead.contains(op) ||
        !llvm::all_of(op->getUsers(), [&](mlir::Operation *user) {
          return dead.contains(user);
        }))
      continue;
    dead.insert(op);
    order.push_back(op);
    for (mlir::Value operand : op->getOperands())
      if (mlir::Operation *def = operand.getDefiningOp())
        worklist.push_back(def);
  }
  // Users come before their operands in this order, so each erase is safe.
  for (mlir::Operation *op : order)
    op->erase();
}

// True when the loop advances its induction variable by exactly one. A rebuilt
// start lands on that grid and on no other, so a wider stride would visit
// values the original nest never had.
static bool stepsByOne(cir::ForOp forOp, mlir::Value ivSlot) {
  for (auto store : forOp.getStep().front().getOps<cir::StoreOp>()) {
    if (store.getAddr() != ivSlot)
      continue;
    mlir::Value stepped = store.getValue();
    if (auto inc = stepped.getDefiningOp<cir::IncOp>()) {
      auto load = inc->getOperand(0).getDefiningOp<cir::LoadOp>();
      return load && load.getAddr() == ivSlot;
    }
    auto add = stepped.getDefiningOp<cir::AddOp>();
    if (!add)
      return false;
    mlir::Value counter = add.getLhs();
    std::optional<int64_t> amount = foldConstant(add.getRhs());
    if (!amount) {
      counter = add.getRhs();
      amount = foldConstant(add.getLhs());
    }
    auto load = counter.getDefiningOp<cir::LoadOp>();
    return amount && *amount == 1 && load && load.getAddr() == ivSlot;
  }
  return false;
}

// The nest running j from zero to scale times the outer induction variable
// plus offset.
struct UpperTriangle {
  int64_t outerBound; // exclusive
  int64_t scale;      // C in j < C * i + K
  int64_t offset;     // K in j < C * i + K
  int64_t outerStart;
  bool orEqual;
};

// Split a constant off a value, in either operand order. Returns the identity
// when the value is not that operation at all, and nothing when it is one but
// neither side is a constant.
template <typename OpTy>
static std::optional<int64_t> peelConstant(mlir::Value &value,
                                           int64_t identity) {
  auto op = value.getDefiningOp<OpTy>();
  if (!op)
    return identity;
  if (std::optional<int64_t> cst = foldConstant(op.getRhs())) {
    value = op.getLhs();
    return cst;
  }
  if (std::optional<int64_t> cst = foldConstant(op.getLhs())) {
    value = op.getRhs();
    return cst;
  }
  return std::nullopt;
}

// Recognize the triangle the swap has to rebuild rather than copy. The inner
// loop must run from zero to a constant multiple of the outer induction
// variable plus a constant, and the outer bound and start must both be known.
static std::optional<UpperTriangle> recognizeUpperTriangle(LoopParts outer,
                                                           LoopParts inner,
                                                           mlir::Value pBound,
                                                           mlir::Value qBound) {
  bool orEqual = exitsOnLessOrEqual(inner.forOp);
  if (!exitsOnLessThan(outer.forOp) ||
      (!exitsOnLessThan(inner.forOp) && !orEqual))
    return std::nullopt;
  // The rebuilt inner start rides the outer loop's own stride.
  if (!stepsByOne(outer.forOp, outer.ivSlot))
    return std::nullopt;

  mlir::Value index = qBound;
  std::optional<int64_t> offset = peelConstant<cir::AddOp>(index, 0);
  std::optional<int64_t> scale = peelConstant<cir::MulOp>(index, 1);
  if (!offset || !scale)
    return std::nullopt;

  auto load = index.getDefiningOp<cir::LoadOp>();
  if (!load || load.getAddr() != outer.ivSlot)
    return std::nullopt;

  std::optional<int64_t> bound = foldConstant(pBound);
  std::optional<int64_t> start = foldConstant(outer.init.getValue());
  std::optional<int64_t> innerStart = foldConstant(inner.init.getValue());
  if (!bound || !start || !innerStart)
    return std::nullopt;
  if (*innerStart != 0 || *bound < 1 || *offset < 0 || *scale < 1)
    return std::nullopt;
  // Above a unit scale the new start needs a division, and the truncation cir
  // division performs only matches the floor the inversion wants while the
  // numerator stays non-negative, which an offset would break.
  if (*scale > 1 && *offset != 0)
    return std::nullopt;
  // Inverting a scaled bound inclusively needs a ceiling, and cir division
  // truncates, so only the unit scale is inclusive here.
  if (orEqual && *scale > 1)
    return std::nullopt;
  int64_t reach = orEqual ? 1 : 0;
  mlir::Type type = qBound.getType();
  if (!isEmittable(type, *scale * (*bound - 1) + *offset + reach) ||
      !isEmittable(type, reach - *offset) || !isEmittable(type, *scale) ||
      !isEmittable(type, *start))
    return std::nullopt;
  return UpperTriangle{*bound, *scale, *offset, *start, orEqual};
}

// Rebuild the bounds so the swapped nest walks the same triangle. j stops
// below scale times bound minus one plus offset, and the outer variable starts
// at j plus one minus offset, held at its original start where that falls
// below it. The swap leaves both loops carrying the other one's test.
static void rewriteUpperTriangle(LoopParts outer, LoopParts inner,
                                 const UpperTriangle &tri) {
  cir::CmpOp cmp = getExitCmp(outer.forOp);
  mlir::Value oldBoundValue = cmp.getRhs();
  mlir::Type type = oldBoundValue.getType();

  mlir::OpBuilder builder(cmp);
  cmp.getRhsMutable().assign(cir::ConstantOp::create(
      builder, cmp.getLoc(),
      cir::IntAttr::get(type, tri.scale * (tri.outerBound - 1) + tri.offset +
                                  (tri.orEqual ? 1 : 0))));
  eraseIfDead(oldBoundValue);

  mlir::Location loc = outer.init.getLoc();
  builder.setInsertionPoint(outer.init);
  mlir::Value index = cir::LoadOp::create(builder, loc, {inner.ivSlot});
  mlir::Value begin;
  int64_t lowest;
  if (tri.scale == 1) {
    // An inclusive test reaches one further, so the start moves back by one.
    int64_t reach = tri.orEqual ? 0 : 1;
    mlir::Value shift = cir::ConstantOp::create(
        builder, loc, cir::IntAttr::get(type, reach - tri.offset));
    begin = cir::AddOp::create(builder, loc, type, index, shift);
    lowest = reach - tri.offset;
  } else {
    // C times i exceeds j exactly when i is past j divided by C.
    mlir::Value divisor = cir::ConstantOp::create(
        builder, loc, cir::IntAttr::get(type, tri.scale));
    mlir::Value one =
        cir::ConstantOp::create(builder, loc, cir::IntAttr::get(type, 1));
    begin = cir::AddOp::create(
        builder, loc, type,
        cir::DivOp::create(builder, loc, type, index, divisor), one);
    lowest = 1;
  }
  // The expression is smallest at j equal to zero, so that is where it can
  // reach below the start the outer loop originally had.
  if (lowest < tri.outerStart) {
    mlir::Value floorValue = cir::ConstantOp::create(
        builder, loc, cir::IntAttr::get(type, tri.outerStart));
    mlir::Value below =
        cir::CmpOp::create(builder, loc, cir::CmpOpKind::lt, begin, floorValue);
    begin = cir::SelectOp::create(builder, loc, type, below, floorValue, begin);
  }
  mlir::Value oldInit = outer.init.getValue();
  outer.init.getValueMutable().assign(begin);
  eraseIfDead(oldInit);
}

// The nest running j from the outer induction variable to a constant, the
// triangle written on the start rather than on the bound.
struct LowerTriangle {
  // The outer loop's start, which the swapped outer loop takes.
  int64_t outerStart;
  int64_t offset; // d in j = i + d
};

// Recognize that form. The inner bound may not exceed the outer bound, so that
// after the swap stopping just past j also respects the bound the outer loop
// originally had.
static std::optional<LowerTriangle> recognizeLowerTriangle(LoopParts outer,
                                                           LoopParts inner,
                                                           mlir::Value pBound,
                                                           mlir::Value qBound) {
  if (!exitsOnLessThan(outer.forOp) || !exitsOnLessThan(inner.forOp))
    return std::nullopt;
  // The rebuilt outer start rides the inner loop's own stride.
  if (!stepsByOne(inner.forOp, inner.ivSlot))
    return std::nullopt;
  // The outer loop's control is relocated by this swap, so its start has to be
  // a value the move carries intact.
  if (!mlir::isa<cir::ConstantOp>(outer.initDef))
    return std::nullopt;
  mlir::Value from = inner.init.getValue();
  std::optional<int64_t> offset = peelConstant<cir::AddOp>(from, 0);
  if (!offset)
    return std::nullopt;
  auto load = from.getDefiningOp<cir::LoadOp>();
  if (!load || load.getAddr() != outer.ivSlot)
    return std::nullopt;
  std::optional<int64_t> outerBound = foldConstant(pBound);
  std::optional<int64_t> innerBound = foldConstant(qBound);
  std::optional<int64_t> start = foldConstant(outer.init.getValue());
  if (!outerBound || !innerBound || !start || *innerBound > *outerBound)
    return std::nullopt;
  // A negative offset would let the rebuilt inner bound run past the bound the
  // outer loop originally had.
  if (*offset < 0)
    return std::nullopt;
  mlir::Type type = inner.init.getValue().getType();
  if (!isEmittable(type, *start + *offset) || !isEmittable(type, 1 - *offset))
    return std::nullopt;
  return LowerTriangle{*start, *offset};
}

// Rebuild the bounds for the start side triangle. j starts where the outer
// loop did, and the outer variable stops just past j. The swap leaves j
// starting at a variable that is no longer in scope.
static void rewriteLowerTriangle(LoopParts outer, LoopParts inner,
                                 const LowerTriangle &tri) {
  mlir::Type type = inner.init.getValue().getType();
  mlir::Value oldStart = inner.init.getValue();
  mlir::OpBuilder builder(inner.init);
  inner.init.getValueMutable().assign(cir::ConstantOp::create(
      builder, inner.init.getLoc(),
      cir::IntAttr::get(type, tri.outerStart + tri.offset)));
  eraseIfDead(oldStart);

  cir::CmpOp cmp = getExitCmp(inner.forOp);
  mlir::Value oldBound = cmp.getRhs();
  builder.setInsertionPoint(cmp);
  mlir::Value index =
      cir::LoadOp::create(builder, cmp.getLoc(), {inner.ivSlot});
  mlir::Value shift = cir::ConstantOp::create(
      builder, cmp.getLoc(), cir::IntAttr::get(type, 1 - tri.offset));
  cmp.getRhsMutable().assign(
      cir::AddOp::create(builder, cmp.getLoc(), type, index, shift));
  eraseIfDead(oldBound);
}

// True when the exit test has a bare load of this loop's induction variable on
// its left, which is what makes cmp.getRhs() the whole bound. It says nothing
// about the right hand side, which the triangle paths expect to name the other
// variable.
static bool exitTestLhsIsIv(cir::ForOp forOp, mlir::Value ivSlot) {
  cir::CmpOp cmp = getExitCmp(forOp);
  if (!cmp)
    return false;
  auto load = cmp.getLhs().getDefiningOp<cir::LoadOp>();
  return load && load.getAddr() == ivSlot;
}

// The nest whose inner exit test compares the product of the two induction
// variables against a constant.
struct ProductNest {
  int64_t limit; // Q in j * i < Q
  int64_t outerStart;
  int64_t outerBound; // exclusive
};

static std::optional<ProductNest> recognizeProductNest(LoopParts outer,
                                                       LoopParts inner,
                                                       mlir::Value pBound,
                                                       mlir::Value qBound) {
  if (!exitsOnLessThan(outer.forOp) || !exitsOnLessThan(inner.forOp))
    return std::nullopt;
  if (!exitTestLhsIsIv(outer.forOp, outer.ivSlot))
    return std::nullopt;
  cir::CmpOp cmp = getExitCmp(inner.forOp);
  auto product = cmp.getLhs().getDefiningOp<cir::MulOp>();
  if (!product)
    return std::nullopt;
  auto left = product.getLhs().getDefiningOp<cir::LoadOp>();
  auto right = product.getRhs().getDefiningOp<cir::LoadOp>();
  if (!left || !right)
    return std::nullopt;
  if (!((left.getAddr() == inner.ivSlot && right.getAddr() == outer.ivSlot) ||
        (left.getAddr() == outer.ivSlot && right.getAddr() == inner.ivSlot)))
    return std::nullopt;

  std::optional<int64_t> limit = foldConstant(qBound);
  std::optional<int64_t> bound = foldConstant(pBound);
  std::optional<int64_t> start = foldConstant(outer.init.getValue());
  std::optional<int64_t> innerStart = foldConstant(inner.init.getValue());
  if (!limit || !bound || !start || !innerStart)
    return std::nullopt;
  // A non-positive outer start leaves the product stuck and the inner loop
  // never ending. Keeping the limit within the outer bound lets the rebuilt
  // inner bound stop without a second clamp.
  if (*innerStart != 0 || *start < 1 || *limit < 1 || *limit > *bound)
    return std::nullopt;
  mlir::Type type = qBound.getType();
  if (!isEmittable(type, (*limit - 1) / *start + 1) ||
      !isEmittable(type, *limit - 1) || !isEmittable(type, *bound))
    return std::nullopt;
  return ProductNest{*limit, *start, *bound};
}

// Rebuild the bounds for the product test. j runs while it leaves room for the
// smallest outer value, and the outer variable stops one past the limit
// divided by j. The swap leaves the outer loop testing a product.
static void rewriteProductNest(LoopParts outer, LoopParts inner,
                               const ProductNest &nest) {
  cir::CmpOp outerCmp = getExitCmp(outer.forOp);
  mlir::Type type = outerCmp.getRhs().getType();
  mlir::Value oldProduct = outerCmp.getLhs();
  mlir::Value oldLimit = outerCmp.getRhs();

  mlir::OpBuilder builder(outerCmp);
  outerCmp.getLhsMutable().assign(
      cir::LoadOp::create(builder, outerCmp.getLoc(), {inner.ivSlot}));
  outerCmp.getRhsMutable().assign(cir::ConstantOp::create(
      builder, outerCmp.getLoc(),
      cir::IntAttr::get(type, (nest.limit - 1) / nest.outerStart + 1)));
  eraseIfDead(oldProduct);
  eraseIfDead(oldLimit);

  cir::CmpOp innerCmp = getExitCmp(inner.forOp);
  mlir::Value oldBound = innerCmp.getRhs();
  mlir::Location loc = innerCmp.getLoc();
  builder.setInsertionPoint(innerCmp);
  mlir::Value index = cir::LoadOp::create(builder, loc, {inner.ivSlot});
  mlir::Value one =
      cir::ConstantOp::create(builder, loc, cir::IntAttr::get(type, 1));
  mlir::Value first =
      cir::CmpOp::create(builder, loc, cir::CmpOpKind::lt, index, one);
  // A zero j leaves the product below the limit for every outer value, so that
  // row keeps the original bound. The divisor is held at one as well, since
  // both arms of the select are evaluated.
  mlir::Value divisor =
      cir::SelectOp::create(builder, loc, type, first, one, index);
  mlir::Value top = cir::ConstantOp::create(
      builder, loc, cir::IntAttr::get(type, nest.limit - 1));
  mlir::Value quotient = cir::AddOp::create(
      builder, loc, type, cir::DivOp::create(builder, loc, type, top, divisor),
      one);
  mlir::Value whole = cir::ConstantOp::create(
      builder, loc, cir::IntAttr::get(type, nest.outerBound));
  innerCmp.getRhsMutable().assign(
      cir::SelectOp::create(builder, loc, type, first, whole, quotient));
  eraseIfDead(oldBound);
}

// Interchange the innermost pair of a nest when legal and profitable.
// Returns null on success, otherwise why the pair was left alone.
// assumeParamsDistinct trusts a runtime disjointness test the caller emits.
// dryRun answers without transforming, so versioning can probe first.
// ignoreImperfect answers for the nest as loop distribution would leave it,
// so it is only meaningful together with dryRun.
static const char *tryInterchangePair(CountedLoop outerCL, CountedLoop innerCL,
                                      bool assumeParamsDistinct = false,
                                      bool dryRun = false,
                                      bool ignoreImperfect = false) {
  mlir::Region &innerBody = innerCL.forOp.getBody();

  // Only handle a genuinely innermost inner loop.
  bool hasDeeper = false;
  innerBody.walk([&](cir::ForOp f) {
    if (f != innerCL.forOp)
      hasDeeper = true;
  });
  if (hasDeeper)
    return "the inner loop of the pair is not innermost";

  std::optional<LoopParts> outer = getLoopParts(outerCL);
  std::optional<LoopParts> inner = getLoopParts(innerCL);
  if (!outer || !inner)
    return "an induction variable is not a plain counter this pass can move";

  // Genuine perfect nest with nothing but the inner loop between headers.
  if (!ignoreImperfect && !isPerfectPair(outerCL.forOp, *inner))
    return kImperfectNest;

  // Rectangular bounds, or one of the two triangular forms rebuilt below. Any
  // other coupling between a bound and the other induction variable would not
  // survive swapping the cond regions.
  mlir::Value pBound = getLoopBoundRHS(outerCL.forOp);
  mlir::Value qBound = getLoopBoundRHS(innerCL.forOp);
  if (!pBound || !qBound)
    return "an exit test is not a comparison against a bound";
  std::optional<ProductNest> product =
      recognizeProductNest(*outer, *inner, pBound, qBound);
  std::optional<UpperTriangle> triangle;
  std::optional<LowerTriangle> lower;
  if (!product) {
    // Every other path copies the cond regions across, so each exit test has
    // to name only its own induction variable.
    if (!exitTestLhsIsIv(outerCL.forOp, outerCL.ivSlot) ||
        !exitTestLhsIsIv(innerCL.forOp, innerCL.ivSlot))
      return "an exit test does not compare its own induction variable";
    triangle = recognizeUpperTriangle(*outer, *inner, pBound, qBound);
    if (!triangle)
      lower = recognizeLowerTriangle(*outer, *inner, pBound, qBound);
    if (!triangle && !lower &&
        (valueDependsOnSlot(pBound, innerCL.ivSlot) ||
         valueDependsOnSlot(qBound, outerCL.ivSlot)))
      return "the bounds couple the two induction variables in a form this "
             "pass cannot rebuild";

    // The lower form replaces the inner start outright, so it is the one case
    // that does not have to carry that value to its new position.
    if (!lower && (!initSurvivesMove(*outer, outerCL.forOp) ||
                   !initSurvivesMove(*inner, outerCL.forOp)))
      return "a start value would read something else from its new position";
  }

  mlir::Value innerStartSlot;
  int64_t innerStartOff;
  innerStartFact(*inner, outerCL.forOp, innerStartSlot, innerStartOff);
  llvm::SmallVector<mlir::Operation *, 2> reductions;
  if (const char *why = interchangeIllegal(
          outerCL.ivSlot, innerCL.ivSlot, innerBody, outerCL.forOp,
          assumeParamsDistinct, innerStartSlot, innerStartOff, reductions))
    return why;
  if (!isInterchangeProfitable(outerCL.ivSlot, innerCL.ivSlot, innerBody))
    return "the swap would not reduce the number of gathering accesses";
  if (formsSerialFpAccumulation(outerCL.ivSlot, innerCL.ivSlot, innerBody))
    return "the swap would turn a streamed floating point update into a "
           "serial accumulation";

  if (dryRun)
    return nullptr;

  interchange(*outer, *inner);
  relaxWrapFlags(reductions);
  if (product)
    rewriteProductNest(*outer, *inner, *product);
  else if (triangle)
    rewriteUpperTriangle(*outer, *inner, *triangle);
  else if (lower)
    rewriteLowerTriangle(*outer, *inner, *lower);
  return nullptr;
}

// Runtime aliasing versioning
//
// Arrays passed as plain pointer parameters have disjointness unknowable at
// compile time. When that is the only obstacle, the nest is duplicated under
// a runtime test comparing the spans it touches. The disjoint side carries
// the interchange and the other side keeps the original order.

// Rebuild an invariant value at the builder's insertion point. Constants and
// ordinary loads of slots the nest never writes, under casts, are the forms
// that read back identically there.
static mlir::Value recreateInvariant(mlir::OpBuilder &builder, mlir::Value v,
                                     cir::ForOp nest) {
  if (auto cast = v.getDefiningOp<cir::CastOp>()) {
    mlir::Value src = recreateInvariant(builder, cast.getSrc(), nest);
    if (!src)
      return {};
    return cir::CastOp::create(builder, cast.getLoc(), cast.getType(),
                               cast.getKind(), src);
  }
  if (auto konst = v.getDefiningOp<cir::ConstantOp>())
    return cir::ConstantOp::create(builder, konst.getLoc(), konst.getValue());
  if (auto add = v.getDefiningOp<cir::AddOp>()) {
    mlir::Value lhs = recreateInvariant(builder, add.getLhs(), nest);
    mlir::Value rhs =
        lhs ? recreateInvariant(builder, add.getRhs(), nest) : mlir::Value();
    if (!rhs)
      return {};
    // The wrap flags are dropped, which only weakens what is promised.
    return cir::AddOp::create(builder, add.getLoc(), add.getType(), lhs, rhs);
  }
  auto load = v.getDefiningOp<cir::LoadOp>();
  if (!load || load.getIsVolatile() || load.getMemOrder())
    return {};
  auto slot = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!slot || !slotUnwrittenIn(nest, load.getAddr()))
    return {};
  // The slot has to already be in scope where the test is emitted.
  mlir::Block *at = builder.getInsertionBlock();
  if (slot->getBlock() != at &&
      !slot->getParentRegion()->isProperAncestor(at->getParent()))
    return {};
  return cir::LoadOp::create(builder, load.getLoc(), {load.getAddr()});
}

// Whether recreateInvariant would succeed, with the scope check made against
// the nest itself since every rebuild position dominates or equals it.
static bool invariantChainOk(mlir::Value v, cir::ForOp nest) {
  if (v.getDefiningOp<cir::ConstantOp>())
    return true;
  if (auto add = v.getDefiningOp<cir::AddOp>())
    return invariantChainOk(add.getLhs(), nest) &&
           invariantChainOk(add.getRhs(), nest);
  auto load = v.getDefiningOp<cir::LoadOp>();
  if (!load || load.getIsVolatile() || load.getMemOrder())
    return false;
  auto slot = load.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!slot || !slotUnwrittenIn(nest, load.getAddr()))
    return false;
  mlir::Block *at = nest->getBlock();
  return (slot->getBlock() == at && slot->isBeforeInBlock(nest)) ||
         slot->getParentRegion()->isProperAncestor(at->getParent());
}

// The number of elements one step of the leading subscript moves over, and
// the scalar element type at the bottom.
static int64_t elemsUnder(mlir::Type type, mlir::Type &elem) {
  int64_t n = 1;
  while (auto arr = mlir::dyn_cast<cir::ArrayType>(type)) {
    n *= (int64_t)arr.getSize();
    type = arr.getElementType();
  }
  elem = type;
  return n;
}

// One object the nest touches, with what is needed to bound its span.
struct Span {
  mlir::Value start;    // dominating pointer to the first element
  int64_t elemsPerStep; // param span is rows * this
  int64_t fixedElems;   // whole object size when not a param
  bool isParam = false;
  bool written = false;
  // Values bounding the rows this object uses, each with whether the row at
  // that value is still touched. Kept per object so no span grows past its
  // own allocation, which would read as an overlap with its heap neighbor.
  llvm::SmallVector<std::pair<mlir::Value, bool>, 4> rowLimits;
};

// Version the nest of the pair behind a runtime disjointness test and return
// the cloned pair, ready for an interchange that assumes distinct parameters.
// Returns nullopt when the spans cannot be bounded, leaving the IR untouched.
static std::optional<std::pair<CountedLoop, CountedLoop>>
versionNest(CountedLoop outerCL, CountedLoop innerCL, mlir::Region &scan) {
  std::optional<LoopParts> outer = getLoopParts(outerCL);
  if (!outer)
    return std::nullopt;
  cir::ForOp nest = outerCL.forOp;

  // The unit to duplicate is the loop with its own IV control, nothing else
  // between them.
  llvm::SmallVector<mlir::Operation *, 4> unit;
  if (outer->ownsAlloca)
    unit.push_back(outer->alloca);
  unit.push_back(outer->initDef);
  unit.push_back(outer->init.getOperation());
  unit.push_back(nest.getOperation());
  llvm::sort(unit, [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });
  llvm::SmallPtrSet<mlir::Operation *, 4> inUnit;
  inUnit.insert_range(unit);
  for (mlir::Operation *op = unit.front(); op != nest.getOperation();
       op = op->getNextNode())
    if (!inUnit.contains(op))
      return std::nullopt;

  // Collect the spans and the values bounding the rows each parameter uses.
  // A leading subscript of p or q is bounded by that loop's exit bound, and
  // anything else invariant is its own bound. Bounds are collected as pairs
  // of a value and whether the row at that value is still touched.
  llvm::DenseMap<mlir::Operation *, Span> spans;
  mlir::Type elemTy;
  bool viable = true;

  auto slotOfLoad = [](mlir::Value v) -> mlir::Value {
    while (auto cast = v.getDefiningOp<cir::CastOp>())
      v = cast.getSrc();
    auto load = v.getDefiningOp<cir::LoadOp>();
    return load ? load.getAddr() : mlir::Value();
  };
  auto boundOf = [&](cir::ForOp loop) -> std::pair<mlir::Value, bool> {
    mlir::Value bound = getLoopBoundRHS(loop);
    bool reach = exitsOnLessOrEqual(loop);
    // A bound naming the other induction variable, as in j <= i, is itself
    // bounded by that loop's bound.
    if (bound && slotOfLoad(bound) == outerCL.ivSlot)
      return {getLoopBoundRHS(outerCL.forOp), true};
    return {bound, reach};
  };

  auto visit = [&](mlir::Value addr, bool isWrite) {
    if (!viable)
      return;
    llvm::SmallVector<mlir::Value, 4> idxs;
    mlir::Value base = peelAddress(addr, idxs);
    mlir::Operation *id = baseObjectId(base, /*assumeParamsDistinct=*/true);
    if (!id) {
      viable = false;
      return;
    }
    // A local of this frame cannot be where a parameter points, since the
    // caller took its argument before the frame existed, so it needs no test.
    if (mlir::isa<cir::AllocaOp>(id))
      return;
    Span &span = spans[id];
    span.written |= isWrite;
    if (!span.start) {
      mlir::Type scalar;
      if (cir::StoreOp init = paramSlotInit(base)) {
        span.isParam = true;
        span.start = init.getValue(); // the argument itself, always in scope
        span.elemsPerStep = elemsUnder(
            mlir::cast<cir::PointerType>(base.getType()).getPointee(), scalar);
      } else {
        span.start = mlir::Value(); // globals rebuilt at the test
        span.fixedElems = elemsUnder(
            mlir::cast<cir::PointerType>(base.getType()).getPointee(), scalar);
      }
      if (!elemTy)
        elemTy = scalar;
      else if (elemTy != scalar)
        viable = false;
    }
    if (!span.isParam)
      return;
    // Bound the rows this parameter uses through its leading subscript.
    if (idxs.empty()) {
      span.rowLimits.push_back({mlir::Value(), true}); // a lone *p, one row
      return;
    }
    switch (classifyIndex(idxs.back(), outerCL.ivSlot, innerCL.ivSlot, scan)) {
    case IdxTag::P:
      span.rowLimits.push_back(boundOf(outerCL.forOp));
      break;
    case IdxTag::Q:
      span.rowLimits.push_back(boundOf(innerCL.forOp));
      break;
    case IdxTag::Inv:
      span.rowLimits.push_back({idxs.back(), true});
      break;
    case IdxTag::Unknown:
      viable = false;
      break;
    }
    if (!span.rowLimits.empty() && !span.rowLimits.back().first &&
        !span.rowLimits.back().second)
      viable = false;
  };

  scan.walk([&](mlir::Operation *op) {
    if (auto l = mlir::dyn_cast<cir::LoadOp>(op))
      visit(l.getAddr(), /*isWrite=*/false);
    else if (auto s = mlir::dyn_cast<cir::StoreOp>(op))
      visit(s.getAddr(), /*isWrite=*/true);
  });
  if (!viable || !elemTy)
    return std::nullopt;

  mlir::OpBuilder builder(unit.front());
  mlir::Location loc = nest.getLoc();
  mlir::MLIRContext *ctx = nest.getContext();
  auto s64 = cir::IntType::get(ctx, 64, /*isSigned=*/true);
  auto boolTy = cir::BoolType::get(ctx);
  auto elemPtrTy = cir::PointerType::get(elemTy);
  auto s64Const = [&](int64_t v) {
    return cir::ConstantOp::create(builder, loc, cir::IntAttr::get(s64, v))
        .getResult();
  };

  // Materialize each span as a start and one past the end in element units.
  // rows = max over the object's own limits, plus one where the limit is
  // inclusive, folded as one select chain.
  llvm::SmallVector<std::pair<Span, std::pair<mlir::Value, mlir::Value>>, 4>
      extents;
  for (auto &kv : spans) {
    Span &span = kv.second;
    mlir::Value rows;
    for (auto [limit, reach] : span.rowLimits) {
      mlir::Value v;
      if (!limit) {
        v = s64Const(1);
      } else {
        v = recreateInvariant(builder, limit, nest);
        if (!v)
          return std::nullopt;
        if (v.getType() != s64)
          v = cir::CastOp::create(builder, loc, s64, cir::CastKind::integral,
                                  v);
        if (reach)
          v = cir::AddOp::create(builder, loc, s64, v, s64Const(1));
      }
      if (!rows) {
        rows = v;
      } else {
        mlir::Value more =
            cir::CmpOp::create(builder, loc, cir::CmpOpKind::gt, v, rows);
        rows = cir::SelectOp::create(builder, loc, s64, more, v, rows);
      }
    }
    if (span.isParam && !rows)
      return std::nullopt;
    mlir::Value start = span.start;
    if (!start) {
      auto gv = mlir::cast<cir::GlobalOp>(kv.first);
      start = cir::GetGlobalOp::create(
          builder, loc, cir::PointerType::get(gv.getSymType()), gv.getName());
    }
    if (start.getType() != elemPtrTy)
      start = cir::CastOp::create(builder, loc, elemPtrTy,
                                  cir::CastKind::bitcast, start);
    mlir::Value count = span.isParam
                            ? cir::MulOp::create(builder, loc, s64, rows,
                                                 s64Const(span.elemsPerStep))
                                  .getResult()
                            : s64Const(span.fixedElems);
    mlir::Value end =
        cir::PtrStrideOp::create(builder, loc, elemPtrTy, start, count);
    extents.push_back({span, {start, end}});
  }

  // Every pair with a write and a parameter needs a disjointness test. Two
  // statically known objects never do.
  mlir::Value cond;
  mlir::Value yes =
      cir::ConstantOp::create(builder, loc, cir::BoolAttr::get(ctx, true));
  mlir::Value no =
      cir::ConstantOp::create(builder, loc, cir::BoolAttr::get(ctx, false));
  for (unsigned i = 0; i < extents.size(); ++i) {
    for (unsigned j = i + 1; j < extents.size(); ++j) {
      const Span &a = extents[i].first, &b = extents[j].first;
      if (!a.written && !b.written)
        continue;
      if (!a.isParam && !b.isParam)
        continue;
      auto [aStart, aEnd] = extents[i].second;
      auto [bStart, bEnd] = extents[j].second;
      mlir::Value aFirst =
          cir::CmpOp::create(builder, loc, cir::CmpOpKind::le, aEnd, bStart);
      mlir::Value bFirst =
          cir::CmpOp::create(builder, loc, cir::CmpOpKind::le, bEnd, aStart);
      mlir::Value apart =
          cir::SelectOp::create(builder, loc, boolTy, aFirst, yes, bFirst);
      cond = cond ? cir::SelectOp::create(builder, loc, boolTy, cond, apart, no)
                        .getResult()
                  : apart;
    }
  }
  if (!cond)
    return std::nullopt; // nothing needed a test so versioning is pointless

  auto ifOp = cir::IfOp::create(
      builder, loc, cond, /*withElseRegion=*/true,
      [](mlir::OpBuilder &b, mlir::Location l) { cir::YieldOp::create(b, l); },
      [](mlir::OpBuilder &b, mlir::Location l) { cir::YieldOp::create(b, l); });

  mlir::IRMapping map;
  mlir::OpBuilder thenBuilder(ifOp.getThenRegion().front().getTerminator());
  cir::ForOp cloned;
  for (mlir::Operation *op : unit) {
    mlir::Operation *copy = thenBuilder.clone(*op, map);
    if (op == nest.getOperation())
      cloned = mlir::cast<cir::ForOp>(copy);
  }
  mlir::Operation *elseYield = ifOp.getElseRegion().front().getTerminator();
  for (mlir::Operation *op : unit)
    op->moveBefore(elseYield);

  llvm::SmallVector<CountedLoop> copyNest = collectPerfectNest(cloned);
  if (copyNest.size() < 2)
    return std::nullopt; // recognition of a fresh clone cannot really fail
  return std::make_pair(copyNest[copyNest.size() - 2],
                        copyNest[copyNest.size() - 1]);
}

// Loop distribution
//
// An imperfect nest cannot be interchanged as it stands. When every statement
// beside the inner loop can get its own copy of the outer loop without
// reversing a dependence, the nest is split and the now perfect middle pair
// interchanged. The split is speculative. The copies are built beside the
// original and the original is erased only once the interchange has succeeded.

// One memory access of a statement group.
struct DistAccess {
  mlir::Operation *baseId;
  llvm::SmallVector<DimComp, 4> dims;
  bool write;
};

// The start the inner loop is known to stay at or above, as slot plus offset.
static void innerStartFact(LoopParts inner, cir::ForOp jFor, mlir::Value &slot,
                           int64_t &off) {
  slot = {};
  off = 0;
  if (!stepsByOne(inner.forOp, inner.ivSlot))
    return;
  mlir::Value v = inner.init.getValue();
  int64_t c = 0;
  if (auto add = v.getDefiningOp<cir::AddOp>()) {
    auto lhs = add.getLhs().getDefiningOp<cir::ConstantOp>();
    auto rhs = add.getRhs().getDefiningOp<cir::ConstantOp>();
    auto konst = rhs ? rhs : lhs;
    auto attr =
        konst ? mlir::dyn_cast<cir::IntAttr>(konst.getValue()) : cir::IntAttr();
    if (!attr)
      return;
    c = attr.getValue().getSExtValue();
    v = rhs ? add.getLhs() : add.getRhs();
  }
  auto load = v.getDefiningOp<cir::LoadOp>();
  if (!load || !load.getAddr().getDefiningOp<cir::AllocaOp>() ||
      !slotUnwrittenIn(jFor, load.getAddr()))
    return;
  slot = load.getAddr();
  off = c;
}

// Where the split happens and what goes where. Indices are positions of the
// statement level ops in stmtBlock, and path descends from the outer loop
// body to stmtBlock so the same positions can be found again in a clone.
struct DistPlan {
  mlir::Block *stmtBlock;
  llvm::SmallVector<unsigned, 8> pre, mid, post;
  llvm::SmallVector<unsigned, 2> path; // op index per level, region 0 front
};

static std::optional<DistPlan>
planDistribution(LoopParts outer, LoopParts inner, CountedLoop outerCL,
                 CountedLoop innerCL, bool assume) {
  cir::ForOp jFor = outer.forOp;

  // The inner loop unit, climbing through scopes that hold nothing else, so
  // the split point sits at the statement level.
  llvm::SmallPtrSet<mlir::Operation *, 4> midSet;
  midSet.insert(inner.forOp.getOperation());
  midSet.insert(inner.init.getOperation());
  midSet.insert(inner.initDef);
  if (inner.ownsAlloca)
    midSet.insert(inner.alloca);
  // A pure value op whose users all sit in the unit belongs with it, such as
  // the i + 1 chain feeding a start. Its execution time is unobservable
  // except through those uses. Grown to a fixed point per level.
  auto absorbFeeders = [&](mlir::Block *blk) {
    bool grew = true;
    while (grew) {
      grew = false;
      for (mlir::Operation &op : llvm::reverse(*blk)) {
        if (midSet.contains(&op) ||
            op.hasTrait<mlir::OpTrait::IsTerminator>() || op.use_empty())
          continue;
        bool pure = mlir::isMemoryEffectFree(&op);
        if (!pure)
          if (auto load = mlir::dyn_cast<cir::LoadOp>(&op))
            pure = !load.getIsVolatile() && !load.getMemOrder() &&
                   load.getAddr().getDefiningOp<cir::AllocaOp>() &&
                   slotUnwrittenIn(jFor, load.getAddr());
        if (!pure)
          continue;
        bool allInUnit =
            llvm::all_of(op.getUsers(), [&](mlir::Operation *user) {
              while (user && user->getBlock() != blk)
                user = user->getParentOp();
              return user && midSet.contains(user);
            });
        if (allInUnit) {
          midSet.insert(&op);
          grew = true;
        }
      }
    }
  };
  mlir::Operation *anchor = inner.forOp.getOperation();
  while (true) {
    mlir::Block *blk = anchor->getBlock();
    absorbFeeders(blk);
    bool unitOnly = llvm::all_of(*blk, [&](mlir::Operation &op) {
      return midSet.contains(&op) || op.hasTrait<mlir::OpTrait::IsTerminator>();
    });
    mlir::Operation *up = blk->getParentOp();
    if (!unitOnly || up == jFor.getOperation() || !mlir::isa<cir::ScopeOp>(up))
      break;
    anchor = up;
    midSet.clear();
    midSet.insert(anchor);
  }
  mlir::Block *stmtBlock = anchor->getBlock();
  absorbFeeders(stmtBlock);

  // The statement block must hang off the loop body through lone scopes.
  llvm::SmallVector<unsigned, 2> path;
  for (mlir::Block *blk = stmtBlock;
       blk->getParentOp() != jFor.getOperation();) {
    mlir::Operation *up = blk->getParentOp();
    if (!mlir::isa<cir::ScopeOp>(up) || up->getNumRegions() != 1 ||
        !up->getRegion(0).hasOneBlock())
      return std::nullopt;
    unsigned idx = 0;
    for (mlir::Operation &op : *up->getBlock()) {
      if (&op == up)
        break;
      ++idx;
    }
    path.push_back(idx);
    blk = up->getBlock();
    if (blk != &jFor.getBody().front())
      if (blk->getParentOp() != jFor.getOperation())
        continue;
  }
  std::reverse(path.begin(), path.end());

  // Partition the statement ops around the inner unit.
  DistPlan plan;
  plan.stmtBlock = stmtBlock;
  plan.path = path;
  bool seenMid = false, pastMid = false;
  unsigned idx = 0;
  for (mlir::Operation &op : *stmtBlock) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      break;
    if (midSet.contains(&op)) {
      if (pastMid)
        return std::nullopt;
      seenMid = true;
      plan.mid.push_back(idx);
    } else if (!seenMid) {
      plan.pre.push_back(idx);
    } else {
      pastMid = true;
      plan.post.push_back(idx);
    }
    ++idx;
  }
  if (plan.pre.empty() && plan.post.empty())
    return std::nullopt; // the imperfection is somewhere else

  // Values may not cross group boundaries except through memory.
  llvm::DenseMap<mlir::Operation *, unsigned> groupOf;
  llvm::SmallVector<mlir::Operation *, 8> stmts;
  for (mlir::Operation &op : *stmtBlock)
    if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
      stmts.push_back(&op);
  for (unsigned g : plan.pre)
    groupOf[stmts[g]] = 0;
  for (unsigned g : plan.mid)
    groupOf[stmts[g]] = 1;
  for (unsigned g : plan.post)
    groupOf[stmts[g]] = 2;
  auto topLevel = [&](mlir::Operation *op) -> mlir::Operation * {
    while (op && op->getBlock() != stmtBlock)
      op = op->getParentOp();
    return op;
  };
  for (mlir::Operation *stmt : stmts) {
    unsigned g = groupOf[stmt];
    bool closed = true;
    stmt->walk([&](mlir::Operation *op) {
      for (mlir::Value v : op->getOperands()) {
        mlir::Operation *def = topLevel(v.getDefiningOp());
        if (def && groupOf.count(def) && groupOf[def] != g)
          closed = false;
      }
      for (mlir::Value v : op->getResults())
        for (mlir::Operation *user : v.getUsers()) {
          mlir::Operation *use = topLevel(user);
          if (!use || !groupOf.count(use) || groupOf[use] != g)
            closed = false;
        }
    });
    if (!closed)
      return std::nullopt;
  }

  // Nothing the loop control reads may be written by any group.
  llvm::SmallPtrSet<mlir::Value, 4> controlSlots;
  auto collectControl = [&](mlir::Region &region) {
    region.walk([&](cir::LoadOp load) { controlSlots.insert(load.getAddr()); });
  };
  collectControl(jFor.getCond());
  collectControl(jFor.getStep());

  // Collect every access per group and check each reversed pair.
  llvm::SmallVector<DistAccess, 8> acc[3];
  bool viable = true;
  for (mlir::Operation *stmt : stmts) {
    unsigned g = groupOf[stmt];
    stmt->walk([&](mlir::Operation *op) {
      mlir::Value addr;
      bool isWrite = false;
      if (auto l = mlir::dyn_cast<cir::LoadOp>(op)) {
        addr = l.getAddr();
      } else if (auto st = mlir::dyn_cast<cir::StoreOp>(op)) {
        addr = st.getAddr();
        isWrite = true;
      } else {
        return;
      }
      llvm::SmallVector<mlir::Value, 4> idxs;
      mlir::Value base = peelAddress(addr, idxs);
      mlir::Operation *id = baseObjectId(base, assume);
      if (!id) {
        viable = false;
        return;
      }
      if (isWrite && controlSlots.contains(addr)) {
        viable = false; // would change the trip set of the split loops
        return;
      }
      DistAccess access{id, {}, isWrite};
      for (mlir::Value idx : idxs)
        access.dims.push_back(
            dimCompOf(idx, outerCL.ivSlot, innerCL.ivSlot, jFor));
      acc[g].push_back(access);
    });
  }
  if (!viable)
    return std::nullopt;

  mlir::Value startSlot;
  int64_t startOff;
  innerStartFact(inner, jFor, startSlot, startOff);
  for (unsigned early = 0; early < 3; ++early)
    for (unsigned late = early + 1; late < 3; ++late)
      for (const DistAccess &a : acc[early])
        for (const DistAccess &b : acc[late])
          if (a.baseId == b.baseId && (a.write || b.write) &&
              subscriptsCanMeet(a.dims, b.dims, /*crossIteration=*/true,
                                startSlot, startOff))
            return std::nullopt;
  return plan;
}

// Split the outer loop of the pair by the plan, interchange the middle pair,
// and erase the original. On any failure the copies are erased instead and
// the original is left exactly as it was.
static bool distributeAndInterchange(CountedLoop outerCL, CountedLoop innerCL,
                                     bool assume) {
  std::optional<LoopParts> outer = getLoopParts(outerCL);
  std::optional<LoopParts> inner = getLoopParts(innerCL);
  if (!outer || !inner)
    return false;
  std::optional<DistPlan> plan =
      planDistribution(*outer, *inner, outerCL, innerCL, assume);
  if (!plan)
    return false;

  llvm::SmallVector<mlir::Operation *, 4> unit;
  if (outer->ownsAlloca)
    unit.push_back(outer->alloca);
  unit.push_back(outer->initDef);
  unit.push_back(outer->init.getOperation());
  unit.push_back(outer->forOp.getOperation());
  llvm::sort(unit, [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });
  llvm::SmallPtrSet<mlir::Operation *, 4> inUnit;
  inUnit.insert_range(unit);
  for (mlir::Operation *op = unit.front(); op != outer->forOp.getOperation();
       op = op->getNextNode())
    if (!inUnit.contains(op))
      return false;

  mlir::OpBuilder builder(unit.back()->getBlock(),
                          ++mlir::Block::iterator(unit.back()));
  llvm::SmallVector<mlir::Operation *, 12> created;
  auto cloneKeeping = [&](llvm::ArrayRef<unsigned> keep) -> cir::ForOp {
    mlir::IRMapping map;
    cir::ForOp copy;
    for (mlir::Operation *op : unit) {
      mlir::Operation *c = builder.clone(*op, map);
      created.push_back(c);
      if (op == outer->forOp.getOperation())
        copy = mlir::cast<cir::ForOp>(c);
    }
    mlir::Block *blk = &copy.getBody().front();
    for (unsigned idx : plan->path)
      blk = &std::next(blk->begin(), idx)->getRegion(0).front();
    llvm::SmallVector<mlir::Operation *, 8> stmts;
    for (mlir::Operation &op : *blk)
      if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
        stmts.push_back(&op);
    llvm::SmallPtrSet<mlir::Operation *, 8> keepSet;
    for (unsigned k : keep)
      keepSet.insert(stmts[k]);
    for (mlir::Operation *op : llvm::reverse(stmts))
      if (!keepSet.contains(op))
        op->erase();
    return copy;
  };

  if (!plan->pre.empty())
    cloneKeeping(plan->pre);
  cir::ForOp midFor = cloneKeeping(plan->mid);
  if (!plan->post.empty())
    cloneKeeping(plan->post);

  llvm::SmallVector<CountedLoop> nest = collectPerfectNest(midFor);
  const char *why = nest.size() < 2
                        ? "the split middle nest was not recognized"
                        : tryInterchangePair(nest[nest.size() - 2],
                                             nest[nest.size() - 1], assume);
  if (why) {
    for (mlir::Operation *op : llvm::reverse(created))
      op->erase();
    return false;
  }
  for (mlir::Operation *op : llvm::reverse(unit))
    op->erase();
  return true;
}

// Pass

struct LoopOptPass : public impl::LoopOptBase<LoopOptPass> {
  using LoopOptBase::LoopOptBase;
  void runOnOperation() override;
};

void LoopOptPass::runOnOperation() {
  llvm::TimeTraceScope scope("CIR Loop Opt");

  // Collect nest heads first because transforming while walking is unsafe.
  // A loop heads a nest unless it is the sole child of its parent loop.
  llvm::SmallVector<cir::ForOp> roots;
  getOperation()->walk([&](cir::ForOp forOp) {
    cir::ForOp parent = forOp->getParentOfType<cir::ForOp>();
    if (!parent) {
      roots.push_back(forOp);
      return;
    }
    llvm::SmallVector<cir::ForOp> kids = immediateChildLoops(parent);
    if (kids.size() != 1 || kids.front() != forOp)
      roots.push_back(forOp);
  });

  for (cir::ForOp root : roots) {
    llvm::SmallVector<CountedLoop> nest = collectPerfectNest(root);
    if (nest.empty())
      continue;

    if (testAnnotate) {
      mlir::Builder b(&getContext());
      root->setAttr("cir.loopopt.nest_depth",
                    b.getI64IntegerAttr((int64_t)nest.size()));
      continue; // recognition only mode never transforms
    }

    if (nest.size() < 2) {
      if (report)
        mlir::emitRemark(root.getLoc())
            << "loop interchange declined, no counted pair was recognized "
               "around this loop";
      continue;
    }

    // Interchange the innermost pair.
    CountedLoop pairOuter = nest[nest.size() - 2];
    CountedLoop pairInner = nest[nest.size() - 1];
    const char *why = tryInterchangePair(pairOuter, pairInner);
    bool versioned = false, distributed = false;
    // When unknown aliasing is the one obstacle, split on a runtime test.
    // Otherwise report what still blocks the pair once aliasing is testable,
    // which is the answer that matters.
    if (why == kUnknownAliasing) {
      const char *deeper = tryInterchangePair(
          pairOuter, pairInner, /*assumeParamsDistinct=*/true, /*dryRun=*/true);
      if (deeper) {
        why = deeper;
      } else if (auto copy = versionNest(pairOuter, pairInner,
                                         pairInner.forOp.getBody())) {
        why = tryInterchangePair(copy->first, copy->second,
                                 /*assumeParamsDistinct=*/true);
        versioned = !why;
      }
    } else if (why == kImperfectNest) {
      // Distribution can make the pair perfect. Only worth doing when the
      // interchange then fires, so ask that question first on the nest as
      // the split would leave it.
      const char *deeper =
          tryInterchangePair(pairOuter, pairInner, /*assumeParamsDistinct=*/
                             false, /*dryRun=*/true, /*ignoreImperfect=*/true);
      bool needsParams = false;
      if (deeper == kUnknownAliasing) {
        deeper = tryInterchangePair(pairOuter, pairInner, true, true, true);
        needsParams = true;
      }
      if (!deeper) {
        if (!needsParams) {
          distributed = distributeAndInterchange(pairOuter, pairInner, false);
        } else if (auto copy = versionNest(pairOuter, pairInner,
                                           pairOuter.forOp.getBody())) {
          distributed =
              distributeAndInterchange(copy->first, copy->second, true);
          versioned = distributed;
        }
        if (distributed)
          why = nullptr;
      } else {
        why = deeper;
      }
    }
    if (report) {
      llvm::Twine how =
          distributed
              ? (versioned ? "loop interchange applied after distributing "
                             "the nest behind a runtime aliasing test"
                           : "loop interchange applied after distributing "
                             "the nest")
          : versioned
              ? "loop interchange applied behind a runtime aliasing test"
              : "loop interchange applied";
      mlir::emitRemark(pairOuter.forOp.getLoc())
          << (why ? llvm::Twine("loop interchange declined, ") + why : how);
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createLoopOptPass() {
  return std::make_unique<LoopOptPass>();
}
