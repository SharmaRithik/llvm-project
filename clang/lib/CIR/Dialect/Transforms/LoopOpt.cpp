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

// A stable identity for a base object used to reason about aliasing. Distinct
// allocas and distinct globals denote distinct objects. Returns the defining
// alloca or the resolved global definition, or null when the base has unknown
// aliasing such as a pointer parameter.
static mlir::Operation *baseObjectId(mlir::Value base) {
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
static bool
isInterchangeLegal(mlir::Value p, mlir::Value q, mlir::Region &innerBody,
                   cir::ForOp nest,
                   llvm::SmallVectorImpl<mlir::Operation *> &reductions) {
  // Bail on any op the dependence analysis cannot model.
  if (hasUnmodeledEffect(innerBody))
    return false;

  // Per base we track the shared dimension tag shape and whether it is written.
  struct BaseInfo {
    llvm::SmallVector<IdxTag, 4> shape;
    bool written = false;
    bool seen = false;
  };
  llvm::DenseMap<mlir::Operation *, BaseInfo> bases;
  bool bail = false;

  auto handle = [&](mlir::Value addr, bool isWrite) {
    if (bail)
      return;
    llvm::SmallVector<mlir::Value, 4> idxs;
    mlir::Value base = peelAddress(addr, idxs);
    mlir::Operation *id = baseObjectId(base);
    if (!id) {
      bail = true; // unknown aliasing such as a pointer parameter
      return;
    }
    llvm::SmallVector<IdxTag, 4> shape;
    for (mlir::Value idx : idxs) {
      IdxTag t = classifyIndex(idx, p, q, innerBody);
      if (t == IdxTag::Unknown) {
        bail = true;
        return;
      }
      shape.push_back(t);
    }
    auto &info = bases[id];
    if (!info.seen) {
      info.seen = true;
      info.shape = shape;
    } else if (info.shape != shape) {
      bail = true; // same base with inconsistent shapes so be safe
      return;
    }
    info.written |= isWrite;
  };

  innerBody.walk([&](mlir::Operation *op) {
    if (auto l = mlir::dyn_cast<cir::LoadOp>(op))
      handle(l.getAddr(), /*isWrite=*/false);
    else if (auto s = mlir::dyn_cast<cir::StoreOp>(op))
      handle(s.getAddr(), /*isWrite=*/true);
  });

  if (bail)
    return false;

  // Every written base must vary with P or Q, or be an order-free reduction.
  for (auto &kv : bases) {
    const BaseInfo &info = kv.second;
    if (!info.written)
      continue;
    bool usesPQ = false;
    for (IdxTag t : info.shape)
      if (t == IdxTag::P || t == IdxTag::Q)
        usesPQ = true;
    if (!usesPQ) {
      mlir::Operation *combine = orderFreeReduction(kv.first, nest);
      if (!combine)
        return false;
      reductions.push_back(combine);
    }
  }
  return true;
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
};

static std::optional<LoopParts> getLoopParts(const CountedLoop &cl) {
  auto alloca = cl.ivSlot.getDefiningOp<cir::AllocaOp>();
  if (!alloca)
    return std::nullopt;
  mlir::Block *block = cl.forOp->getBlock();
  // The IV alloca must live in the loop block so it moves with the loop.
  if (alloca->getBlock() != block)
    return std::nullopt;

  cir::StoreOp init;
  for (auto store : block->getOps<cir::StoreOp>()) {
    if (store.getAddr() != cl.ivSlot)
      continue;
    if (!store->isBeforeInBlock(cl.forOp))
      continue;
    if (init) // more than one init store so bail
      return std::nullopt;
    init = store;
  }
  if (!init)
    return std::nullopt;
  mlir::Operation *initDef = init.getValue().getDefiningOp();
  // The init value moves with the store. A constant always survives that. A
  // load does only while its slot holds the same value, which the caller
  // checks against the whole nest.
  if (!initDef || !mlir::isa<cir::ConstantOp, cir::LoadOp>(initDef) ||
      !initDef->hasOneUse())
    return std::nullopt;
  if (initDef->getBlock() != block)
    return std::nullopt;
  return LoopParts{cl.forOp, cl.ivSlot, alloca, init, initDef};
}

// True when the init value reads the same thing from its new position.
static bool initSurvivesMove(const LoopParts &lp, cir::ForOp nest) {
  auto load = mlir::dyn_cast<cir::LoadOp>(lp.initDef);
  if (!load)
    return true;
  // Moving a volatile or atomic read across a loop boundary changes how many
  // times it runs, which is observable.
  if (load.getIsVolatile() || load.getMemOrder())
    return false;
  bool written = false;
  nest.walk([&](cir::StoreOp store) {
    if (store.getAddr() == load.getAddr())
      written = true;
  });
  return !written;
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
static void moveIvControlBefore(LoopParts lp, mlir::Operation *target) {
  lp.alloca->moveBefore(target);
  lp.initDef->moveBefore(target);
  lp.init->moveBefore(target);
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
  moveIvControlBefore(inner, fOuter);
  moveIvControlBefore(outer, fInner);
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
  if (!exitsOnLessThan(outer.forOp) || !exitsOnLessThan(inner.forOp))
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
  mlir::Type type = qBound.getType();
  if (!isEmittable(type, *scale * (*bound - 1) + *offset) ||
      !isEmittable(type, 1 - *offset) || !isEmittable(type, *scale) ||
      !isEmittable(type, *start))
    return std::nullopt;
  return UpperTriangle{*bound, *scale, *offset, *start};
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
      cir::IntAttr::get(type, tri.scale * (tri.outerBound - 1) + tri.offset)));
  eraseIfDead(oldBoundValue);

  mlir::Location loc = outer.init.getLoc();
  builder.setInsertionPoint(outer.init);
  mlir::Value index = cir::LoadOp::create(builder, loc, {inner.ivSlot});
  mlir::Value begin;
  int64_t lowest;
  if (tri.scale == 1) {
    mlir::Value shift = cir::ConstantOp::create(
        builder, loc, cir::IntAttr::get(type, 1 - tri.offset));
    begin = cir::AddOp::create(builder, loc, type, index, shift);
    lowest = 1 - tri.offset;
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
  auto load = inner.init.getValue().getDefiningOp<cir::LoadOp>();
  if (!load || load.getAddr() != outer.ivSlot)
    return std::nullopt;
  std::optional<int64_t> outerBound = foldConstant(pBound);
  std::optional<int64_t> innerBound = foldConstant(qBound);
  std::optional<int64_t> start = foldConstant(outer.init.getValue());
  if (!outerBound || !innerBound || !start || *innerBound > *outerBound)
    return std::nullopt;
  if (!isEmittable(inner.init.getValue().getType(), *start))
    return std::nullopt;
  return LowerTriangle{*start};
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
      builder, inner.init.getLoc(), cir::IntAttr::get(type, tri.outerStart)));
  eraseIfDead(oldStart);

  cir::CmpOp cmp = getExitCmp(inner.forOp);
  mlir::Value oldBound = cmp.getRhs();
  builder.setInsertionPoint(cmp);
  mlir::Value index =
      cir::LoadOp::create(builder, cmp.getLoc(), {inner.ivSlot});
  mlir::Value one = cir::ConstantOp::create(builder, cmp.getLoc(),
                                            cir::IntAttr::get(type, 1));
  cmp.getRhsMutable().assign(
      cir::AddOp::create(builder, cmp.getLoc(), type, index, one));
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
static bool tryInterchangePair(CountedLoop outerCL, CountedLoop innerCL) {
  mlir::Region &innerBody = innerCL.forOp.getBody();

  // Only handle a genuinely innermost inner loop.
  bool hasDeeper = false;
  innerBody.walk([&](cir::ForOp f) {
    if (f != innerCL.forOp)
      hasDeeper = true;
  });
  if (hasDeeper)
    return false;

  std::optional<LoopParts> outer = getLoopParts(outerCL);
  std::optional<LoopParts> inner = getLoopParts(innerCL);
  if (!outer || !inner)
    return false;

  // Genuine perfect nest with nothing but the inner loop between headers.
  if (!isPerfectPair(outerCL.forOp, *inner))
    return false;

  // Rectangular bounds, or one of the two triangular forms rebuilt below. Any
  // other coupling between a bound and the other induction variable would not
  // survive swapping the cond regions.
  mlir::Value pBound = getLoopBoundRHS(outerCL.forOp);
  mlir::Value qBound = getLoopBoundRHS(innerCL.forOp);
  if (!pBound || !qBound)
    return false;
  std::optional<ProductNest> product =
      recognizeProductNest(*outer, *inner, pBound, qBound);
  std::optional<UpperTriangle> triangle;
  std::optional<LowerTriangle> lower;
  if (!product) {
    // Every other path copies the cond regions across, so each exit test has
    // to name only its own induction variable.
    if (!exitTestLhsIsIv(outerCL.forOp, outerCL.ivSlot) ||
        !exitTestLhsIsIv(innerCL.forOp, innerCL.ivSlot))
      return false;
    triangle = recognizeUpperTriangle(*outer, *inner, pBound, qBound);
    if (!triangle)
      lower = recognizeLowerTriangle(*outer, *inner, pBound, qBound);
    if (!triangle && !lower &&
        (valueDependsOnSlot(pBound, innerCL.ivSlot) ||
         valueDependsOnSlot(qBound, outerCL.ivSlot)))
      return false;

    // The lower form replaces the inner start outright, so it is the one case
    // that does not have to carry that value to its new position.
    if (!lower && (!initSurvivesMove(*outer, outerCL.forOp) ||
                   !initSurvivesMove(*inner, outerCL.forOp)))
      return false;
  }

  llvm::SmallVector<mlir::Operation *, 2> reductions;
  if (!isInterchangeLegal(outerCL.ivSlot, innerCL.ivSlot, innerBody,
                          outerCL.forOp, reductions))
    return false;
  if (!isInterchangeProfitable(outerCL.ivSlot, innerCL.ivSlot, innerBody))
    return false;

  interchange(*outer, *inner);
  relaxWrapFlags(reductions);
  if (product)
    rewriteProductNest(*outer, *inner, *product);
  else if (triangle)
    rewriteUpperTriangle(*outer, *inner, *triangle);
  else if (lower)
    rewriteLowerTriangle(*outer, *inner, *lower);
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

    if (nest.size() < 2)
      continue;

    // Interchange the innermost pair.
    tryInterchangePair(nest[nest.size() - 2], nest[nest.size() - 1]);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createLoopOptPass() {
  return std::make_unique<LoopOptPass>();
}
