//===- MatmulCompiler.cpp - Compile a matmul to WGSL ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelOutlining.h"
#include "StructuredSchedule.h"
#include "SharedMemorySchedule.h"
#include "RuntimeBindings.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Target/WGSL/MatmulCompiler.h"
#include "mlir/Target/WGSL/WGSLEmitter.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>

using namespace mlir;

namespace {
struct Matmul {
  func::FuncOp function;
  linalg::MatmulTransposeBOp operation;
  int64_t m, n, k;
  Type rhsType;
  Type accType;
  bool rhsTruncated;

  bool isGemv() const { return n <= 4 && accType.isF32() && !rhsTruncated; }
};

template <typename CastOp>
bool isPointwiseCast(linalg::GenericOp op) {
  if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      !op.getOutputs()[0].getDefiningOp<tensor::EmptyOp>() ||
      op.getNumLoops() != 2 || op.getNumParallelLoops() != 2 ||
      llvm::any_of(op.getIndexingMapsArray(), [](AffineMap m) { return !m.isIdentity(); }))
    return false;
  Block &body = op.getRegion().front();
  auto cast = dyn_cast<CastOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  return body.getOperations().size() == 2 && cast && yield &&
         cast.getIn() == body.getArgument(0) &&
         yield.getValues()[0] == cast.getResult();
}

// Restrict the body to the supported precision, then specialize it.
FailureOr<linalg::MatmulTransposeBOp>
specializeMatmul(IRRewriter &b, linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 2 ||
      generic.getNumDpsInits() != 1 || generic.getNumResults() != 1 ||
      !linalg::isaContractionOpInterface(generic))
    return failure();
  Block &body = generic.getRegion().front();
  Type accType = body.getArgument(2).getType();
  if (!accType.isF16() && !accType.isF32())
    return failure();
  unsigned expectedCasts = 0, casts = 0, multiplies = 0, adds = 0;
  for (Value input : body.getArguments().take_front(2)) {
    if (input.getType() == accType)
      continue;
    if (!input.getType().isF16() || !accType.isF32())
      return failure();
    ++expectedCasts;
  }
  for (Operation &op : body) {
    if (auto flags = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
      if (flags.getValue() != arith::FastMathFlags::none)
        return failure();
    if (auto extend = dyn_cast<arith::ExtFOp>(op)) {
      if ((extend.getIn() != body.getArgument(0) &&
           extend.getIn() != body.getArgument(1)) ||
          !extend.getIn().getType().isF16() ||
          extend.getType() != accType || !accType.isF32())
        return failure();
      ++casts;
    } else if (auto multiply = dyn_cast<arith::MulFOp>(op)) {
      if (multiply.getType() != accType)
        return failure();
      ++multiplies;
    } else if (auto add = dyn_cast<arith::AddFOp>(op)) {
      if (add.getType() != accType)
        return failure();
      ++adds;
    } else if (!isa<linalg::YieldOp>(op)) {
      return failure();
    }
  }
  if (casts != expectedCasts || multiplies != 1 || adds != 1)
    return failure();
  // Also puts permuted loops in the named op's order.
  b.setInsertionPoint(generic);
  auto specialized = linalg::specializeGenericOp(b, generic);
  if (failed(specialized))
    return failure();
  auto matmul = dyn_cast<linalg::MatmulTransposeBOp>(specialized->getOperation());
  if (!matmul)
    return failure();
  return matmul;
}

// Match the whole graph and reject any extra computation.
FailureOr<Matmul> matchMatmul(ModuleOp module) {
  auto functions = llvm::to_vector(module.getOps<func::FuncOp>());
  if (functions.size() != 1 || !llvm::hasSingleElement(module.getBody()->getOperations()))
    return module.emitError("expected one matmul function"), failure();
  auto f = functions.front();
  if (f.isExternal() || !f.getBody().hasOneBlock() || f.getNumArguments() != 2 ||
      f.getNumResults() != 1)
    return f.emitError("expected two tensor arguments and one tensor result"), failure();
  auto a = dyn_cast<RankedTensorType>(f.getArgument(0).getType());
  auto b = dyn_cast<RankedTensorType>(f.getArgument(1).getType());
  auto c = dyn_cast<RankedTensorType>(f.getResultTypes()[0]);
  if (!a || !b || !c || !a.hasStaticShape() || !b.hasStaticShape() ||
      !c.hasStaticShape() || a.getRank() != 2 || b.getRank() != 2 ||
      c.getRank() != 2 || !a.getElementType().isF16() ||
      !(b.getElementType().isF16() || b.getElementType().isF32()) ||
      !c.getElementType().isF32())
    return f.emitError("expected static rank-2 f16 x (f16 or f32) -> f32 tensors"), failure();
  int64_t m = a.getDimSize(0), n = b.getDimSize(0), k = a.getDimSize(1);
  if (b.getDimSize(1) != k || c.getDimSize(0) != m || c.getDimSize(1) != n ||
      m <= 0 || n <= 0 || k <= 0 ||
      m > INT32_MAX / k || n > INT32_MAX / k || m > INT32_MAX / n)
    return f.emitError("incompatible or unsupported matmul dimensions"), failure();
  auto ret = dyn_cast<func::ReturnOp>(f.getBody().front().getTerminator());
  Value returned = ret && ret.getNumOperands() == 1 ? ret.getOperand(0) : Value();
  auto extend = returned ? returned.getDefiningOp<linalg::GenericOp>() : linalg::GenericOp();
  if (extend && extend.getNumDpsInputs() != 1)
    extend = {};
  if (extend) {
    if (!isPointwiseCast<arith::ExtFOp>(extend) ||
        !getElementTypeOrSelf(extend.getInputs()[0]).isF16())
      return extend.emitError("expected a pointwise f16-to-f32 result conversion"), failure();
    returned = extend.getInputs()[0];
  }
  auto mm = returned ? returned.getDefiningOp<linalg::MatmulTransposeBOp>()
                     : linalg::MatmulTransposeBOp();
  if (!mm && returned) {
    IRRewriter rewriter(module.getContext());
    auto specialized = specializeMatmul(
        rewriter, returned.getDefiningOp<linalg::GenericOp>());
    if (succeeded(specialized)) {
      mm = *specialized;
      returned = mm.getResult(0);
    }
  }
  if (!mm || mm.getInputs()[0] != f.getArgument(0))
    return f.emitError("expected return of a transpose-B linalg.matmul"), failure();
  AffineExpr dm, dn, dk;
  bindDims(module.getContext(), dm, dn, dk);
  auto maps = AffineMap::inferFromExprList(ArrayRef<ArrayRef<AffineExpr>>{{dm, dk}, {dn, dk}, {dm, dn}}, module.getContext());
  if (mm.getIndexingMapsArray() != maps)
    return mm.emitError("unsupported matmul indexing maps"), failure();
  auto fill = mm.getOutputs()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_PosZeroFloat()) ||
      !fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
    return mm.emitError("matmul accumulator must be initialized to zero"), failure();
  Value rhs = mm.getInputs()[1];
  linalg::GenericOp cast;
  if (b.getElementType().isF32() && rhs != f.getArgument(1)) {
    cast = rhs.getDefiningOp<linalg::GenericOp>();
    if (!cast || cast.getNumDpsInputs() != 1 || cast.getNumDpsInits() != 1 ||
        cast.getInputs()[0] != f.getArgument(1) ||
        !cast.getOutputs()[0].getDefiningOp<tensor::EmptyOp>() ||
        !getElementTypeOrSelf(rhs).isF16() || cast.getNumLoops() != 2 ||
        cast.getNumParallelLoops() != 2 ||
        llvm::any_of(cast.getIndexingMapsArray(), [](AffineMap map) { return !map.isIdentity(); }))
      return mm.emitError("f32 RHS requires an explicit pointwise truncation to f16"), failure();
    Block &body = cast.getRegion().front();
    auto trunc = dyn_cast<arith::TruncFOp>(body.front());
    auto yield = dyn_cast<linalg::YieldOp>(body.back());
    if (body.getOperations().size() != 2 || !trunc || !yield ||
        trunc.getIn() != body.getArgument(0) || yield.getValues()[0] != trunc.getResult())
      return cast.emitError("only a pointwise f32-to-f16 cast is supported"), failure();
  } else if (rhs != f.getArgument(1)) {
    return mm.emitError("unexpected RHS computation"), failure();
  }
  if (b.getElementType().isF32() && !cast &&
      (n > 4 || !getElementTypeOrSelf(returned).isF32()))
    return mm.emitError("unconverted f32 RHS requires N <= 4 and f32 accumulation"), failure();
  for (Operation &op : f.getBody().front()) {
    if (&op == mm || &op == fill || &op == ret || (cast && &op == cast) ||
        (extend && &op == extend) ||
        isa<arith::ConstantOp, tensor::EmptyOp>(op))
      continue;
    return op.emitError("unsupported computation in matmul function"), failure();
  }
  return Matmul{f, mm, m, n, k, b.getElementType(), getElementTypeOrSelf(returned), bool(cast)};
}

struct KernelBuilder {
  IRRewriter &b;
  Location loc;
  Value tid;
  Value c(int64_t value) { return arith::ConstantIndexOp::create(b, loc, value); }
  Value add(Value x, Value y) { return arith::AddIOp::create(b, loc, x, y); }
  Value mul(Value x, Value y) { return arith::MulIOp::create(b, loc, x, y); }
  Value div(Value x, Value y) { return arith::DivUIOp::create(b, loc, x, y); }
  Value rem(Value x, Value y) { return arith::RemUIOp::create(b, loc, x, y); }
  Value lt(Value x, Value y) { return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, x, y); }
  void barrier() {
    // Storage writes follow all shared-memory barriers.
    auto op = gpu::BarrierOp::create(b, loc);
    op.setAddressSpacesAttr(b.getArrayAttr(
        {gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup)}));
  }
};

// Vectorize the tile with upstream Linalg, keeping the accumulator in SSA.
FailureOr<Value> vectorizeTile(IRRewriter &b, Location loc,
                              linalg::MatmulTransposeBOp original, Value a,
                              Value rhs, Value accumulator) {
  OpBuilder::InsertionGuard guard(b);
  auto insertionPoint = b.saveInsertionPoint();
  auto type = cast<VectorType>(accumulator.getType());
  auto outType = MemRefType::get(type.getShape(), type.getElementType());
  auto fn = func::FuncOp::create(b, loc, "__wgsl_vectorize_tile",
      b.getFunctionType({a.getType(), rhs.getType(), outType}, {}));
  Block *block = fn.addEntryBlock();
  b.setInsertionPointToStart(block);
  auto op = mlir::clone(b, original, TypeRange{}, block->getArguments());
  func::ReturnOp::create(b, loc);
  b.setInsertionPoint(op);
  if (failed(linalg::vectorize(b, op, {}, {}, false, false, false, true))) {
    fn.erase();
    return failure();
  }
  b.eraseOp(op);
  b.restoreInsertionPoint(insertionPoint);
  IRMapping mapping;
  mapping.map(block->getArgument(0), a);
  mapping.map(block->getArgument(1), rhs);
  Value result;
  for (Operation &nested : block->getOperations()) {
    if (isa<func::ReturnOp>(nested))
      continue;
    if (auto read = dyn_cast<vector::TransferReadOp>(nested)) {
      if (read.getBase() == block->getArgument(2)) {
        mapping.map(read.getResult(), accumulator);
        continue;
      }
    }
    if (auto write = dyn_cast<vector::TransferWriteOp>(nested)) {
      result = mapping.lookup(write.getVector());
      continue;
    }
    b.clone(nested, mapping);
  }
  fn.erase();
  if (!result)
    return failure();
  return result;
}

struct Schedule {
  bool mma, vectorized = false, prefetch = false, reuse = false, km = false;
  int64_t bm, bn, bk, tm, tn, threads, fragmentN = 16, pad = 0;
  int64_t inputElements() const {
    return km ? bk * (bm + bn + 2 * pad) : (bm + bn) * (bk + pad);
  }
  int64_t outputElements() const { return mma ? bn * (bm + pad) : 0; }
  uint32_t sharedBytes(Type acc) const {
    return reuse ? std::max(inputElements(), outputElements()) * 2
                 : inputElements() * 2 + outputElements() * (acc.isF16() ? 2 : 4);
  }
};

FailureOr<Schedule> selectSchedule(Matmul mm, const MlirWgslMatmulOptions &o) {
  Schedule s{o.use_mma != 0};
  s.bm = s.bn = s.mma ? 16 : 32;
  s.bk = 16;
  s.tm = s.tn = s.mma ? 16 : 4;
  s.threads = s.mma ? 32 : 64;
  if (s.mma && (o.subgroup_size != 32 ||
      !(o.mma_shapes & (mm.accType.isF16() ? 1 : 2))))
    return mm.function.emitError("requested schedule exceeds target capabilities"), failure();
  if (!s.mma && !mm.accType.isF32())
    return mm.function.emitError("register tiling requires f32 accumulation"), failure();
  if (o.optimization_level) {
    s.vectorized = o.storage_alignment % 4 == 0 && mm.m % 4 == 0 && mm.k % 4 == 0;
    if (s.mma) {
      if (mm.n <= 16) {
        s.bm = s.tm = 16;
        s.bn = s.tn = s.fragmentN =
            (o.mma_shapes & (mm.accType.isF16() ? 4 : 8)) ? 8 : 16;
        s.bk = 128;
      } else if (mm.n <= 32) {
        s.bm = s.bn = 32; s.tm = 16; s.tn = 32;
        s.threads = 64; s.bk = 128;
      } else {
        s.bm = s.bn = mm.n <= 128 ? 64 : 128;
        s.tm = s.tn = s.bm / 2;
        s.threads = 128; s.bk = 32;
      }
    } else {
      s.bk = 8;
      s.km = true;
    }
  }
  if (o.optimization_level >= 2 && o.nvidia_native) {
    if (s.mma) {
      s.pad = mm.n <= 16 || mm.n > 128 ? 8 : 0;
      s.reuse = mm.accType.isF16();
      s.prefetch = s.vectorized && mm.n > 16;
      // Projection tiles measured on native NVIDIA.
      if (mm.rhsType.isF32() && mm.m >= 512 && mm.m <= 8192 &&
          mm.n >= 256 && mm.n <= 512 && mm.k >= 2048 && mm.k <= 8192 &&
          s.vectorized) {
        if (mm.m <= 3072) {
          s.bm = s.bn = 64; s.tm = s.tn = 32; s.threads = 128;
        } else {
          s.bm = s.bn = 128; s.tm = 64; s.tn = 32; s.threads = 256;
        }
      }
    } else if (mm.n <= 64) {
      s.bm = 128; s.bn = 16; s.bk = 32; s.threads = 128; s.pad = 4;
    } else if (mm.n <= 128) {
      s.bm = s.bn = 64; s.bk = 16; s.threads = 256;
      s.prefetch = s.vectorized;
    } else {
      s.bm = 128; s.bn = 64; s.tm = 8; s.bk = 16; s.threads = 256; s.pad = 4;
    }
    // Small aligned projections need more workgroups than the large tile gives.
    if (!s.mma && s.vectorized && mm.m <= 512 && mm.n > 128 &&
        mm.n <= 512 && mm.k <= 512) {
      s.bm = s.bn = 32; s.tm = s.tn = 4;
      s.bk = 8; s.threads = 64; s.pad = 0;
    }
  }
  auto exceedsResources = [&] {
    return s.threads > o.max_workgroup_size ||
           s.sharedBytes(mm.accType) > o.max_shared_memory_bytes;
  };
  // Shrink the output tile before dropping the K tile and prefetch.
  while (s.mma && mm.accType.isF32() && exceedsResources() &&
         s.bm > 32 && s.bn > 32) {
    s.bm /= 2; s.bn /= 2;
    s.tm = s.bm / 2; s.tn = s.bn / 2;
    s.threads = 128;
  }
  // Resource fallback changes the schedule, never the requested precision.
  if (s.threads > o.max_workgroup_size || s.sharedBytes(mm.accType) > o.max_shared_memory_bytes) {
    s.prefetch = s.reuse = false; s.pad = 0;
    s.bm = s.bn = 32;
    s.tm = s.tn = s.mma ? 16 : 4;
    s.bk = 16; s.threads = s.mma ? 128 : 64;
    s.fragmentN = 16;
  }
  if (s.mma && (s.threads > o.max_workgroup_size ||
                s.sharedBytes(mm.accType) > o.max_shared_memory_bytes)) {
    s.bm = s.bn = s.tm = s.tn = 16;
    s.threads = 32;
  }
  if (s.threads > o.max_workgroup_size || s.sharedBytes(mm.accType) > o.max_shared_memory_bytes)
    return mm.function.emitError("requested schedule exceeds target capabilities"), failure();
  return s;
}

struct ScheduledBuilder : KernelBuilder {
  Value view(Value base, int64_t offset, int64_t rows, int64_t cols,
             int64_t stride0, int64_t stride1 = 1) {
    auto baseType = cast<MemRefType>(base.getType());
    auto type = MemRefType::get({rows, cols}, baseType.getElementType(),
        StridedLayoutAttr::get(b.getContext(), offset, {stride0, stride1}), baseType.getMemorySpace());
    return memref::ReinterpretCastOp::create(b, loc, type, base,
        b.getIndexAttr(offset),
        SmallVector<OpFoldResult>{b.getIndexAttr(rows), b.getIndexAttr(cols)},
        SmallVector<OpFoldResult>{b.getIndexAttr(stride0), b.getIndexAttr(stride1)});
  }
  Value both(Value x, Value y) { return arith::AndIOp::create(b, loc, x, y); }
  Value zero(Type type) {
    if (auto v = dyn_cast<VectorType>(type))
      return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(v, b.getFloatAttr(v.getElementType(), 0)));
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(type, 0));
  }
};

LogicalResult scheduleGemv(ModuleOp module, Matmul mm,
                           const MlirWgslMatmulOptions &options,
                           MlirWgslMatmulKernel &metadata) {
  int64_t subgroup = options.subgroup_size;
  if (subgroup < 4 || subgroup > 128 || !llvm::isPowerOf2_64(subgroup) ||
      options.max_workgroup_size < subgroup)
    return mm.function.emitError("GEMV requires a supported fixed subgroup size");
  if ((options.merged_inputs && !mm.rhsType.isF16()) ||
      options.batch_count > INT32_MAX / (mm.m * mm.n))
    return mm.function.emitError("unsupported input binding or batch extent");
  int64_t groups = std::min<int64_t>(4, options.max_workgroup_size / subgroup);
  int64_t width = options.optimization_level && options.storage_alignment % 4 == 0 && mm.k % 4 == 0 ? 4 : 1;
  metadata.tile_m = groups;
  metadata.tile_n = mm.n;
  metadata.workgroup_size = groups * subgroup;
  metadata.shared_memory_bytes = 0;
  IRRewriter b(module.getContext());
  Location loc = mm.function.getLoc();
  auto storageType = [&](Type scalar) {
    return MemRefType::get({ShapedType::kDynamic},
                          width == 4 ? Type(VectorType::get({4}, scalar)) : scalar);
  };
  SmallVector<Type> arguments{storageType(b.getF16Type())};
  if (!options.merged_inputs)
    arguments.push_back(storageType(mm.rhsType));
  arguments.push_back(MemRefType::get({ShapedType::kDynamic}, b.getF32Type()));
  arguments.push_back(MemRefType::get({16}, b.getI32Type()));
  unsigned outputArg = options.merged_inputs ? 1 : 2, paramArg = outputArg + 1;
  b.setInsertionPointToEnd(module.getBody());
  auto gpuModule = gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(gpuModule.getBody());
  auto function = gpu::GPUFuncOp::create(b, loc, "main", b.getFunctionType(arguments, {}));
  function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
  function.setKnownBlockSizeAttr(b.getDenseI32ArrayAttr({(int32_t)metadata.workgroup_size, 1, 1}));
  function.setArgAttr(paramArg, "wgsl.address_space", b.getStringAttr("uniform"));
  b.setInsertionPointToStart(&function.getBody().front());
  KernelBuilder kb{b, loc, {}};
  auto param = [&](int64_t i) -> Value {
    Value v = memref::LoadOp::create(b, loc, function.getArgument(paramArg), kb.c(i));
    return arith::IndexCastUIOp::create(b, loc, b.getIndexType(), v);
  };
  Value offA = param(0), offB = param(1), offC = param(2);
  Value strideA = param(6), strideB = param(7);
  Value gx = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value gy = gpu::BlockIdOp::create(b, loc, gpu::Dimension::y);
  Value nx = gpu::GridDimOp::create(b, loc, gpu::Dimension::x);
  Value linear = kb.add(gx, kb.mul(gy, nx));
  int64_t blocks = llvm::divideCeil(mm.m, groups);
  Value batch = kb.div(linear, kb.c(blocks));
  auto valid = scf::IfOp::create(b, loc, kb.lt(batch, kb.c(options.batch_count)), false);
  b.setInsertionPointToStart(valid.thenBlock());
  if (options.batch_count > 1) {
    Value br2 = param(14), br3 = param(15), size2 = kb.mul(param(12), br2);
    Value i2 = kb.rem(batch, size2), i3 = kb.div(batch, size2);
    offA = kb.add(offA, kb.add(kb.mul(kb.div(i2, br2), param(8)), kb.mul(kb.div(i3, br3), param(10))));
    offB = kb.add(offB, kb.add(kb.mul(i2, param(9)), kb.mul(i3, param(11))));
    offC = kb.add(offC, kb.mul(batch, kb.c(mm.m * mm.n)));
  }
  Value sg = gpu::SubgroupIdOp::create(b, loc, IntegerAttr());
  Value lane = gpu::LaneIdOp::create(b, loc, IntegerAttr());
  Value row = kb.add(kb.mul(kb.rem(linear, kb.c(blocks)), kb.c(groups)), sg);
  Type accumulator = width == 4 ? Type(VectorType::get({4}, b.getF32Type())) : b.getF32Type();
  auto zero = [&](Type type) -> Value {
    if (auto vector = dyn_cast<VectorType>(type))
      return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(vector, b.getFloatAttr(vector.getElementType(), 0)));
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(type, 0));
  };
  SmallVector<Value> initial(mm.n, zero(accumulator));
  // Count chunks so the loop's final increment cannot overflow a signed index.
  int64_t chunks = llvm::divideCeil(mm.k, subgroup * width);
  auto loop = scf::ForOp::create(b, loc, kb.c(0), kb.c(chunks), kb.c(1), initial);
  b.setInsertionPointToStart(loop.getBody());
  Value col = kb.add(kb.mul(loop.getInductionVar(), kb.c(subgroup * width)), kb.mul(lane, kb.c(width)));
  auto load = [&](Value source, Value offset, Value index, bool checkRow) -> Value {
    Type type = cast<MemRefType>(source.getType()).getElementType();
    Value condition;
    if (mm.k % (subgroup * width))
      condition = kb.lt(col, kb.c(mm.k));
    if (checkRow && mm.m % groups) {
      Value inRow = kb.lt(row, kb.c(mm.m));
      condition = condition ? arith::AndIOp::create(b, loc, condition, inRow) : inRow;
    }
    auto read = [&]() -> Value {
      return memref::LoadOp::create(b, loc, source, kb.div(kb.add(offset, index), kb.c(width)));
    };
    if (!condition)
      return read();
    auto guard = scf::IfOp::create(b, loc, TypeRange{type}, condition, true);
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(guard.thenBlock());
    scf::YieldOp::create(b, loc, read());
    b.setInsertionPointToStart(guard.elseBlock());
    scf::YieldOp::create(b, loc, zero(type));
    return guard.getResult(0);
  };
  Value weights = load(function.getArgument(0), offA, kb.add(kb.mul(row, strideA), col), true);
  weights = arith::ExtFOp::create(b, loc, accumulator, weights);
  SmallVector<Value> next;
  for (int64_t n = 0; n < mm.n; ++n) {
    Value rhs = load(function.getArgument(options.merged_inputs ? 0 : 1), offB,
                     kb.add(kb.mul(kb.c(n), strideB), col), false);
    if (mm.rhsType.isF16())
      rhs = arith::ExtFOp::create(b, loc, accumulator, rhs);
    Value acc = loop.getRegionIterArgs()[n];
    if (width == 4)
      next.push_back(vector::FMAOp::create(b, loc, weights, rhs, acc));
    else
      next.push_back(arith::AddFOp::create(b, loc, arith::MulFOp::create(b, loc, weights, rhs), acc));
  }
  scf::YieldOp::create(b, loc, next);
  b.setInsertionPointAfter(loop);
  for (int64_t n = 0; n < mm.n; ++n) {
    Value sum = loop.getResult(n);
    if (width == 4)
      sum = vector::ReductionOp::create(b, loc, vector::CombiningKind::ADD, sum);
    sum = gpu::SubgroupReduceOp::create(b, loc, sum, gpu::AllReduceOperation::ADD, false);
    Value leader = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, kb.c(0));
    if (mm.m % groups)
      leader = arith::AndIOp::create(b, loc, leader, kb.lt(row, kb.c(mm.m)));
    auto store = scf::IfOp::create(b, loc, leader, false);
    OpBuilder::InsertionGuard insertion(b);
    b.setInsertionPointToStart(store.thenBlock());
    Value index = kb.add(offC, kb.add(kb.c(n * mm.m), row));
    memref::StoreOp::create(b, loc, sum, function.getArgument(outputArg), index);
  }
  b.setInsertionPointAfter(valid);
  gpu::ReturnOp::create(b, loc);
  mm.function.erase();
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
  return verify(module);
}

LogicalResult schedule(ModuleOp module, Matmul mm,
                       const MlirWgslMatmulOptions &options,
                       MlirWgslMatmulKernel &metadata) {
  if (mm.isGemv())
    return scheduleGemv(module, mm, options, metadata);
  auto selected = selectSchedule(mm, options);
  if (failed(selected))
    return failure();
  Schedule s = *selected;
  if ((options.merged_inputs && !mm.rhsType.isF16()) ||
      options.batch_count > INT32_MAX / (mm.m * mm.n))
    return mm.function.emitError("unsupported input binding or batch extent");
  MLIRContext *ctx = module.getContext();
  IRRewriter b(ctx);
  Location loc = mm.function.getLoc();
  metadata.tile_m = s.bm; metadata.tile_n = s.bn;
  metadata.workgroup_size = s.threads;
  metadata.shared_memory_bytes = s.sharedBytes(mm.accType);
  auto wg = gpu::AddressSpaceAttr::get(ctx, gpu::AddressSpace::Workgroup);
  int64_t abSize = s.reuse ? std::max(s.inputElements(), s.outputElements()) : s.inputElements();
  SmallVector<Type> sharedTypes{MemRefType::get({abSize}, b.getF16Type(), MemRefLayoutAttrInterface{}, wg)};
  if (s.mma && !s.reuse)
    sharedTypes.push_back(MemRefType::get({s.outputElements()}, mm.accType, MemRefLayoutAttrInterface{}, wg));
  Value sharedAllocation, aTile, bTile;
  auto structured = wgsl::buildStructuredSchedule(
      module, mm.operation, {s.bm, s.bn, s.bk,
      [&](IRRewriter &rewriter, linalg::LinalgOp contraction) {
        OpBuilder::InsertionGuard insertion(rewriter);
        rewriter.setInsertionPointToStart(&mm.function.getBody().front());
        sharedAllocation = memref::AllocaOp::create(
            rewriter, loc, cast<MemRefType>(sharedTypes.front()));
        ScheduledBuilder views{{rewriter, loc, Value()}};
        aTile = views.view(sharedAllocation, 0, s.bm, s.bk,
                           s.km ? 1 : s.bk + s.pad,
                           s.km ? s.bm + s.pad : 1);
        int64_t offset = s.km ? s.bk * (s.bm + s.pad)
                              : s.bm * (s.bk + s.pad);
        bTile = views.view(sharedAllocation, offset, s.bn, s.bk,
                           s.km ? 1 : s.bk + s.pad,
                           s.km ? s.bn + s.pad : 1);
        return wgsl::placeSharedInputPads(rewriter, contraction, {aTile, bTile});
      }});
  if (failed(structured))
    return failure();
  // Input views and their copies stay scalar until the final ABI packing step.
  auto storageType = [&](Type scalar) {
    return MemRefType::get({ShapedType::kDynamic}, scalar);
  };
  SmallVector<Type> arguments{storageType(b.getF16Type())};
  if (!options.merged_inputs)
    arguments.push_back(storageType(mm.rhsType));
  arguments.push_back(storageType(b.getF32Type()));
  arguments.push_back(MemRefType::get({16}, b.getI32Type()));
  unsigned outputArg = options.merged_inputs ? 1 : 2, paramArg = outputArg + 1;
  auto kernel = wgsl::createKernelLaunch(
      b, module, loc, arguments, sharedTypes, {(int32_t)s.threads, 1, 1});
  auto function = kernel.host;
  function.setArgAttr(paramArg, "wgsl.address_space", b.getStringAttr("uniform"));
  if (s.vectorized)
    for (unsigned i = 0; i < paramArg; ++i)
      function.setArgAttr(i, "wgsl.pack_width", b.getI64IntegerAttr(4));
  ScheduledBuilder kb{{b, loc, gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x)}};
  auto param = [&](int64_t i) -> Value {
    Value v = memref::LoadOp::create(b, loc, function.getArgument(paramArg), kb.c(i));
    return arith::IndexCastUIOp::create(b, loc, b.getIndexType(), v);
  };
  Value offA = param(0), offB = param(1), offC = param(2);
  Value strideA = param(6), strideB = param(7);
  Value gx = gpu::BlockIdOp::create(b, loc, gpu::Dimension::x);
  Value gy = gpu::BlockIdOp::create(b, loc, gpu::Dimension::y);
  Value nx = gpu::GridDimOp::create(b, loc, gpu::Dimension::x);
  Value linear = kb.add(gx, kb.mul(gy, nx));
  int64_t blocksM = llvm::divideCeil(mm.m, s.bm), blocksN = llvm::divideCeil(mm.n, s.bn);
  Value batch = kb.div(linear, kb.c(blocksM * blocksN));
  auto valid = scf::IfOp::create(b, loc, kb.lt(batch, kb.c(options.batch_count)), false);
  b.setInsertionPointToStart(valid.thenBlock());
  if (options.batch_count > 1) {
    Value br2 = param(14), br3 = param(15), size2 = kb.mul(param(12), br2);
    Value i2 = kb.rem(batch, size2), i3 = kb.div(batch, size2);
    offA = kb.add(offA, kb.add(kb.mul(kb.div(i2, br2), param(8)), kb.mul(kb.div(i3, br3), param(10))));
    offB = kb.add(offB, kb.add(kb.mul(i2, param(9)), kb.mul(i3, param(11))));
    offC = kb.add(offC, kb.mul(batch, kb.c(mm.m * mm.n)));
  }
  // The unsigned remainder bounds the coordinate and keeps it nonnegative.
  linear = kb.rem(linear, kb.c(blocksM * blocksN));
  auto logicalView = [&](unsigned argument, Value storage, Value offset,
                         ArrayRef<int64_t> shape, ArrayRef<OpFoldResult> strides) {
    auto type = cast<MemRefType>(structured->function.getArgument(argument).getType());
    SmallVector<OpFoldResult> sizes;
    for (int64_t size : shape)
      sizes.push_back(b.getIndexAttr(size));
    SmallVector<int64_t> staticStrides;
    for (OpFoldResult stride : strides)
      staticStrides.push_back(getConstantIntValue(stride).value_or(ShapedType::kDynamic));
    auto viewType = MemRefType::get(shape, type.getElementType(),
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, staticStrides));
    Value view = memref::ReinterpretCastOp::create(
        b, loc, viewType, storage, offset, sizes, strides);
    if (viewType != type)
      view = memref::CastOp::create(b, loc, type, view);
    structured->function.getArgument(argument).replaceAllUsesWith(view);
  };
  logicalView(0, function.getArgument(0), offA, {mm.m, mm.k},
              {strideA, b.getIndexAttr(1)});
  logicalView(1, function.getArgument(options.merged_inputs ? 0 : 1), offB,
              {mm.n, mm.k}, {strideB, b.getIndexAttr(1)});
  logicalView(2, function.getArgument(outputArg), offC, {mm.m, mm.n},
              {b.getIndexAttr(1), b.getIndexAttr(mm.m)});
  auto shared = kernel.launch.getWorkgroupAttributionBBArgs();
  sharedAllocation.replaceAllUsesWith(shared[0]);
  b.eraseOp(sharedAllocation.getDefiningOp());
  Block &source = structured->function.getBody().front();
  for (Operation &op : llvm::make_early_inc_range(source.without_terminator()))
    op.moveBefore(b.getInsertionBlock(), b.getInsertionPoint());
  Value baseM = structured->offsetM, baseN = structured->offsetN;
  auto workgroups = structured->workgroups;
  transform::gpu::GpuBlockIdBuilder blockIds(ctx, true);
  blockIds.idBuilder = [&](RewriterBase &rewriter, Location location,
                            ArrayRef<int64_t>, ArrayRef<int64_t>) {
    // Bound both sides so affine folding removes unit grid dims.
    SmallVector<OpFoldResult> basis{rewriter.getIndexAttr(blocksN),
                                    rewriter.getIndexAttr(blocksM)};
    auto indices = affine::delinearizeIndex(rewriter, location, linear, basis,
                                          /*hasOuterBound=*/false);
    if (failed(indices))
      return transform::gpu::IdBuilderResult{
          "could not delinearize workgroup ID", {}, {}};
    return transform::gpu::IdBuilderResult{
        "", {(*indices)[2], (*indices)[1]}, {}};
  };
  SmallVector<int64_t> grid{blocksM * blocksN, 1, 1};
  auto mapped = transform::gpu::mapForallToBlocksImpl(
      b, transform::TransformOpInterface(), workgroups, grid, blockIds);
  if (!mapped.succeeded())
    return mapped.checkAndReport();
  auto loop = structured->reduction;
  moveLoopInvariantCode(cast<LoopLikeOpInterface>(loop.getOperation()));
  auto workgroupMatmul = structured->contraction;
  for (Value operand : workgroupMatmul->getOperands())
    if (!loop.isDefinedOutsideOfLoop(operand)) {
      auto error = workgroupMatmul.emitError("shared contraction view depends on K");
      error.attachNote() << "dependent operand: " << operand;
      if (Operation *definition = operand.getDefiningOp())
        error.attachNote() << "defined by " << *definition;
      return failure();
    }
  // Distribute the contraction and keep its K loop inside each worker.
  workgroupMatmul->moveBefore(loop);
  b.setInsertionPoint(workgroupMatmul);
  Value cTile;
  if (s.mma)
    cTile = kb.view(shared[s.reuse ? 0 : 1], 0, s.bn, s.bm, s.bm + s.pad);
  SmallVector<Attribute> mapping;
  for (gpu::MappingId id : {gpu::MappingId::LinearDim0, gpu::MappingId::LinearDim1})
    mapping.push_back(s.mma ? Attribute(gpu::GPUWarpMappingAttr::get(ctx, id))
                            : Attribute(gpu::GPUThreadMappingAttr::get(ctx, id)));
  scf::SCFTilingOptions tiling;
  tiling.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp)
      .setTileSizes({b.getIndexAttr(s.tm), b.getIndexAttr(s.tn), b.getIndexAttr(0)})
      .setMapping(mapping);
  auto tiled = scf::tileUsingSCF(b, cast<TilingInterface>(workgroupMatmul.getOperation()), tiling);
  if (failed(tiled))
    return module.emitError("Linalg workgroup tiling failed");
  b.eraseOp(workgroupMatmul);
  auto workers = normalizeForallOp(b, cast<scf::ForallOp>(tiled->loops.front().getOperation()));
  if (failed(workers))
    return module.emitError("SCF worker normalization failed");
  auto tiledMatmul = cast<linalg::MatmulTransposeBOp>(tiled->tiledOps.front());
  Value a = tiledMatmul.getInputs()[0], rhs = tiledMatmul.getInputs()[1];
  Value localM = getValueOrCreateConstantIndexOp(b, loc,
      a.getDefiningOp<memref::SubViewOp>().getMixedOffsets()[0]);
  Value localN = getValueOrCreateConstantIndexOp(b, loc,
      rhs.getDefiningOp<memref::SubViewOp>().getMixedOffsets()[0]);
  b.setInsertionPoint(tiledMatmul);
  auto accType = VectorType::get({s.tm, s.tn}, mm.accType);
  Value zero = kb.zero(accType);
  loop->moveBefore(tiledMatmul);
  tiledMatmul->moveBefore(loop.getBody()->getTerminator());
  auto copies = wgsl::distributeSharedMemoryOperations(
      b, kernel.launch, {s.threads, s.vectorized ? 4 : 1});
  if (failed(copies))
    return failure();
  // Dynamic boundary copy loops stay rolled.
  if (llvm::any_of(*copies, [](const auto &group) { return group.hasSerialCopyLoop; }))
    s.prefetch = false;
  b.setInsertionPoint(tiledMatmul);
  kb.barrier();
  bool vectorizationFailed = false;
  auto carried = loop.replaceWithAdditionalYields(
      b, ValueRange{zero}, false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument> accumulators) {
        b.setInsertionPoint(tiledMatmul);
        auto product = vectorizeTile(b, loc, tiledMatmul, a, rhs, accumulators[0]);
        if (failed(product)) {
          vectorizationFailed = true;
          return SmallVector<Value>{accumulators[0]};
        }
        b.eraseOp(tiledMatmul);
        kb.barrier();
        return SmallVector<Value>{*product};
      });
  if (failed(carried) || vectorizationFailed)
    return module.emitError("structured accumulator vectorization failed");
  loop = cast<scf::ForOp>(carried->getOperation());
  // Lower the accumulator init and output conversion to SSA and stores.
  SmallVector<Operation *> outputOps;
  kernel.launch.walk([&](linalg::LinalgOp op) {
    if (!loop->isAncestor(op))
      outputOps.push_back(op);
  });
  for (Operation *op : llvm::reverse(outputOps)) {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    if (linalgOp.getNumDpsInputs() == 1 &&
        getElementTypeOrSelf(linalgOp.getDpsInputs()[0]).isF32() &&
        getElementTypeOrSelf(linalgOp.getDpsInits()[0]).isF16())
      return op->emitError("input conversion was not fused into K");
    b.eraseOp(op);
  }
  b.setInsertionPointAfter(loop);
  Value result = loop.getResults().back();
  if (s.mma) {
    auto transpose = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0}, ctx);
    vector::TransferWriteOp::create(b, loc, result, cTile,
        ValueRange{localN, localM}, AffineMapAttr::get(transpose), Value(), b.getBoolArrayAttr({true, true}));
    kb.barrier();
  }
  int64_t width = s.vectorized ? 4 : 1;
  int64_t outputs = s.mma ? s.bm * s.bn / (s.threads * width) : s.tm * s.tn / width;
  for (int64_t i = 0; i < outputs; ++i) {
    Value row, col;
    if (s.mma) {
      Value index = kb.add(kb.mul(kb.tid, kb.c(width)), kb.c(i * s.threads * width));
      row = kb.rem(index, kb.c(s.bm)); col = kb.div(index, kb.c(s.bm));
    } else {
      row = kb.add(localM, kb.c((i % (s.tm / width)) * width));
      col = kb.add(localN, kb.c(i / (s.tm / width)));
    }
    Value gr = kb.add(baseM, row), gc = kb.add(baseN, col);
    OpBuilder::InsertionGuard guard(b);
    if (mm.m % s.bm != 0 || mm.n % s.bn != 0) {
      auto store = scf::IfOp::create(b, loc,
          kb.both(kb.lt(gr, kb.c(mm.m)), kb.lt(gc, kb.c(mm.n))), false);
      b.setInsertionPointToStart(store.thenBlock());
    }
    Value value;
    if (s.mma) {
      if (s.vectorized)
        value = vector::LoadOp::create(b, loc, VectorType::get({4}, mm.accType), cTile, ValueRange{col, row});
      else
        value = memref::LoadOp::create(b, loc, cTile, ValueRange{col, row});
      if (mm.accType.isF16())
        value = arith::ExtFOp::create(b, loc, s.vectorized ? Type(VectorType::get({4}, b.getF32Type())) : b.getF32Type(), value);
    } else {
      SmallVector<Value> elements;
      for (int64_t j = 0; j < width; ++j)
        elements.push_back(vector::ExtractOp::create(b, loc, result,
            ArrayRef<int64_t>{(i % (s.tm / width)) * width + j, i / (s.tm / width)}));
      value = s.vectorized ? vector::FromElementsOp::create(b, loc, VectorType::get({4}, b.getF32Type()), elements).getResult() : elements[0];
    }
    Value index = kb.add(offC, kb.add(kb.mul(gc, kb.c(mm.m)), gr));
    if (s.vectorized)
      vector::StoreOp::create(b, loc, value, function.getArgument(outputArg), index);
    else
      memref::StoreOp::create(b, loc, value, function.getArgument(outputArg), index);
  }
  b.setInsertionPointAfter(valid);
  gpu::TerminatorOp::create(b, loc);
  // Schedule copies after view cleanup to expose loads for pipelining.
  if (s.prefetch)
    loop->setAttr("wgsl.prefetch", b.getUnitAttr());
  mm.function.erase();
  auto outlined = wgsl::outlineKernelLaunch(module, kernel);
  if (failed(outlined))
    return failure();
  (*outlined)->setAttr("wgsl.mma_shape", b.getDenseI64ArrayAttr({16, s.fragmentN, 16}));
  return verify(module);
}

// Stage 0 loads the next tile. Stage 1 writes shared memory and computes.
LogicalResult pipelineSharedCopies(ModuleOp module) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) {
    if (loop->hasAttr("wgsl.prefetch"))
      loops.push_back(loop);
  });
  IRRewriter b(module.getContext());
  for (scf::ForOp loop : loops) {
    loop->removeAttr("wgsl.prefetch");
    SmallVector<Operation *> loads, stores, compute;
    bool computing = false, mixedAccess = false;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (isa<gpu::BarrierOp>(op)) {
        if (!computing) {
          stores.push_back(&op);
          computing = true;
        } else {
          compute.push_back(&op);
        }
        continue;
      }
      if (computing) {
        compute.push_back(&op);
        continue;
      }
      bool writes = false, reads = false;
      op.walk([&](Operation *nested) {
        writes |= isa<memref::StoreOp, vector::StoreOp,
                       vector::TransferWriteOp>(nested);
        reads |= isa<memref::LoadOp, vector::LoadOp,
                      vector::TransferReadOp>(nested);
      });
      mixedAccess |= writes && reads;
      (writes ? stores : loads).push_back(&op);
    }
    if (mixedAccess || !computing)
      continue;
    scf::PipeliningOption options;
    options.getScheduleFn = [&](scf::ForOp, auto &schedule) {
      for (Operation *op : stores)
        schedule.emplace_back(op, 1);
      for (Operation *op : loads)
        schedule.emplace_back(op, 0);
      for (Operation *op : compute)
        schedule.emplace_back(op, 1);
    };
    b.setInsertionPoint(loop);
    if (failed(scf::pipelineForLoop(b, loop, options)))
      return loop.emitError("structured copy pipelining failed");
  }
  return success();
}

LogicalResult lower(ModuleOp module, bool mma) {
  MLIRContext *ctx = module.getContext();
  IRRewriter rewriter(ctx);
  SmallVector<scf::ForallOp> workers;
  module.walk([&](scf::ForallOp op) { workers.push_back(op); });
  for (scf::ForallOp op : workers) {
    auto kernel = op->getParentOfType<gpu::GPUFuncOp>();
    int64_t threads = kernel.getKnownBlockSize().value()[0];
    transform::gpu::GpuWarpIdBuilder subgroupIds(ctx, 32, true);
    // WGSL does not guarantee subgroup_id == local_invocation_index / 32.
    subgroupIds.idBuilder = [](RewriterBase &b, Location loc,
                               ArrayRef<int64_t> sizes, ArrayRef<int64_t> block) {
      if (llvm::product_of(sizes) * 32 != llvm::product_of(block))
        return transform::gpu::IdBuilderResult{"subgroup tile count must cover the workgroup", {}, {}};
      Value id = gpu::SubgroupIdOp::create(b, loc, IntegerAttr());
      SmallVector<OpFoldResult> basis;
      for (int64_t size : llvm::reverse(sizes))
        basis.push_back(b.getIndexAttr(size));
      auto indices = affine::delinearizeIndex(b, loc, id, basis);
      if (failed(indices))
        return transform::gpu::IdBuilderResult{"could not delinearize subgroup ID", {}, {}};
      return transform::gpu::IdBuilderResult{"", llvm::to_vector(llvm::reverse(*indices)), {}};
    };
    auto mapped = transform::gpu::mapOneForallToThreadsImpl(
        rewriter, std::nullopt, op, {threads, 1, 1}, 32,
        /*syncAfterDistribute=*/false, mma ? &subgroupIds : nullptr);
    if (!mapped.succeeded()) {
      return mapped.checkAndReport();
    }
  }
  if (failed(wgsl::lowerPackedStorageBindings(module)))
    return failure();
  PassManager pre(ctx);
  pre.addPass(memref::createFoldMemRefAliasOpsPass());
  pre.addPass(createLowerAffinePass());
  pre.addPass(createCanonicalizerPass());
  pre.addPass(createCSEPass());
  // Hoist copy subviews so pipeline stages carry values, not descriptors.
  pre.addPass(createLoopInvariantCodeMotionPass());
  if (failed(pre.run(module)) || failed(pipelineSharedCopies(module)))
    return failure();
  if (mma) {
    int64_t fragmentN = 16;
    module.walk([&](gpu::GPUFuncOp f) {
      fragmentN = f->getAttrOfType<DenseI64ArrayAttr>("wgsl.mma_shape")[1];
    });
    RewritePatternSet patterns(ctx);
    // Same patterns as apply_patterns.gpu.unroll_vectors_subgroup_mma.
    gpu::populateSubgroupMmaUnrollPatterns(patterns, 16, fragmentN, 16);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return failure();
    PassManager pm(ctx);
    // Canonicalize separately: insert-chain folding can undo vector unrolling.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    vector::VectorUnrollLoopCarriedOptions tiles;
    tiles.targetShape = {16, fragmentN};
    pm.addPass(vector::createVectorUnrollLoopCarried(tiles));
    pm.addPass(createConvertVectorToGPUPass());
    if (failed(pm.run(module)))
      return failure();
  } else {
    RewritePatternSet patterns(ctx);
    vector::populateVectorContractLoweringPatterns(patterns, vector::VectorContractLowering::OuterProduct);
    vector::populateVectorTransposeLoweringPatterns(patterns, vector::VectorTransposeLowering::EltWise);
    vector::populateVectorUnrollPatterns(patterns,
        vector::UnrollVectorOptions().setNativeShape({1, 4})
            .setFilterConstraint([](Operation *op) { return success(isa<vector::TransferReadOp>(op)); }));
    populateVectorToSCFConversionPatterns(patterns, VectorTransferToSCFOptions().enableFullUnroll());
    vector::populateVectorTransferLoweringPatterns(patterns, 1);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return failure();
    PassManager pm(ctx);
    vector::VectorUnrollLoopCarriedOptions opts;
    opts.targetShape = {1, 4};
    pm.addPass(vector::createVectorUnrollLoopCarried(opts));
    if (failed(pm.run(module)))
      return failure();
    RewritePatternSet shortVectors(ctx);
    vector::populateVectorUnrollPatterns(shortVectors,
        vector::UnrollVectorOptions().setNativeShapeFn(
            [](Operation *op) -> std::optional<SmallVector<int64_t>> {
              if (op->getNumResults() != 1)
                return std::nullopt;
              auto type = dyn_cast<VectorType>(op->getResult(0).getType());
              return type && type.getRank() == 1 && type.getDimSize(0) > 4
                         ? std::optional(SmallVector<int64_t>{4}) : std::nullopt;
            }));
    vector::ExtractOp::getCanonicalizationPatterns(shortVectors, ctx);
    vector::ExtractStridedSliceOp::getCanonicalizationPatterns(shortVectors, ctx);
    vector::populateVectorExtractStridedSliceToExtractInsertChainPatterns(shortVectors);
    if (failed(applyPatternsGreedily(module, std::move(shortVectors))))
      return failure();
  }
  PassManager clean(ctx);
  clean.addPass(createLowerAffinePass());
  clean.addPass(arith::createArithUnsignedWhenEquivalentPass());
  // Fold subgroup ID sign corrections. MMA only, as it slows register tiles.
  if (mma && !workers.empty())
    clean.addPass(arith::createArithIntRangeOpts());
  clean.addPass(createCanonicalizerPass());
  clean.addPass(createCSEPass());
  clean.addPass(createLoopInvariantCodeMotionPass());
  return clean.run(module);
}

char *copyString(StringRef text) {
  char *copy = static_cast<char *>(std::malloc(text.size() + 1));
  if (copy) {
    std::memcpy(copy, text.data(), text.size());
    copy[text.size()] = 0;
  }
  return copy;
}
char *printIR(ModuleOp module) {
  std::string str;
  llvm::raw_string_ostream out(str);
  module.print(out);
  return copyString(str);
}
} // namespace

extern "C" int mlirWgslCompileMatmul(const char *source,
    const MlirWgslMatmulOptions *options, MlirWgslMatmulKernel *result) {
  if (!result)
    return 1;
  *result = {};
  if (!source || !options || options->abi_version != 2 || options->use_mma > 1 ||
      options->optimization_level > 2 || options->storage_alignment == 0 ||
      options->batch_count == 0 || options->merged_inputs > 1) {
    result->error = copyString("invalid compiler request or ABI version\n");
    return 1;
  }
  DialectRegistry registry;
  wgsl::registerStructuredScheduleInterfaces(registry);
  MLIRContext ctx(registry);
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                  linalg::LinalgDialect, memref::MemRefDialect, scf::SCFDialect,
                  tensor::TensorDialect, vector::VectorDialect>();
  std::string errors;
  llvm::raw_string_ostream errorStream(errors);
  ScopedDiagnosticHandler diagnostics(&ctx, [&](Diagnostic &diag) {
    diag.print(errorStream);
    errorStream << '\n';
    return success();
  });
  auto module = parseSourceString<ModuleOp>(source, &ctx);
  if (module) {
    auto mm = matchMatmul(*module);
    if (succeeded(mm) && succeeded(schedule(*module, *mm, *options, *result))) {
      result->scheduled_mlir = printIR(*module);
      if (succeeded(lower(*module, options->use_mma && !mm->isGemv()))) {
        result->lowered_mlir = printIR(*module);
        std::string wgsl;
        llvm::raw_string_ostream out(wgsl);
        if (succeeded(wgsl::translateToWGSL(*module, out))) {
          result->wgsl = copyString(wgsl);
          if (result->wgsl && result->scheduled_mlir && result->lowered_mlir)
            return 0;
        }
      }
    }
  }
  result->error = copyString(errors.empty() ? "compilation failed\n" : errors);
  return 1;
}
extern "C" void mlirWgslFreeMatmulKernel(MlirWgslMatmulKernel *result) {
  if (!result)
    return;
  std::free(result->wgsl);
  std::free(result->scheduled_mlir);
  std::free(result->lowered_mlir);
  std::free(result->error);
  *result = {};
}
