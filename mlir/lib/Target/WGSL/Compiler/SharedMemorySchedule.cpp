//===- SharedMemorySchedule.cpp - Shared-memory staging -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SharedMemorySchedule.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using mlir::wgsl::CooperativeCopyGroup;

namespace {

bool isWorkgroupBuffer(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  auto space = type ? dyn_cast_or_null<gpu::AddressSpaceAttr>(
                          type.getMemorySpace())
                    : gpu::AddressSpaceAttr();
  return space && space.getValue() == gpu::AddressSpace::Workgroup;
}

// Retarget the pad's fill and copy to the shared-memory layout.
LogicalResult retargetAllocation(IRRewriter &rewriter, Value oldBuffer,
                                Value sharedBuffer) {
  llvm::SetVector<Operation *> users;
  for (Operation *user : oldBuffer.getUsers())
    users.insert(user);
  for (Operation *user : users) {
    if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      rewriter.setInsertionPoint(view);
      auto type = memref::SubViewOp::inferRankReducedResultType(
          view.getType().getShape(), cast<MemRefType>(sharedBuffer.getType()),
          view.getMixedOffsets(), view.getMixedSizes(), view.getMixedStrides());
      auto replacement = memref::SubViewOp::create(
          rewriter, view.getLoc(), cast<MemRefType>(type), sharedBuffer,
          view.getMixedOffsets(), view.getMixedSizes(), view.getMixedStrides());
      if (failed(retargetAllocation(rewriter, view, replacement)))
        return failure();
      rewriter.eraseOp(view);
      continue;
    }
    // These users accept a different layout or memory space.
    if (!isa<linalg::LinalgOp, bufferization::ToTensorOp,
             bufferization::MaterializeInDestinationOp, memref::CopyOp,
             memref::DimOp>(user))
      return user->emitError("unexpected use of a padded workgroup allocation");
    rewriter.modifyOpInPlace(user, [&] {
      for (OpOperand &operand : user->getOpOperands())
        if (operand.get() == oldBuffer)
          operand.set(sharedBuffer);
    });
  }
  return success();
}

// Use insert_slice so empty-tensor elimination can forward the destination.
LogicalResult forwardPointwiseDestination(
    IRRewriter &rewriter, linalg::GenericOp producer,
    bufferization::ToTensorOp sharedTensor) {
  if (!producer || !producer.hasPureTensorSemantics() ||
      producer.getNumDpsInits() != 1 ||
      producer.getNumParallelLoops() != producer.getNumLoops() ||
      producer.payloadUsesValueFromOperand(producer.getDpsInitOperand(0)) ||
      !llvm::all_of(producer.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }))
    return success();

  Value result = producer->getResult(0);
  bufferization::MaterializeInDestinationOp materialize;
  SmallVector<tensor::DimOp> dimensions;
  for (Operation *user : result.getUsers()) {
    if (auto dim = dyn_cast<tensor::DimOp>(user)) {
      dimensions.push_back(dim);
      continue;
    }
    auto candidate = dyn_cast<bufferization::MaterializeInDestinationOp>(user);
    if (!candidate || candidate.getSource() != result || materialize)
      return producer.emitError(
          "shared pointwise input must have a single materialization");
    materialize = candidate;
  }
  auto subview = materialize
                     ? materialize.getDest().getDefiningOp<memref::SubViewOp>()
                     : memref::SubViewOp();
  if (!subview || subview.getSource() != sharedTensor.getBuffer() ||
      producer->getBlock() != sharedTensor->getBlock())
    return producer.emitError("unexpected shared pointwise destination view");

  // Resolve dims through the destination before moving the producer.
  for (tensor::DimOp dim : dimensions)
    rewriter.modifyOpInPlace(dim, [&] {
      dim.getSourceMutable().set(producer.getDpsInits()[0]);
    });
  producer->moveAfter(sharedTensor);
  rewriter.setInsertionPointAfter(producer);
  auto update = tensor::InsertSliceOp::create(
      rewriter, producer.getLoc(), result, sharedTensor.getResult(),
      subview.getMixedOffsets(), subview.getMixedSizes(),
      subview.getMixedStrides());
  rewriter.replaceUsesWithIf(
      sharedTensor.getResult(), update.getResult(),
      [&](OpOperand &use) { return use.getOwner() != update.getOperation(); });
  rewriter.eraseOp(materialize);
  return success();
}

// Use the full K extent so fill and copy assign the same threads.
FailureOr<MemRefType> getFullTileType(Value value) {
  MemRefType fullType;
  while (value) {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || type.getRank() != 2 || !isWorkgroupBuffer(value))
      break;
    if (type.hasStaticShape())
      fullType = type;
    if (auto subview = value.getDefiningOp<memref::SubViewOp>())
      value = subview.getSource();
    else if (auto cast = value.getDefiningOp<memref::CastOp>())
      value = cast.getSource();
    else
      break;
  }
  if (!fullType)
    return failure();
  return fullType;
}

// Only visit ops between the two sentinels, never the K loop.
// Fold tile sizes before emulating masks to keep vec4 loads.
LogicalResult lowerCopyVectors(Block *parent, Operation *previous,
                               Operation *end) {
  MLIRContext *context = end->getContext();
  auto collect = [&]() {
    SmallVector<Operation *> operations;
    auto begin = previous ? std::next(previous->getIterator()) : parent->begin();
    for (Operation &root : llvm::make_range(begin, end->getIterator()))
      root.walk([&](Operation *op) { operations.push_back(op); });
    return operations;
  };
  auto canonicalizations = [&](RewritePatternSet &patterns) {
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, context);
  };
  auto populate = [&](RewritePatternSet &patterns) {
    canonicalizations(patterns);
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);
    vector::populateVectorBroadcastLoweringPatterns(patterns);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    vector::populateVectorMaskMaterializationPatterns(
        patterns, /*force32BitVectorIndices=*/true);
    populateVectorToSCFConversionPatterns(
        patterns, VectorTransferToSCFOptions().enableFullUnroll());
    vector::populateVectorTransferLoweringPatterns(patterns, 1);
  };
  auto apply = [&](RewritePatternSet &&patterns) {
    SmallVector<Operation *> operations = collect();
    // Limit the worklist to this copy group, not the reduction loop.
    GreedyRewriteConfig config;
    config.setScope(parent->getParent())
        .setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    return applyOpPatternsGreedily(operations, std::move(patterns), config);
  };
  RewritePatternSet canonical(context);
  canonicalizations(canonical);
  if (failed(apply(std::move(canonical))))
    return failure();
  RewritePatternSet patterns(context);
  populate(patterns);
  if (failed(apply(std::move(patterns))))
    return failure();

  // Unroll the scalar copy loops from VectorToSCF, never the K loop.
  SmallVector<scf::ForOp> scalarLoops;
  for (Operation *operation : collect())
    if (auto loop = dyn_cast<scf::ForOp>(operation))
      scalarLoops.push_back(loop);
  // Post-order walk: inner loops first.
  for (scf::ForOp loop : scalarLoops)
    if (getConstantIntValue(loop.getLowerBound()) &&
        getConstantIntValue(loop.getUpperBound()) &&
        getConstantIntValue(loop.getStep()))
      if (failed(loopUnrollFull(loop)))
        return failure();
  RewritePatternSet cleanup(context);
  populate(cleanup);
  if (failed(apply(std::move(cleanup))))
    return failure();
  // Emulate only the masks left on boundary accesses.
  RewritePatternSet masks(context);
  canonicalizations(masks);
  vector::populateVectorMaskedLoadStoreEmulationPatterns(masks);
  if (failed(apply(std::move(masks))))
    return failure();
  // Keep ExtractToShapeCast out of this set, it undoes shape-cast lowering.
  RewritePatternSet shapeCasts(context);
  vector::populateVectorShapeCastLoweringPatterns(shapeCasts);
  vector::InsertOp::getCanonicalizationPatterns(shapeCasts, context);
  return apply(std::move(shapeCasts));
}

FailureOr<CooperativeCopyGroup>
distributeOne(IRRewriter &rewriter, linalg::LinalgOp operation,
              const wgsl::CooperativeCopyOptions &options) {
  Value destination = operation.getDpsInits()[0];
  auto fullTile = getFullTileType(destination);
  if (failed(fullTile))
    return operation.emitError("cooperative copy requires a static shared tile"),
           failure();
  int64_t columns = fullTile->getDimSize(1);
  if (columns % options.vectorWidth)
    return operation.emitError("shared tile is not divisible by copy width"),
           failure();
  int64_t columnThreads = std::min(columns / options.vectorWidth,
                                   options.threads);
  if (options.threads % columnThreads)
    return operation.emitError("copy thread grid does not cover the workgroup"),
           failure();
  int64_t rowThreads = options.threads / columnThreads;

  Operation *previous = operation->getPrevNode();
  Operation *next = operation->getNextNode();
  Block *parent = operation->getBlock();
  // A serial copy loop handles tiles larger than one vector per thread.
  scf::SCFTilingOptions serial;
  serial.setTileSizes({rewriter.getIndexAttr(rowThreads),
                       rewriter.getIndexAttr(columnThreads * options.vectorWidth)});
  rewriter.setInsertionPoint(operation);
  auto chunks = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(operation.getOperation()), serial);
  if (failed(chunks))
    return failure();
  auto chunk = cast<linalg::LinalgOp>(chunks->tiledOps.front());
  scf::SCFTilingOptions parallel;
  parallel.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp)
      .setNumThreads({rewriter.getIndexAttr(rowThreads),
                      rewriter.getIndexAttr(columnThreads)})
      .setTileSizes({rewriter.getIndexAttr(1),
                     rewriter.getIndexAttr(options.vectorWidth)})
      .setMapping({gpu::GPUThreadMappingAttr::get(
                       rewriter.getContext(), gpu::MappingId::LinearDim1),
                   gpu::GPUThreadMappingAttr::get(
                       rewriter.getContext(), gpu::MappingId::LinearDim0)});
  rewriter.setInsertionPoint(chunk);
  auto workers = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(chunk.getOperation()), parallel);
  if (failed(workers))
    return failure();
  rewriter.eraseOp(chunk);
  auto tiled = cast<linalg::LinalgOp>(workers->tiledOps.front());
  rewriter.setInsertionPoint(tiled);
  if (options.vectorWidth == 1) {
    // WGSL has no vector<1x1>, so scalar schedules lower to scalar loops.
    if (failed(linalg::linalgOpToLoops(rewriter, tiled)))
      return tiled.emitError("cooperative scalar copy lowering failed"),
             failure();
  } else if (failed(linalg::vectorize(rewriter, tiled, {1, options.vectorWidth},
                                     {false, false}))) {
    return tiled.emitError("cooperative copy vectorization failed"), failure();
  }
  rewriter.eraseOp(tiled);
  auto forall = cast<scf::ForallOp>(workers->loops.front().getOperation());
  rewriter.setInsertionPoint(forall);
  auto mapped = transform::gpu::mapOneForallToThreadsImpl(
      rewriter, std::nullopt, forall, {options.threads, 1, 1}, 32,
      /*syncAfterDistribute=*/false);
  if (!mapped.succeeded()) {
    (void)mapped.checkAndReport();
    return failure();
  }

  // Unroll only static copy loops. Dynamic ones stay rolled.
  for (LoopLikeOpInterface loop : llvm::reverse(chunks->loops)) {
    auto serialLoop = cast<scf::ForOp>(loop.getOperation());
    if (getConstantIntValue(serialLoop.getLowerBound()) &&
        getConstantIntValue(serialLoop.getUpperBound()) &&
        getConstantIntValue(serialLoop.getStep()))
      if (failed(loopUnrollFull(serialLoop)))
        return failure();
  }
  // Move the original after the new loop nest so the sentinels enclose it.
  operation->moveBefore(parent, next ? next->getIterator() : parent->end());
  if (failed(lowerCopyVectors(parent, previous, operation)))
    return failure();
  CooperativeCopyGroup group;
  auto begin = previous ? std::next(previous->getIterator()) : parent->begin();
  for (Operation &root : llvm::make_range(begin, operation->getIterator())) {
    group.roots.push_back(&root);
    root.walk([&](scf::ForOp) { group.hasSerialCopyLoop = true; });
  }
  rewriter.eraseOp(operation);
  return group;
}

} // namespace

LogicalResult mlir::wgsl::placeSharedInputPads(
    IRRewriter &rewriter, linalg::LinalgOp paddedContraction,
    ArrayRef<Value> sharedViews) {
  if (sharedViews.size() != paddedContraction.getNumDpsInputs())
    return paddedContraction.emitError("expected a shared view for each input");
  OpBuilder::InsertionGuard guard(rewriter);
  for (auto [index, shared] : llvm::enumerate(sharedViews)) {
    Value operand = paddedContraction.getDpsInputs()[index];
    auto pad = operand.getDefiningOp<tensor::PadOp>();
    auto sharedType = dyn_cast<MemRefType>(shared.getType());
    if (!pad || !pad.hasZeroLowPad() || !sharedType ||
        !isWorkgroupBuffer(shared) ||
        sharedType.getShape() != pad.getResultType().getShape() ||
        sharedType.getElementType() != pad.getResultType().getElementType())
      return paddedContraction.emitError(
          "input padding must have zero low padding and match the shared view");
    auto producer = pad.getSource().getDefiningOp<linalg::GenericOp>();
    linalg::BufferizeToAllocationOptions options;
    options.allocOp = linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloca;
    options.memcpyOp = linalg::BufferizeToAllocationOptions::MemcpyOp::
        MaterializeInDestination;
    Value allocated = linalg::bufferizeToAllocation(
        rewriter, options, pad, sharedType.getMemorySpace());
    if (!allocated || failed(retargetAllocation(rewriter, allocated, shared)))
      return failure();
    rewriter.eraseOp(allocated.getDefiningOp());
    auto sharedTensor = paddedContraction.getDpsInputs()[index]
                            .getDefiningOp<bufferization::ToTensorOp>();
    if (!sharedTensor ||
        failed(forwardPointwiseDestination(rewriter, producer, sharedTensor)))
      return failure();
    rewriter.setInsertionPoint(paddedContraction);
  }
  // Forward the destination before canonicalization folds the insert_slice.
  return bufferization::eliminateEmptyTensors(
      rewriter, paddedContraction->getParentOfType<func::FuncOp>());
}

LogicalResult mlir::wgsl::createStructuredBufferCopy(OpBuilder &builder,
                                                   Location loc, Value source,
                                                   Value destination) {
  linalg::CopyOp::create(builder, loc, source, destination);
  return success();
}

FailureOr<SmallVector<mlir::wgsl::CooperativeCopyGroup>>
mlir::wgsl::distributeSharedMemoryOperations(
    IRRewriter &rewriter, Operation *scope,
    const CooperativeCopyOptions &options) {
  if (options.threads <= 0 ||
      (options.vectorWidth != 1 && options.vectorWidth != 4))
    return scope->emitError("invalid cooperative copy configuration"), failure();
  SmallVector<linalg::LinalgOp> operations;
  scope->walk([&](linalg::LinalgOp op) {
    if (op.hasPureBufferSemantics() && op.getNumDpsInits() == 1 &&
        op.getNumLoops() == 2 && op.getNumParallelLoops() == 2 &&
        isWorkgroupBuffer(op.getDpsInits()[0]))
      operations.push_back(op);
  });
  SmallVector<CooperativeCopyGroup> groups;
  for (linalg::LinalgOp op : operations) {
    auto result = distributeOne(rewriter, op, options);
    if (failed(result))
      return failure();
    groups.push_back(std::move(*result));
  }
  return groups;
}
