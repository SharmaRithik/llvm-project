//===- StructuredSchedule.cpp - Structured matmul schedule ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StructuredSchedule.h"
#include "SharedMemorySchedule.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
constexpr StringLiteral workgroupMarker = "wgsl.structured.workgroups";
constexpr StringLiteral reductionMarker = "wgsl.structured.reduction";
constexpr StringLiteral contractionMarker = "wgsl.structured.contraction";

// Same slice cleanup as transform.structured.fuse.
void addSliceCleanup(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  tensor::populateBubbleUpExtractSliceOpPatterns(patterns);
  // Localize tiled empty destinations so casts bufferize into shared memory.
  tensor::populateFoldTensorEmptyPatterns(patterns);
}

LogicalResult cleanupSlices(Operation *scope) {
  RewritePatternSet patterns(scope->getContext());
  addSliceCleanup(patterns);
  return applyPatternsGreedily(scope, std::move(patterns));
}

bool isRhsConversion(Operation *op) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 ||
      generic.getNumParallelLoops() != generic.getNumLoops())
    return false;
  return getElementTypeOrSelf(generic.getInputs()[0]).isF32() &&
         getElementTypeOrSelf(generic.getOutputs()[0]).isF16();
}

FailureOr<linalg::MatmulTransposeBOp>
padContraction(IRRewriter &rewriter, linalg::MatmulTransposeBOp contraction,
               ArrayRef<int64_t> dimensions, ArrayRef<int64_t> multiples,
               bool nofoldInputs) {
  linalg::LinalgPaddingOptions padding;
  padding.setPaddingDimensions(dimensions)
      .setPadToMultipleOf(multiples)
      .setNofoldFlags({nofoldInputs, nofoldInputs, false})
      .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);
  // rewriteAsPaddedOp infers a zero of each operand's actual element type.
  linalg::LinalgOp padded;
  SmallVector<Value> replacements;
  SmallVector<tensor::PadOp> pads;
  if (failed(linalg::rewriteAsPaddedOp(rewriter, cast<linalg::LinalgOp>(contraction.getOperation()), padding, padded,
                                      replacements, pads)))
    return failure();
  rewriter.replaceOp(contraction, replacements);
  return cast<linalg::MatmulTransposeBOp>(padded.getOperation());
}

// Only the output is restrict. A and B may share one llama.cpp binding.
LogicalResult addOutputDestination(IRRewriter &rewriter, func::FuncOp function) {
  auto ret = dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 1 || function.getNumArguments() != 2)
    return function.emitError("structured schedule expects two inputs and one result");
  auto result = dyn_cast<RankedTensorType>(ret.getOperand(0).getType());
  if (!result || result.getRank() != 2 || !result.hasStaticShape())
    return function.emitError("structured schedule expects a static rank-2 result");
  auto layout = StridedLayoutAttr::get(
      function.getContext(), ShapedType::kDynamic,
      {ShapedType::kDynamic, ShapedType::kDynamic});
  auto destinationType = MemRefType::get(result.getShape(),
                                        result.getElementType(), layout);
  if (failed(function.insertArgument(2, destinationType,
                                     rewriter.getDictionaryAttr({}),
                                     function.getLoc())))
    return failure();
  function.setArgAttr(0, "wgsl.logical_binding", rewriter.getStringAttr("A"));
  function.setArgAttr(1, "wgsl.logical_binding", rewriter.getStringAttr("B"));
  function.setArgAttr(2, "wgsl.logical_binding", rewriter.getStringAttr("C"));
  function.setArgAttr(0, "bufferization.writable", rewriter.getBoolAttr(false));
  function.setArgAttr(1, "bufferization.writable", rewriter.getBoolAttr(false));
  rewriter.setInsertionPoint(ret);
  bufferization::MaterializeInDestinationOp::create(
      rewriter, ret.getLoc(), TypeRange{}, ret.getOperand(0),
      function.getArgument(2), /*restrict=*/true, /*writable=*/true);
  ret.getOperandsMutable().clear();
  function.setType(rewriter.getFunctionType(function.getArgumentTypes(), {}));
  return success();
}

} // namespace

void mlir::wgsl::registerStructuredScheduleInterfaces(DialectRegistry &registry) {
  registry.insert<bufferization::BufferizationDialect>();
  linalg::registerTilingInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
}

FailureOr<mlir::wgsl::StructuredSchedule>
mlir::wgsl::buildStructuredSchedule(ModuleOp module,
                                    linalg::MatmulTransposeBOp matmul,
                                    const StructuredScheduleOptions &options) {
  if (options.tileM <= 0 || options.tileN <= 0 || options.tileK <= 0)
    return module.emitError("structured tile sizes must be positive"), failure();
  func::FuncOp function = matmul->getParentOfType<func::FuncOp>();
  if (!function || !function.getBody().hasOneBlock())
    return module.emitError("expected a single-block matmul function"), failure();
  IRRewriter rewriter(module.getContext());
  module.getContext()->getOrLoadDialect<bufferization::BufferizationDialect>();
  auto ret = dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto root = ret && ret.getNumOperands() == 1
                  ? dyn_cast_or_null<linalg::LinalgOp>(ret.getOperand(0).getDefiningOp())
                  : linalg::LinalgOp();
  if (!root)
    return function.emitError("expected a Linalg result producer"), failure();

  // Fuse the output conversion to avoid an M*N f16 temporary.
  if (failed(addOutputDestination(rewriter, function)))
    return failure();
  scf::SCFTileAndFuseOptions outerOptions;
  SmallVector<OpFoldResult> outerSizes(root.getNumLoops(), rewriter.getIndexAttr(0));
  outerSizes[0] = rewriter.getIndexAttr(options.tileM);
  outerSizes[1] = rewriter.getIndexAttr(options.tileN);
  outerOptions.tilingOptions
      .setLoopType(scf::SCFTilingOptions::LoopType::ForallOp)
      .setTileSizes(outerSizes)
      .setMapping({gpu::GPUBlockMappingAttr::get(module.getContext(), gpu::MappingId::LinearDim0),
                   gpu::GPUBlockMappingAttr::get(module.getContext(), gpu::MappingId::LinearDim1)});
  outerOptions.setFusionControlFn(
      [](tensor::ExtractSliceOp, OpResult producer, bool)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        // Keep the RHS cast behind a slice so row and K pads combine.
        if (isa<linalg::LinalgOp>(producer.getOwner()) &&
            !isRhsConversion(producer.getOwner()))
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        return std::nullopt;
      });
  RewritePatternSet outerCleanup(module.getContext());
  addSliceCleanup(outerCleanup);
  outerOptions.cleanupPatterns = std::move(outerCleanup);
  rewriter.setInsertionPoint(root);
  auto outer = scf::tileConsumerAndFuseProducersUsingSCF(
      rewriter, cast<TilingInterface>(root.getOperation()), outerOptions);
  if (failed(outer) || outer->loops.size() != 1)
    return function.emitError("whole-graph workgroup tiling failed"), failure();
  rewriter.replaceOp(root, outer->replacements.lookup(root->getResult(0)));
  auto normalized = normalizeForallOp(
      rewriter, cast<scf::ForallOp>(outer->loops.front().getOperation()));
  if (failed(normalized))
    return function.emitError("workgroup normalization failed"), failure();
  scf::ForallOp workgroups = *normalized;
  workgroups->setAttr(workgroupMarker, rewriter.getUnitAttr());
  linalg::MatmulTransposeBOp workgroupContraction;
  workgroups.walk([&](linalg::MatmulTransposeBOp op) { workgroupContraction = op; });
  if (!workgroupContraction)
    return function.emitError("tiling did not retain the matmul"), failure();

  // Pad M and N before K tiling to keep the accumulator out of the K loop.
  auto paddedWorkgroup = padContraction(rewriter, workgroupContraction, {0, 1},
                                        {options.tileM, options.tileN}, false);
  if (failed(paddedWorkgroup))
    return function.emitError("workgroup tensor padding failed"), failure();
  scf::SCFTileAndFuseOptions reductionOptions;
  reductionOptions.tilingOptions.setTileSizes(
      {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
       rewriter.getIndexAttr(options.tileK)});
  reductionOptions.setFusionControlFn(
      [](tensor::ExtractSliceOp, OpResult producer, bool destination)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        // Never reinitialize a reduction accumulator on each K iteration.
        if (!destination && isRhsConversion(producer.getOwner()))
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        return std::nullopt;
      });
  RewritePatternSet reductionCleanup(module.getContext());
  addSliceCleanup(reductionCleanup);
  reductionOptions.cleanupPatterns = std::move(reductionCleanup);
  rewriter.setInsertionPoint(*paddedWorkgroup);
  auto reduction = scf::tileConsumerAndFuseProducersUsingSCF(
      rewriter, cast<TilingInterface>(paddedWorkgroup->getOperation()),
      reductionOptions);
  if (failed(reduction) || reduction->loops.size() != 1)
    return function.emitError("structured K tiling and fusion failed"), failure();
  rewriter.replaceOp(*paddedWorkgroup,
                      reduction->replacements.lookup(paddedWorkgroup->getResult(0)));
  auto reductionLoop = cast<scf::ForOp>(reduction->loops.front().getOperation());
  reductionLoop->setAttr(reductionMarker, rewriter.getUnitAttr());
  auto reducedContraction = cast<linalg::MatmulTransposeBOp>(
      reduction->tiledAndFusedOps.front());
  auto paddedReduction = padContraction(rewriter, reducedContraction, {2},
                                        {options.tileK}, true);
  if (failed(paddedReduction))
    return function.emitError("reduction tensor padding failed"), failure();
  (*paddedReduction)->setAttr(contractionMarker, rewriter.getUnitAttr());
  if (failed(cleanupSlices(function)))
    return failure();
  // Fuse the RHS cast slices here. Tiling tensor.pad would add an scf.if.
  SmallVector<tensor::ExtractSliceOp> rhsSlices;
  reductionLoop.walk([&](tensor::ExtractSliceOp slice) {
    if (isRhsConversion(slice.getSource().getDefiningOp()))
      rhsSlices.push_back(slice);
  });
  SmallVector<LoopLikeOpInterface> reductionLoops{reductionLoop};
  for (tensor::ExtractSliceOp slice : rhsSlices) {
    rewriter.setInsertionPoint(slice);
    if (!scf::tileAndFuseProducerOfSlice(rewriter, slice, reductionLoops))
      return function.emitError("structured RHS conversion fusion failed"), failure();
  }
  if (failed(cleanupSlices(function)))
    return failure();
  if (options.prepareSharedMemory &&
      failed(options.prepareSharedMemory(
          rewriter, cast<linalg::LinalgOp>(paddedReduction->getOperation()))))
    return function.emitError("shared tensor placement failed"), failure();
  // Decompose remaining constant pads into fill and insert_slice.
  RewritePatternSet paddingPatterns(module.getContext());
  linalg::populateDecomposePadPatterns(paddingPatterns);
  addSliceCleanup(paddingPatterns);
  if (failed(applyPatternsGreedily(function, std::move(paddingPatterns))) ||
      failed(bufferization::eliminateEmptyTensors(rewriter, function)))
    return failure();
  RewritePatternSet empties(module.getContext());
  bufferization::populateEmptyTensorToAllocTensorPattern(empties);
  if (failed(applyPatternsGreedily(function, std::move(empties))))
    return failure();

  bufferization::OneShotBufferizationOptions bufferOptions;
  bufferOptions.bufferizeFunctionBoundaries = true;
  bufferOptions.functionArgTypeConverterFn =
      [](bufferization::TensorLikeType tensor, Attribute memorySpace, func::FuncOp,
         const bufferization::BufferizationOptions &) -> bufferization::BufferLikeType {
    auto ranked = cast<RankedTensorType>(tensor);
    auto layout = StridedLayoutAttr::get(ranked.getContext(),
                                         ShapedType::kDynamic,
                                         {ShapedType::kDynamic, 1});
    return cast<bufferization::BufferLikeType>(MemRefType::get(
        ranked.getShape(), ranked.getElementType(), layout, memorySpace));
  };
  bufferOptions.memCpyFn = createStructuredBufferCopy;
  bufferization::BufferizationState state;
  if (failed(bufferization::runOneShotModuleBufferize(module, bufferOptions, state)))
    return function.emitError("structured One-Shot Bufferize failed"), failure();
  RewritePatternSet bufferCleanup(module.getContext());
  for (RegisteredOperationName op : module.getContext()->getRegisteredOperations())
    if (op.getDialectNamespace() != "scf")
      op.getCanonicalizationPatterns(bufferCleanup, module.getContext());
  // Keep the K loop even when K < BK.
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      bufferCleanup, scf::ForOp::getOperationName());
  FrozenRewritePatternSet frozenCleanup(std::move(bufferCleanup));
  if (failed(applyPatternsGreedily(function, frozenCleanup)))
    return failure();
  DominanceInfo dominance(function);
  eliminateCommonSubExpressions(rewriter, dominance, function);
  if (failed(applyPatternsGreedily(function, frozenCleanup)))
    return failure();
  if (failed(verify(module)))
    return failure();

  StructuredSchedule result;
  result.function = function;
  function.walk([&](Operation *op) {
    if (op->hasAttr(workgroupMarker))
      result.workgroups = dyn_cast<scf::ForallOp>(op);
    if (op->hasAttr(reductionMarker))
      result.reduction = dyn_cast<scf::ForOp>(op);
    if (op->hasAttr(contractionMarker))
      result.contraction = dyn_cast<linalg::MatmulTransposeBOp>(op);
    if (auto generic = dyn_cast<linalg::GenericOp>(op))
      result.pointwise.push_back(generic);
    if (isa<linalg::FillOp, linalg::CopyOp>(op))
      result.transfers.push_back(cast<linalg::LinalgOp>(op));
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      result.allocations.push_back(op);
  });
  if (!result.workgroups || !result.reduction || !result.contraction)
    return function.emitError("bufferization lost a structured schedule stage"), failure();

  // The output subview offsets are the tiler's M and N coordinates.
  result.workgroups.walk([&](memref::SubViewOp view) {
    Value source = view.getSource();
    while (auto cast = source.getDefiningOp<memref::CastOp>())
      source = cast.getSource();
    if (result.outputTile || source != function.getArgument(2))
      return;
    auto offsets = view.getMixedOffsets();
    rewriter.setInsertionPoint(view);
    result.offsetM = getValueOrCreateConstantIndexOp(rewriter, view.getLoc(), offsets[0]);
    result.offsetN = getValueOrCreateConstantIndexOp(rewriter, view.getLoc(), offsets[1]);
    result.outputTile = view.getResult();
  });
  if (!result.outputTile) {
    // With a single tile the identity slice may have folded away.
    bool singleton = llvm::all_of(result.workgroups.getMixedUpperBound(),
                                 [](OpFoldResult bound) {
                                   return getConstantIntValue(bound) == 1;
                                 });
    if (!singleton)
      return function.emitError("lost the tiled output destination view"), failure();
    rewriter.setInsertionPointToStart(result.workgroups.getBody());
    result.offsetM = result.offsetN =
        arith::ConstantIndexOp::create(rewriter, function.getLoc(), 0);
    result.outputTile = function.getArgument(2);
  }
  return result;
}
