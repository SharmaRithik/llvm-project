//===- VectorUnrollLoopCarried.cpp - Split loop-carried vectors -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_VECTORUNROLLLOOPCARRIED
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

using namespace mlir;

/// The type of one tile: the tile shape without its leading unit dims.
static VectorType getTileType(VectorType type, ArrayRef<int64_t> tileShape) {
  ArrayRef<int64_t> shape = tileShape;
  while (shape.size() > 1 && shape.front() == 1)
    shape = shape.drop_front();
  return VectorType::get(shape, type.getElementType());
}

/// Returns the tile shape for `type`, or failure if it is not split.
static FailureOr<SmallVector<int64_t>>
getTileShape(VectorType type, ArrayRef<int64_t> targetShape) {
  if (targetShape.empty() ||
      llvm::any_of(targetShape, [](int64_t dim) { return dim <= 0; }))
    return failure();
  if (type.isScalable() || type.getRank() != (int64_t)targetShape.size())
    return failure();
  if (type.getShape() == targetShape && getTileType(type, targetShape) == type)
    return failure();
  std::optional<SmallVector<int64_t>> ratio =
      computeShapeRatio(type.getShape(), targetShape);
  if (!ratio)
    return failure();
  return llvm::to_vector(targetShape);
}

void mlir::vector::populateVectorUnrollLoopCarriedConversion(
    TypeConverter &converter, ArrayRef<int64_t> targetShape) {
  SmallVector<int64_t> target(targetShape);
  converter.addConversion([](Type type) { return type; });
  converter.addConversion(
      [target](VectorType type,
               SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
        FailureOr<SmallVector<int64_t>> tile = getTileShape(type, target);
        if (failed(tile))
          return std::nullopt;
        int64_t count = computeProduct(
            *computeShapeRatio(type.getShape(), ArrayRef<int64_t>(*tile)));
        results.append(count, getTileType(type, *tile));
        return success();
      });

  // Tiles -> whole vector.
  converter.addSourceMaterialization([target](OpBuilder &b, Type resultType,
                                              ValueRange inputs,
                                              Location loc) -> Value {
    auto type = dyn_cast<VectorType>(resultType);
    if (!type)
      return Value();
    FailureOr<SmallVector<int64_t>> tile = getTileShape(type, target);
    if (failed(tile))
      return Value();
    Value result = ub::PoisonOp::create(b, loc, type);
    VectorType tileType = VectorType::get(*tile, type.getElementType());
    SmallVector<int64_t> strides(tile->size(), 1);
    for (auto [input, offsets] :
         llvm::zip(inputs, StaticTileOffsetRange(type.getShape(), *tile))) {
      Value piece = input;
      if (piece.getType() != tileType)
        piece = vector::ShapeCastOp::create(b, loc, tileType, piece);
      result = vector::InsertStridedSliceOp::create(b, loc, piece, result,
                                                    offsets, strides);
    }
    return result;
  });

  // Whole vector -> tiles.
  converter.addTargetMaterialization(
      [target](OpBuilder &b, TypeRange resultTypes, ValueRange inputs,
               Location loc) -> SmallVector<Value> {
        if (inputs.size() != 1)
          return {};
        auto type = dyn_cast<VectorType>(inputs[0].getType());
        if (!type)
          return {};
        FailureOr<SmallVector<int64_t>> tile = getTileShape(type, target);
        if (failed(tile))
          return {};
        SmallVector<int64_t> strides(tile->size(), 1);
        SmallVector<Value> results;
        for (auto [resultType, offsets] : llvm::zip(
                 resultTypes, StaticTileOffsetRange(type.getShape(), *tile))) {
          Value piece = vector::ExtractStridedSliceOp::create(
              b, loc, inputs[0], offsets, *tile, strides);
          if (piece.getType() != resultType)
            piece = vector::ShapeCastOp::create(b, loc, resultType, piece);
          results.push_back(piece);
        }
        return results;
      });
}

namespace {
struct VectorUnrollLoopCarriedPass
    : public vector::impl::VectorUnrollLoopCarriedBase<
          VectorUnrollLoopCarriedPass> {
  using Base::Base;

  void runOnOperation() override {
    if (targetShape.empty() ||
        llvm::any_of(targetShape, [](int64_t dim) { return dim <= 0; })) {
      getOperation()->emitError("target-shape must contain positive dimensions");
      return signalPassFailure();
    }
    MLIRContext *ctx = &getContext();
    TypeConverter converter;
    vector::populateVectorUnrollLoopCarriedConversion(converter, targetShape);

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    // Fold the extract/insert_strided_slice glue left by the materializations.
    RewritePatternSet cleanup(ctx);
    vector::ExtractStridedSliceOp::getCanonicalizationPatterns(cleanup, ctx);
    vector::InsertStridedSliceOp::getCanonicalizationPatterns(cleanup, ctx);
    vector::ShapeCastOp::getCanonicalizationPatterns(cleanup, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(cleanup))))
      return signalPassFailure();
  }
};
} // namespace
