//=- VectorEmulateMaskedLoadStore.cpp - Emulate 'vector.maskedload/store' op =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to emulate the
// 'vector.maskedload' and 'vector.maskedstore' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

/// Returns the scalar condition of a mask whose lanes are all equal.
static Value getUniformMaskCondition(Value mask, PatternRewriter &rewriter) {
  if (auto broadcast = mask.getDefiningOp<vector::BroadcastOp>()) {
    if (broadcast.getSource().getType().isInteger(1))
      return broadcast.getSource();
  }

  auto getSplatValue = [](Value value) -> std::optional<bool> {
    DenseIntElementsAttr attr;
    if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat() ||
        !attr.getElementType().isInteger(1))
      return std::nullopt;
    return attr.getSplatValue<bool>();
  };

  Location loc = mask.getLoc();
  if (std::optional<bool> value = getSplatValue(mask))
    return arith::ConstantIntOp::create(rewriter, loc, *value, 1);

  auto select = mask.getDefiningOp<arith::SelectOp>();
  if (!select || !select.getCondition().getType().isInteger(1))
    return {};
  std::optional<bool> trueValue = getSplatValue(select.getTrueValue());
  std::optional<bool> falseValue = getSplatValue(select.getFalseValue());
  if (!trueValue || !falseValue)
    return {};
  if (*trueValue == *falseValue)
    return arith::ConstantIntOp::create(rewriter, loc, *trueValue, 1);
  if (*trueValue)
    return select.getCondition();

  Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
  return arith::XOrIOp::create(rewriter, loc, select.getCondition(), one);
}

/// Convert vector.maskedload
///
/// Before:
///
///   vector.maskedload %base[%idx_0, %idx_1], %mask, %pass_thru
///
/// After:
///
///   %ivalue = %pass_thru
///   %m = vector.extract %mask[0]
///   %result0 = scf.if %m {
///     %v = memref.load %base[%idx_0, %idx_1]
///     %combined = vector.insert %v, %ivalue[0]
///     scf.yield %combined
///   } else {
///     scf.yield %ivalue
///   }
///   %m = vector.extract %mask[1]
///   %result1 = scf.if %m {
///     %v = memref.load %base[%idx_0, %idx_1 + 1]
///     %combined = vector.insert %v, %result0[1]
///     scf.yield %combined
///   } else {
///     scf.yield %result0
///   }
///   ...
///
struct VectorMaskedLoadOpConverter final
    : OpRewritePattern<vector::MaskedLoadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp maskedLoadOp,
                                PatternRewriter &rewriter) const override {
    VectorType maskVType = maskedLoadOp.getMaskVectorType();
    if (maskVType.getShape().size() != 1)
      return rewriter.notifyMatchFailure(
          maskedLoadOp, "expected vector.maskedload with 1-D mask");
    if (maskVType.isScalable())
      return rewriter.notifyMatchFailure(maskedLoadOp,
                                         "cannot scalarize a scalable mask");

    Location loc = maskedLoadOp.getLoc();
    if (Value condition =
            getUniformMaskCondition(maskedLoadOp.getMask(), rewriter)) {
      auto ifOp = scf::IfOp::create(
          rewriter, loc, condition,
          [&](OpBuilder &builder, Location loc) {
            Value loaded = vector::LoadOp::create(
                builder, loc, maskedLoadOp.getVectorType(),
                maskedLoadOp.getBase(), maskedLoadOp.getIndices(),
                /*nontemporal=*/false,
                llvm::MaybeAlign(maskedLoadOp.getAlignment().value_or(0)));
            scf::YieldOp::create(builder, loc, loaded);
          },
          [&](OpBuilder &builder, Location loc) {
            scf::YieldOp::create(builder, loc, maskedLoadOp.getPassThru());
          });
      rewriter.replaceOp(maskedLoadOp, ifOp.getResults());
      return success();
    }

    int64_t maskLength = maskVType.getShape()[0];

    Type indexType = rewriter.getIndexType();
    Value mask = maskedLoadOp.getMask();
    Value base = maskedLoadOp.getBase();
    Value iValue = maskedLoadOp.getPassThru();
    auto indices = llvm::to_vector_of<Value>(maskedLoadOp.getIndices());
    Value one = arith::ConstantOp::create(rewriter, loc, indexType,
                                          IntegerAttr::get(indexType, 1));
    for (int64_t i = 0; i < maskLength; ++i) {
      auto maskBit = vector::ExtractOp::create(rewriter, loc, mask, i);

      auto ifOp = scf::IfOp::create(
          rewriter, loc, maskBit,
          [&](OpBuilder &builder, Location loc) {
            auto loadedValue = memref::LoadOp::create(
                builder, loc, base, indices, /*nontemporal=*/false,
                llvm::MaybeAlign(maskedLoadOp.getAlignment().value_or(0)));
            auto combinedValue =
                vector::InsertOp::create(builder, loc, loadedValue, iValue, i);
            scf::YieldOp::create(builder, loc, combinedValue.getResult());
          },
          [&](OpBuilder &builder, Location loc) {
            scf::YieldOp::create(builder, loc, iValue);
          });
      iValue = ifOp.getResult(0);
      if (!indices.empty())
        indices.back() =
            arith::AddIOp::create(rewriter, loc, indices.back(), one);
    }

    rewriter.replaceOp(maskedLoadOp, iValue);

    return success();
  }
};

/// Convert vector.maskedstore
///
/// Before:
///
///   vector.maskedstore %base[%idx_0, %idx_1], %mask, %value
///
/// After:
///
///   %m = vector.extract %mask[0]
///   scf.if %m {
///     %extracted = vector.extract %value[0]
///     memref.store %extracted, %base[%idx_0, %idx_1]
///   }
///   %m = vector.extract %mask[1]
///   scf.if %m {
///     %extracted = vector.extract %value[1]
///     memref.store %extracted, %base[%idx_0, %idx_1 + 1]
///   }
///   ...
///
struct VectorMaskedStoreOpConverter final
    : OpRewritePattern<vector::MaskedStoreOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp maskedStoreOp,
                                PatternRewriter &rewriter) const override {
    VectorType maskVType = maskedStoreOp.getMaskVectorType();
    if (maskVType.getShape().size() != 1)
      return rewriter.notifyMatchFailure(
          maskedStoreOp, "expected vector.maskedstore with 1-D mask");
    if (maskVType.isScalable())
      return rewriter.notifyMatchFailure(maskedStoreOp,
                                         "cannot scalarize a scalable mask");

    Location loc = maskedStoreOp.getLoc();
    if (Value condition =
            getUniformMaskCondition(maskedStoreOp.getMask(), rewriter)) {
      auto ifOp = scf::IfOp::create(rewriter, loc, condition, /*else=*/false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        vector::StoreOp::create(
            rewriter, loc, maskedStoreOp.getValueToStore(),
            maskedStoreOp.getBase(), maskedStoreOp.getIndices(),
            /*nontemporal=*/false,
            llvm::MaybeAlign(maskedStoreOp.getAlignment().value_or(0)));
      }
      rewriter.eraseOp(maskedStoreOp);
      return success();
    }

    int64_t maskLength = maskVType.getShape()[0];

    Type indexType = rewriter.getIndexType();
    Value mask = maskedStoreOp.getMask();
    Value base = maskedStoreOp.getBase();
    Value value = maskedStoreOp.getValueToStore();
    bool nontemporal = false;
    auto indices = llvm::to_vector_of<Value>(maskedStoreOp.getIndices());
    Value one = arith::ConstantOp::create(rewriter, loc, indexType,
                                          IntegerAttr::get(indexType, 1));
    for (int64_t i = 0; i < maskLength; ++i) {
      auto maskBit = vector::ExtractOp::create(rewriter, loc, mask, i);

      auto ifOp = scf::IfOp::create(rewriter, loc, maskBit, /*else=*/false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto extractedValue = vector::ExtractOp::create(rewriter, loc, value, i);
      memref::StoreOp::create(
          rewriter, loc, extractedValue, base, indices, nontemporal,
          llvm::MaybeAlign(maskedStoreOp.getAlignment().value_or(0)));

      rewriter.setInsertionPointAfter(ifOp);
      if (!indices.empty())
        indices.back() =
            arith::AddIOp::create(rewriter, loc, indices.back(), one);
    }

    rewriter.eraseOp(maskedStoreOp);

    return success();
  }
};

} // namespace

void mlir::vector::populateVectorMaskedLoadStoreEmulationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<VectorMaskedLoadOpConverter, VectorMaskedStoreOpConverter>(
      patterns.getContext(), benefit);
}
