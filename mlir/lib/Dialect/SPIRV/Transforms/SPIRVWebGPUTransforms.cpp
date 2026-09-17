//===- SPIRVWebGPUTransforms.cpp - WebGPU-specific transforms -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SPIR-V transforms used when targetting WebGPU.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/SPIRVWebGPUTransforms.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <cstdint>

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVWEBGPUPREPAREPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

namespace mlir {
namespace spirv {
namespace {
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//
static Attribute getScalarOrSplatAttr(Type type, int64_t value) {
  APInt sizedValue(getElementTypeOrSelf(type).getIntOrFloatBitWidth(), value);
  if (auto intTy = dyn_cast<IntegerType>(type))
    return IntegerAttr::get(intTy, sizedValue);

  return SplatElementsAttr::get(cast<ShapedType>(type), sizedValue);
}

static Value lowerExtendedMultiplication(Operation *mulOp,
                                         PatternRewriter &rewriter, Value lhs,
                                         Value rhs, bool signExtendArguments) {
  Location loc = mulOp->getLoc();
  Type argTy = lhs.getType();
  // Emulate 64-bit multiplication by splitting each input element of type i32
  // into 2 16-bit digits of type i32. This is so that the intermediate
  // multiplications and additions do not overflow. We extract these 16-bit
  // digits from i32 vector elements by masking (low digit) and shifting right
  // (high digit).
  //
  // The multiplication algorithm used is the standard (long) multiplication.
  // Multiplying two i32 integers produces 64 bits of result, i.e., 4 16-bit
  // digits.
  //   - With zero-extended arguments, we end up emitting only 4 multiplications
  //     and 4 additions after constant folding.
  //   - With sign-extended arguments, we end up emitting 8 multiplications and
  //     and 12 additions after CSE.
  Value cstLowMask = ConstantOp::create(
      rewriter, loc, lhs.getType(), getScalarOrSplatAttr(argTy, (1 << 16) - 1));
  auto getLowDigit = [&rewriter, loc, cstLowMask](Value val) {
    return BitwiseAndOp::create(rewriter, loc, val, cstLowMask);
  };

  Value cst16 = ConstantOp::create(rewriter, loc, lhs.getType(),
                                   getScalarOrSplatAttr(argTy, 16));
  auto getHighDigit = [&rewriter, loc, cst16](Value val) {
    return ShiftRightLogicalOp::create(rewriter, loc, val, cst16);
  };

  auto getSignDigit = [&rewriter, loc, cst16, &getHighDigit](Value val) {
    // We only need to shift arithmetically by 15, but the extra
    // sign-extension bit will be truncated by the logical shift, so this is
    // fine. We do not have to introduce an extra constant since any
    // value in [15, 32) would do.
    return getHighDigit(
        ShiftRightArithmeticOp::create(rewriter, loc, val, cst16));
  };

  Value cst0 = ConstantOp::create(rewriter, loc, lhs.getType(),
                                  getScalarOrSplatAttr(argTy, 0));

  Value lhsLow = getLowDigit(lhs);
  Value lhsHigh = getHighDigit(lhs);
  Value lhsExt = signExtendArguments ? getSignDigit(lhs) : cst0;
  Value rhsLow = getLowDigit(rhs);
  Value rhsHigh = getHighDigit(rhs);
  Value rhsExt = signExtendArguments ? getSignDigit(rhs) : cst0;

  std::array<Value, 4> lhsDigits = {lhsLow, lhsHigh, lhsExt, lhsExt};
  std::array<Value, 4> rhsDigits = {rhsLow, rhsHigh, rhsExt, rhsExt};
  std::array<Value, 4> resultDigits = {cst0, cst0, cst0, cst0};

  for (auto [i, lhsDigit] : llvm::enumerate(lhsDigits)) {
    for (auto [j, rhsDigit] : llvm::enumerate(rhsDigits)) {
      if (i + j >= resultDigits.size())
        continue;

      if (lhsDigit == cst0 || rhsDigit == cst0)
        continue;

      Value &thisResDigit = resultDigits[i + j];
      Value mul = IMulOp::create(rewriter, loc, lhsDigit, rhsDigit);
      Value current = rewriter.createOrFold<IAddOp>(loc, thisResDigit, mul);
      thisResDigit = getLowDigit(current);

      if (i + j + 1 != resultDigits.size()) {
        Value &nextResDigit = resultDigits[i + j + 1];
        Value carry = rewriter.createOrFold<IAddOp>(loc, nextResDigit,
                                                    getHighDigit(current));
        nextResDigit = carry;
      }
    }
  }

  auto combineDigits = [loc, cst16, &rewriter](Value low, Value high) {
    Value highBits = ShiftLeftLogicalOp::create(rewriter, loc, high, cst16);
    return BitwiseOrOp::create(rewriter, loc, low, highBits);
  };
  Value low = combineDigits(resultDigits[0], resultDigits[1]);
  Value high = combineDigits(resultDigits[2], resultDigits[3]);

  return CompositeConstructOp::create(rewriter, loc,
                                      mulOp->getResultTypes().front(),
                                      llvm::ArrayRef({low, high}));
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

template <typename MulExtendedOp, bool SignExtendArguments>
struct ExpandMulExtendedPattern final : OpRewritePattern<MulExtendedOp> {
  using OpRewritePattern<MulExtendedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();

    // Currently, WGSL only supports 32-bit integer types. Any other integer
    // types should already have been promoted/demoted to i32.
    auto elemTy = cast<IntegerType>(getElementTypeOrSelf(lhs.getType()));
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(
          loc,
          llvm::formatv("Unexpected integer type for WebGPU: '{0}'", elemTy));

    Value mul = lowerExtendedMultiplication(op, rewriter, lhs, rhs,
                                            SignExtendArguments);
    rewriter.replaceOp(op, mul);
    return success();
  }
};

using ExpandSMulExtendedPattern =
    ExpandMulExtendedPattern<SMulExtendedOp, true>;
using ExpandUMulExtendedPattern =
    ExpandMulExtendedPattern<UMulExtendedOp, false>;

template <typename Op, typename ArithOp>
struct ExpandAddCarryOrSubBorrowPattern final : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();

    // Currently, WGSL only supports 32-bit integer types. Any other integer
    // types should already have been promoted/demoted to i32.
    Type argTy = lhs.getType();
    auto elemTy = cast<IntegerType>(getElementTypeOrSelf(argTy));
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(
          loc,
          llvm::formatv("Unexpected integer type for WebGPU: '{0}'", elemTy));

    Value one = ConstantOp::create(rewriter, loc, argTy,
                                   getScalarOrSplatAttr(argTy, 1));
    Value zero = ConstantOp::create(rewriter, loc, argTy,
                                    getScalarOrSplatAttr(argTy, 0));

    Value out = ArithOp::create(rewriter, loc, lhs, rhs);
    // For add: carry iff out < lhs (unsigned overflow).
    // For sub: borrow iff lhs < rhs (unsigned underflow).
    Value cmp;
    if constexpr (std::is_same_v<Op, IAddCarryOp>)
      cmp = ULessThanOp::create(rewriter, loc, out, lhs);
    else
      cmp = ULessThanOp::create(rewriter, loc, lhs, rhs);
    Value flag = SelectOp::create(rewriter, loc, cmp, one, zero);

    Value result = CompositeConstructOp::create(rewriter, loc,
                                                op->getResultTypes().front(),
                                                llvm::ArrayRef({out, flag}));

    rewriter.replaceOp(op, result);
    return success();
  }
};

using ExpandAddCarryPattern =
    ExpandAddCarryOrSubBorrowPattern<IAddCarryOp, IAddOp>;
using ExpandSubBorrowPattern =
    ExpandAddCarryOrSubBorrowPattern<ISubBorrowOp, ISubOp>;

struct ExpandIsInfPattern final : OpRewritePattern<IsInfOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IsInfOp op,
                                PatternRewriter &rewriter) const override {
    // We assume values to be finite and turn `IsInf` info `false`.
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        op, op.getType(), getScalarOrSplatAttr(op.getType(), 0));
    return success();
  }
};

struct ExpandIsNanPattern final : OpRewritePattern<IsNanOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IsNanOp op,
                                PatternRewriter &rewriter) const override {
    // We assume values to be finite and turn `IsNan` info `false`.
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
        op, op.getType(), getScalarOrSplatAttr(op.getType(), 0));
    return success();
  }
};

/// WGSL is strictly typed and Naga assigns `u32` to values whose SPIR-V type has
/// signedness 0, which is how MLIR serializes signless integers. Naga then
/// rejects `OpSNegate` on such a value and compiles `GLSL.std.450 SAbs` as a
/// plain `abs`, which is the identity on `u32`. Both ops are emitted by the
/// `arith.remsi` lowering. Rewrite them into ops that mean the same thing for
/// any signedness: `SNegate x` is `0 - x` in two's complement, and `SAbs x` is
/// `select(x <s 0, 0 - x, x)`.
struct ExpandSNegatePattern final : OpRewritePattern<SNegateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SNegateOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value zero = ConstantOp::getZero(type, loc, rewriter);
    rewriter.replaceOpWithNewOp<ISubOp>(op, type, zero, op.getOperand());
    return success();
  }
};

struct ExpandSAbsPattern final : OpRewritePattern<GLSAbsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GLSAbsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value operand = op.getOperand();
    Value zero = ConstantOp::getZero(type, loc, rewriter);
    Value negated = ISubOp::create(rewriter, loc, type, zero, operand);
    Type boolType = rewriter.getI1Type();
    if (auto vecType = dyn_cast<VectorType>(type))
      boolType = VectorType::get(vecType.getShape(), boolType);
    Value isNegative =
        SLessThanOp::create(rewriter, loc, boolType, operand, zero);
    rewriter.replaceOpWithNewOp<SelectOp>(op, type, isNegative, negated,
                                          operand);
    return success();
  }
};

/// `ShiftRightArithmetic` on a signless integer is emitted by the extended
/// multiplication expansion above (sign extension of the 16 bit digits) and by
/// `arith.shrsi`. Naga emits it as a logical shift on `u32`, which silently
/// drops the sign. Express it through logical shifts only:
/// `sra(x, n) = select(x <s 0, ~(~x >>l n), x >>l n)`.
struct ExpandShiftRightArithmeticPattern final
    : OpRewritePattern<ShiftRightArithmeticOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShiftRightArithmeticOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value x = op.getOperand1();
    Value n = op.getOperand2();
    Value zero = ConstantOp::getZero(type, loc, rewriter);
    Value logical = ShiftRightLogicalOp::create(rewriter, loc, type, x, n);
    Value notX = NotOp::create(rewriter, loc, type, x);
    Value notShifted = ShiftRightLogicalOp::create(rewriter, loc, type, notX, n);
    Value negativeCase = NotOp::create(rewriter, loc, type, notShifted);
    Type boolType = rewriter.getI1Type();
    if (auto vecType = dyn_cast<VectorType>(type))
      boolType = VectorType::get(vecType.getShape(), boolType);
    Value isNegative = SLessThanOp::create(rewriter, loc, boolType, x, zero);
    rewriter.replaceOpWithNewOp<SelectOp>(op, type, isNegative, negativeCase,
                                          logical);
    return success();
  }
};

/// Returns `select(x <s 0, 0 - x, x)`, the absolute value spelled without a
/// signed builtin.
static Value emitSignlessAbs(PatternRewriter &rewriter, Location loc, Value x,
                             Value zero, Type boolType) {
  Type type = x.getType();
  Value neg = ISubOp::create(rewriter, loc, type, zero, x);
  Value isNeg = SLessThanOp::create(rewriter, loc, boolType, x, zero);
  return SelectOp::create(rewriter, loc, type, isNeg, neg, x);
}

static Type boolTypeFor(PatternRewriter &rewriter, Type type) {
  Type boolType = rewriter.getI1Type();
  if (auto vecType = dyn_cast<VectorType>(type))
    boolType = VectorType::get(vecType.getShape(), boolType);
  return boolType;
}

/// `SDiv` on a signless integer is read by Naga as an unsigned division.
/// Express it as `UDiv(|a|, |b|)` with the sign restored:
/// `q = udiv(|a|, |b|); select((a <s 0) != (b <s 0), 0 - q, q)`.
struct ExpandSDivPattern final : OpRewritePattern<SDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SDivOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getType();
    Type boolType = boolTypeFor(rewriter, type);
    Value a = op.getOperand1();
    Value b = op.getOperand2();
    Value zero = ConstantOp::getZero(type, loc, rewriter);
    Value absA = emitSignlessAbs(rewriter, loc, a, zero, boolType);
    Value absB = emitSignlessAbs(rewriter, loc, b, zero, boolType);
    Value q = UDivOp::create(rewriter, loc, type, absA, absB);
    Value negQ = ISubOp::create(rewriter, loc, type, zero, q);
    Value aNeg = SLessThanOp::create(rewriter, loc, boolType, a, zero);
    Value bNeg = SLessThanOp::create(rewriter, loc, boolType, b, zero);
    Value signDiffers =
        LogicalNotEqualOp::create(rewriter, loc, boolType, aNeg, bNeg);
    rewriter.replaceOpWithNewOp<SelectOp>(op, type, signDiffers, negQ, q);
    return success();
  }
};

/// `GLSL.std.450 SMin` and `SMax` on signless integers become unsigned min
/// and max in a WGSL translator that typed the values `u32`. A signed
/// comparison plus select says the same thing for any signedness.
template <typename Op, bool IsMin>
struct ExpandSMinMaxPattern final : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type type = op.getType();
    Type boolType = boolTypeFor(rewriter, type);
    Value a = op.getOperand(0);
    Value b = op.getOperand(1);
    Value aLess = SLessThanOp::create(rewriter, loc, boolType, a, b);
    if (IsMin)
      rewriter.replaceOpWithNewOp<SelectOp>(op, type, aLess, a, b);
    else
      rewriter.replaceOpWithNewOp<SelectOp>(op, type, aLess, b, a);
    return success();
  }
};
using ExpandSMinPattern = ExpandSMinMaxPattern<GLSMinOp, true>;
using ExpandSMaxPattern = ExpandSMinMaxPattern<GLSMaxOp, false>;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//
struct WebGPUPreparePass final
    : impl::SPIRVWebGPUPreparePassBase<WebGPUPreparePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateSPIRVExpandExtendedMultiplicationPatterns(patterns);
    populateSPIRVExpandNonFiniteArithmeticPatterns(patterns);
    populateSPIRVExpandSignednessDependentIntegerPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public Interface
//===----------------------------------------------------------------------===//
void populateSPIRVExpandExtendedMultiplicationPatterns(
    RewritePatternSet &patterns) {
  // WGSL currently does not support extended multiplication ops, see:
  // https://github.com/gpuweb/gpuweb/issues/1565.
  patterns.add<ExpandSMulExtendedPattern, ExpandUMulExtendedPattern,
               ExpandAddCarryPattern, ExpandSubBorrowPattern>(
      patterns.getContext());
}

void populateSPIRVExpandNonFiniteArithmeticPatterns(
    RewritePatternSet &patterns) {
  // WGSL currently does not support `isInf` and `isNan`, see:
  // https://github.com/gpuweb/gpuweb/pull/2311.
  patterns.add<ExpandIsInfPattern, ExpandIsNanPattern>(patterns.getContext());
}

} // namespace spirv
} // namespace mlir

void mlir::spirv::populateSPIRVExpandSignednessDependentIntegerPatterns(
    RewritePatternSet &patterns) {
  // SNegate and SAbs read a sign that signless MLIR integers do not carry into
  // SPIR-V. Naga types such values as u32 and rejects or misreads the ops, see
  // the discussion in the pass documentation.
  patterns.add<ExpandSNegatePattern, ExpandSAbsPattern,
               ExpandShiftRightArithmeticPattern, ExpandSDivPattern,
               ExpandSMinPattern, ExpandSMaxPattern>(patterns.getContext());
}
