//===- SPIRVWebGPUTransforms.h - WebGPU-specific Transforms -*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines SPIR-V transforms used when targetting WebGPU.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRV_WEBGPU_TRANSFORMS_H
#define MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRV_WEBGPU_TRANSFORMS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace spirv {

/// Appends patterns to expand extended multiplication and adition ops into
/// regular arithmetic ops. Extended arithmetic ops are not supported by the
/// WebGPU Shading Language (WGSL).
void populateSPIRVExpandExtendedMultiplicationPatterns(
    RewritePatternSet &patterns);

/// Appends patterns to expand non-finite arithmetic ops `IsNan` and `IsInf`.
/// These are not supported by the WebGPU Shading Language (WGSL). We follow
/// fast math assumptions and assume that all floating point values are finite.
/// Appends patterns to expand `spirv.SNegate`, `spirv.GL.SAbs`,
/// `spirv.ShiftRightArithmetic`, `spirv.SDiv`, `spirv.GL.SMin` and
/// `spirv.GL.SMax` on signless
/// integers into ops whose meaning does not depend on the SPIR-V signedness
/// bit, which WGSL translators use to pick `i32` or `u32`.
void populateSPIRVExpandSignednessDependentIntegerPatterns(
    RewritePatternSet &patterns);

void populateSPIRVExpandNonFiniteArithmeticPatterns(
    RewritePatternSet &patterns);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRV_WEBGPU_TRANSFORMS_H
