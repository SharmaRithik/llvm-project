//===- KernelOutlining.h - Outline WGSL kernels -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_WGSL_COMPILER_KERNELOUTLINING_H
#define MLIR_LIB_TARGET_WGSL_COMPILER_KERNELOUTLINING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::wgsl {

/// Temporary host function and launch for one kernel. The last three host
/// arguments are the dispatch size.
struct KernelLaunch {
  func::FuncOp host;
  gpu::LaunchOp launch;
  unsigned numBindings;
};

/// Creates the host and launch, with insertion at the start of the body.
KernelLaunch createKernelLaunch(IRRewriter &builder, ModuleOp module,
                                Location loc, TypeRange bindingTypes,
                                TypeRange workgroupTypes,
                                ArrayRef<int32_t> blockSize);

/// Outlines the launch with gpu-kernel-outlining and erases the host.
FailureOr<gpu::GPUFuncOp> outlineKernelLaunch(ModuleOp module,
                                            KernelLaunch kernel);

} // namespace mlir::wgsl

#endif // MLIR_LIB_TARGET_WGSL_COMPILER_KERNELOUTLINING_H
