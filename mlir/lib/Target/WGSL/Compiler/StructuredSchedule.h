//===- StructuredSchedule.h - Structured matmul schedule --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_WGSL_COMPILER_STRUCTUREDSCHEDULE_H
#define MLIR_LIB_TARGET_WGSL_COMPILER_STRUCTUREDSCHEDULE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include <functional>

namespace mlir::wgsl {

struct StructuredScheduleOptions {
  int64_t tileM, tileN, tileK;

  // Places input pads in shared memory before bufferization.
  std::function<LogicalResult(IRRewriter &, linalg::LinalgOp)>
      prepareSharedMemory;
};

struct StructuredSchedule {
  // The rewritten function, with A, B and C as memref arguments.
  func::FuncOp function;
  scf::ForallOp workgroups;
  scf::ForOp reduction;
  linalg::MatmulTransposeBOp contraction;
  // Output tile coordinates from the tiler.
  Value offsetM, offsetN, outputTile;
  SmallVector<linalg::GenericOp> pointwise;
  SmallVector<linalg::LinalgOp> transfers;
  SmallVector<Operation *> allocations;
};

// Registers the upstream interfaces the schedule needs.
void registerStructuredScheduleInterfaces(DialectRegistry &registry);

// Tiles, pads and bufferizes the matmul. The caller lowers the result.
FailureOr<StructuredSchedule>
buildStructuredSchedule(ModuleOp module, linalg::MatmulTransposeBOp matmul,
                        const StructuredScheduleOptions &options);

} // namespace mlir::wgsl

#endif // MLIR_LIB_TARGET_WGSL_COMPILER_STRUCTUREDSCHEDULE_H
