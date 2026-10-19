//===- SharedMemorySchedule.h - Shared-memory staging -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_WGSL_COMPILER_SHAREDMEMORYSCHEDULE_H
#define MLIR_LIB_TARGET_WGSL_COMPILER_SHAREDMEMORYSCHEDULE_H

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::wgsl {

/// Places the input pads of `paddedContraction` in the shared views.
LogicalResult placeSharedInputPads(IRRewriter &rewriter,
                                  linalg::LinalgOp paddedContraction,
                                  ArrayRef<Value> sharedViews);

/// One-Shot Bufferize memcpy callback that keeps copies structured.
LogicalResult createStructuredBufferCopy(OpBuilder &builder, Location loc,
                                        Value source, Value destination);

struct CooperativeCopyOptions {
  int64_t threads;
  int64_t vectorWidth;
};

struct CooperativeCopyGroup {
  /// Operations that replace one input copy, in execution order.
  SmallVector<Operation *> roots;
  /// Set when a partial tile needs a serial copy loop.
  bool hasSerialCopyLoop = false;
};

/// Distributes shared-memory fills, copies and casts across threads.
FailureOr<SmallVector<CooperativeCopyGroup>>
distributeSharedMemoryOperations(IRRewriter &rewriter, Operation *scope,
                                const CooperativeCopyOptions &options);

} // namespace mlir::wgsl

#endif // MLIR_LIB_TARGET_WGSL_COMPILER_SHAREDMEMORYSCHEDULE_H
