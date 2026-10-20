//===- RuntimeBindings.h - WGSL runtime bindings ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_WGSL_COMPILER_RUNTIMEBINDINGS_H
#define MLIR_LIB_TARGET_WGSL_COMPILER_RUNTIMEBINDINGS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::wgsl {

/// Turns scalar bindings tagged with wgsl.pack_width into vec4 storage.
/// Scalar stores to these bindings are rejected.
LogicalResult lowerPackedStorageBindings(ModuleOp module);

} // namespace mlir::wgsl

#endif // MLIR_LIB_TARGET_WGSL_COMPILER_RUNTIMEBINDINGS_H
