//===- WGSLEmitter.h - Translation to WGSL ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_WGSL_WGSLEMITTER_H
#define MLIR_TARGET_WGSL_WGSLEMITTER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
namespace wgsl {

/// Translates the kernel gpu.func in `op` to a WGSL compute shader. Argument
/// attributes wgsl.binding, wgsl.group, wgsl.address_space, wgsl.access and
/// wgsl.name control the generated resource variables.
LogicalResult translateToWGSL(Operation *op, raw_ostream &os,
                              StringRef kernelName = {});

} // namespace wgsl

/// Registers the WGSL emitter translation with MLIR.
void registerToWGSLTranslation();

} // namespace mlir

#endif // MLIR_TARGET_WGSL_WGSLEMITTER_H
