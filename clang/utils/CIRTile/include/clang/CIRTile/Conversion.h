//===- Conversion.h - CIR to CUDA Tile conversion -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIRTILE_CONVERSION_H
#define CLANG_CIRTILE_CONVERSION_H

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace clang::CIRTile {

mlir::OwningOpRef<mlir::cuda_tile::ModuleOp>
convertToCudaTile(mlir::ModuleOp module);

} // namespace clang::CIRTile

#endif // CLANG_CIRTILE_CONVERSION_H
