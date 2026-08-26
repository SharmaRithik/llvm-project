//===- Annotation.h - CIR Tile annotation decoding --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIRTILE_ANNOTATION_H
#define CLANG_CIRTILE_ANNOTATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace clang::CIRTile {

enum class IntrinsicKind { Kernel, BlockId, Zero, Load, Mma, Store };

struct IntrinsicAnnotation {
  IntrinsicKind kind;
  llvm::SmallVector<int64_t, 4> arguments;
};

llvm::StringRef getAnnotationName(IntrinsicKind kind);

mlir::FailureOr<std::optional<IntrinsicAnnotation>>
decodeAnnotation(cir::FuncOp function);

mlir::LogicalResult validateAnnotations(mlir::ModuleOp module,
                                        llvm::raw_ostream *output = nullptr);

} // namespace clang::CIRTile

#endif // CLANG_CIRTILE_ANNOTATION_H
