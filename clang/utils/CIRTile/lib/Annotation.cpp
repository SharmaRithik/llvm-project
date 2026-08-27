//===- Annotation.cpp - CIR Tile annotation decoding ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRTile/Annotation.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <utility>

namespace clang::CIRTile {
namespace {

constexpr llvm::StringLiteral AnnotationPrefix = "cir_tile.";
constexpr llvm::StringLiteral AnnotationV1Prefix = "cir_tile.v1.";

std::optional<IntrinsicKind> parseKind(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<IntrinsicKind>>(name)
      .Case("kernel", IntrinsicKind::Kernel)
      .Case("bid", IntrinsicKind::BlockId)
      .Case("zero", IntrinsicKind::Zero)
      .Case("load", IntrinsicKind::Load)
      .Case("mma", IntrinsicKind::Mma)
      .Case("store", IntrinsicKind::Store)
      .Default(std::nullopt);
}

unsigned getArgumentCount(IntrinsicKind kind) {
  switch (kind) {
  case IntrinsicKind::Kernel:
    return 0;
  case IntrinsicKind::BlockId:
    return 1;
  case IntrinsicKind::Zero:
    return 2;
  case IntrinsicKind::Load:
  case IntrinsicKind::Store:
    return 4;
  case IntrinsicKind::Mma:
    return 3;
  }
  llvm_unreachable("unknown CIR Tile intrinsic");
}

bool isValidDimension(int64_t value) {
  return value > 0 && value <= std::numeric_limits<int32_t>::max() &&
         llvm::isPowerOf2_64(static_cast<uint64_t>(value));
}

mlir::LogicalResult
validateTileElementCount(cir::FuncOp function,
                         const IntrinsicAnnotation &annotation, int64_t rows,
                         int64_t columns) {
  if (rows <= mlir::cuda_tile::maxTileNumElements / columns)
    return mlir::success();
  function.emitError()
      << "annotation '" << getAnnotationName(annotation.kind)
      << "' produces a tile larger than the CUDA Tile limit of "
      << mlir::cuda_tile::maxTileNumElements << " elements";
  return mlir::failure();
}

mlir::LogicalResult
validateAnnotationValues(cir::FuncOp function,
                         const IntrinsicAnnotation &annotation) {
  if (annotation.kind == IntrinsicKind::BlockId) {
    if (annotation.arguments[0] >= 0 && annotation.arguments[0] <= 2)
      return mlir::success();
    function.emitError() << "annotation '" << getAnnotationName(annotation.kind)
                         << "' requires an axis from 0 to 2";
    return mlir::failure();
  }
  if (annotation.kind == IntrinsicKind::Kernel)
    return mlir::success();

  if (!llvm::all_of(annotation.arguments, isValidDimension)) {
    function.emitError()
        << "annotation '" << getAnnotationName(annotation.kind)
        << "' requires positive power-of-two dimensions that fit in 32 bits";
    return mlir::failure();
  }

  if (annotation.kind == IntrinsicKind::Load ||
      annotation.kind == IntrinsicKind::Store) {
    if (annotation.arguments[2] % annotation.arguments[0] != 0 ||
        annotation.arguments[3] % annotation.arguments[1] != 0) {
      function.emitError()
          << "annotation '" << getAnnotationName(annotation.kind)
          << "' requires tensor dimensions divisible by tile dimensions";
      return mlir::failure();
    }
    return validateTileElementCount(
        function, annotation, annotation.arguments[0], annotation.arguments[1]);
  }

  if (annotation.kind == IntrinsicKind::Zero)
    return validateTileElementCount(
        function, annotation, annotation.arguments[0], annotation.arguments[1]);

  for (auto [rows, columns] :
       {std::pair(annotation.arguments[0], annotation.arguments[2]),
        std::pair(annotation.arguments[2], annotation.arguments[1]),
        std::pair(annotation.arguments[0], annotation.arguments[1])})
    if (mlir::failed(
            validateTileElementCount(function, annotation, rows, columns)))
      return mlir::failure();
  return mlir::success();
}

} // namespace

llvm::StringRef getAnnotationName(IntrinsicKind kind) {
  switch (kind) {
  case IntrinsicKind::Kernel:
    return "cir_tile.v1.kernel";
  case IntrinsicKind::BlockId:
    return "cir_tile.v1.bid";
  case IntrinsicKind::Zero:
    return "cir_tile.v1.zero";
  case IntrinsicKind::Load:
    return "cir_tile.v1.load";
  case IntrinsicKind::Mma:
    return "cir_tile.v1.mma";
  case IntrinsicKind::Store:
    return "cir_tile.v1.store";
  }
  llvm_unreachable("unknown CIR Tile intrinsic");
}

mlir::FailureOr<std::optional<IntrinsicAnnotation>>
decodeAnnotation(cir::FuncOp function) {
  std::optional<IntrinsicAnnotation> result;
  mlir::ArrayAttr annotations = function.getAnnotationsAttr();
  if (!annotations)
    return result;

  for (mlir::Attribute attribute : annotations) {
    auto annotation = mlir::dyn_cast<cir::AnnotationAttr>(attribute);
    if (!annotation)
      continue;

    llvm::StringRef name = annotation.getName().getValue();
    if (!name.starts_with(AnnotationPrefix))
      continue;
    if (result) {
      function.emitError("has more than one CIR Tile annotation");
      return mlir::failure();
    }
    if (!name.starts_with(AnnotationV1Prefix)) {
      function.emitError() << "uses unsupported CIR Tile annotation '" << name
                           << "', expected cir_tile.v1.*";
      return mlir::failure();
    }

    std::optional<IntrinsicKind> kind =
        parseKind(name.drop_front(AnnotationV1Prefix.size()));
    if (!kind) {
      function.emitError() << "uses unknown CIR Tile annotation '" << name
                           << "'";
      return mlir::failure();
    }

    mlir::ArrayAttr arguments = annotation.getArgs();
    size_t argumentCount = arguments ? arguments.size() : 0;
    if (argumentCount != getArgumentCount(*kind)) {
      function.emitError() << "annotation '" << name << "' expects "
                           << getArgumentCount(*kind)
                           << " integer arguments, got " << argumentCount;
      return mlir::failure();
    }

    result.emplace(IntrinsicAnnotation{*kind, {}});
    if (arguments) {
      for (mlir::Attribute argument : arguments) {
        auto integer = mlir::dyn_cast<mlir::IntegerAttr>(argument);
        if (!integer) {
          function.emitError()
              << "annotation '" << name << "' requires integer arguments";
          return mlir::failure();
        }
        result->arguments.push_back(integer.getInt());
      }
    }
  }
  if (result && mlir::failed(validateAnnotationValues(function, *result)))
    return mlir::failure();
  return result;
}

mlir::LogicalResult validateIntrinsic(cir::FuncOp function,
                                      const IntrinsicAnnotation &annotation) {
  cir::FuncType actual = function.getFunctionType();
  if (annotation.kind == IntrinsicKind::Kernel) {
    if (!actual.isVarArg() && actual.hasVoidReturn())
      return mlir::success();
    function.emitError("kernel intrinsic must be non-variadic and return void");
    return mlir::failure();
  }

  mlir::MLIRContext *context = function.getContext();
  mlir::Type s32 = cir::IntType::get(context, 32, true);
  mlir::Type f16 = cir::FP16Type::get(context);
  mlir::Type f32 = cir::SingleType::get(context);
  mlir::Type result = cir::VoidType::get(context);
  llvm::SmallVector<mlir::Type, 4> inputs;
  auto vector = [](mlir::Type element, int64_t rows, int64_t columns) {
    return cir::VectorType::get(element, static_cast<uint64_t>(rows * columns));
  };

  switch (annotation.kind) {
  case IntrinsicKind::Kernel:
    llvm_unreachable("handled above");
  case IntrinsicKind::BlockId:
    result = s32;
    break;
  case IntrinsicKind::Zero:
    result = vector(f32, annotation.arguments[0], annotation.arguments[1]);
    break;
  case IntrinsicKind::Load:
    inputs = {cir::PointerType::get(f16), s32, s32};
    result = vector(f16, annotation.arguments[0], annotation.arguments[1]);
    break;
  case IntrinsicKind::Mma:
    inputs = {vector(f16, annotation.arguments[0], annotation.arguments[2]),
              vector(f16, annotation.arguments[2], annotation.arguments[1]),
              vector(f32, annotation.arguments[0], annotation.arguments[1])};
    result = vector(f32, annotation.arguments[0], annotation.arguments[1]);
    break;
  case IntrinsicKind::Store:
    inputs = {cir::PointerType::get(f32),
              vector(f32, annotation.arguments[0], annotation.arguments[1]),
              s32, s32};
    break;
  }

  cir::FuncType expected = cir::FuncType::get(inputs, result);
  if (actual == expected)
    return mlir::success();
  function.emitError() << "intrinsic '" << getAnnotationName(annotation.kind)
                       << "' requires type " << expected << ", got " << actual;
  return mlir::failure();
}

mlir::LogicalResult validateAnnotations(mlir::ModuleOp module,
                                        llvm::raw_ostream *output) {
  mlir::WalkResult result = module.walk([&](cir::FuncOp function) {
    auto annotation = decodeAnnotation(function);
    if (mlir::failed(annotation))
      return mlir::WalkResult::interrupt();
    if (*annotation && mlir::failed(validateIntrinsic(function, **annotation)))
      return mlir::WalkResult::interrupt();
    if (!*annotation || !output)
      return mlir::WalkResult::advance();

    *output << '@' << function.getSymName() << ' '
            << getAnnotationName((*annotation)->kind);
    if (!(*annotation)->arguments.empty()) {
      *output << '(';
      llvm::ListSeparator separator;
      for (int64_t argument : (*annotation)->arguments)
        *output << separator << argument;
      *output << ')';
    }
    *output << '\n';
    return mlir::WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

} // namespace clang::CIRTile
