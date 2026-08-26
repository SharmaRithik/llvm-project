//===- Annotation.cpp - CIR Tile annotation decoding ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRTile/Annotation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringSwitch.h"

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

    if (*kind == IntrinsicKind::BlockId &&
        (result->arguments[0] < 0 || result->arguments[0] > 2)) {
      function.emitError() << "annotation '" << name
                           << "' requires an axis from 0 to 2";
      return mlir::failure();
    }
  }
  return result;
}

mlir::LogicalResult validateAnnotations(mlir::ModuleOp module,
                                        llvm::raw_ostream *output) {
  mlir::WalkResult result = module.walk([&](cir::FuncOp function) {
    auto annotation = decodeAnnotation(function);
    if (mlir::failed(annotation))
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
