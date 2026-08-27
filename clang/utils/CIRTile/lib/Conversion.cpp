//===- Conversion.cpp - CIR to CUDA Tile conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRTile/Conversion.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "clang/CIRTile/Annotation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::CIRTile {
namespace {

class Converter {
public:
  explicit Converter(mlir::ModuleOp module)
      : sourceModule(module), context(module.getContext()),
        symbols(module.getOperation()), builder(context) {}

  mlir::OwningOpRef<mlir::cuda_tile::ModuleOp> convert();

private:
  mlir::Type convertScalarType(mlir::Type type);
  mlir::FailureOr<mlir::Type> convertArgumentType(mlir::Type type,
                                                  cir::FuncOp kernel);
  mlir::LogicalResult convertKernel(cir::FuncOp kernel);
  mlir::LogicalResult convertCall(cir::CallOp call);
  mlir::LogicalResult convertOperation(mlir::Operation &operation);

  mlir::ModuleOp sourceModule;
  mlir::MLIRContext *context;
  mlir::SymbolTable symbols;
  mlir::OpBuilder builder;
  mlir::cuda_tile::ModuleOp targetModule;
  llvm::DenseMap<mlir::Value, mlir::Value> values;
  mlir::cuda_tile::GetTileBlockIdOp blockIds;
};

mlir::Type Converter::convertScalarType(mlir::Type type) {
  if (mlir::isa<cir::FP16Type>(type))
    return builder.getF16Type();
  if (mlir::isa<cir::SingleType>(type))
    return builder.getF32Type();
  if (mlir::isa<cir::DoubleType>(type))
    return builder.getF64Type();
  if (auto integer = mlir::dyn_cast<cir::IntType>(type)) {
    if (!integer.isBitInt() &&
        llvm::is_contained({8u, 16u, 32u, 64u}, integer.getWidth()))
      return builder.getIntegerType(integer.getWidth());
  }
  return {};
}

mlir::FailureOr<mlir::Type> Converter::convertArgumentType(mlir::Type type,
                                                           cir::FuncOp kernel) {
  mlir::Type elementType;
  if (auto pointer = mlir::dyn_cast<cir::PointerType>(type)) {
    if (pointer.getAddrSpace()) {
      kernel.emitError() << "does not support addressed kernel pointer type "
                         << type;
      return mlir::failure();
    }
    elementType = convertScalarType(pointer.getPointee());
    if (elementType)
      elementType = mlir::cuda_tile::PointerType::get(elementType);
  } else {
    elementType = convertScalarType(type);
  }

  if (!elementType) {
    kernel.emitError() << "does not support kernel argument type " << type;
    return mlir::failure();
  }
  return mlir::cuda_tile::TileType::get({}, elementType);
}

mlir::LogicalResult Converter::convertCall(cir::CallOp call) {
  std::optional<llvm::StringRef> calleeName = call.getCallee();
  if (!calleeName) {
    call.emitError("does not support indirect calls");
    return mlir::failure();
  }

  cir::FuncOp callee = symbols.lookup<cir::FuncOp>(*calleeName);
  if (!callee) {
    call.emitError() << "cannot resolve callee @" << *calleeName;
    return mlir::failure();
  }

  auto annotation = decodeAnnotation(callee);
  if (mlir::failed(annotation))
    return mlir::failure();
  if (!*annotation) {
    call.emitError() << "callee @" << *calleeName
                     << " is not a CIR Tile intrinsic";
    return mlir::failure();
  }
  if ((*annotation)->kind != IntrinsicKind::BlockId) {
    call.emitError() << "lowering for annotation '"
                     << getAnnotationName((*annotation)->kind)
                     << "' is not implemented";
    return mlir::failure();
  }
  if (!call.getArgs().empty() || call.getNumResults() != 1) {
    call.emitError(
        "CIR Tile block ID call must have no arguments and one result");
    return mlir::failure();
  }

  if (!blockIds)
    blockIds =
        mlir::cuda_tile::GetTileBlockIdOp::create(builder, call.getLoc());
  values[call.getResult()] =
      blockIds->getResult(static_cast<unsigned>((*annotation)->arguments[0]));
  return mlir::success();
}

mlir::LogicalResult Converter::convertOperation(mlir::Operation &operation) {
  if (auto call = mlir::dyn_cast<cir::CallOp>(operation))
    return convertCall(call);
  if (auto returnOp = mlir::dyn_cast<cir::ReturnOp>(operation)) {
    if (!returnOp.getOperands().empty()) {
      returnOp.emitError("CIR Tile kernels cannot return values");
      return mlir::failure();
    }
    mlir::cuda_tile::ReturnOp::create(builder, returnOp.getLoc());
    return mlir::success();
  }

  operation.emitError("is not supported in CIR Tile kernel lowering");
  return mlir::failure();
}

mlir::LogicalResult Converter::convertKernel(cir::FuncOp kernel) {
  cir::FuncType sourceType = kernel.getFunctionType();
  if (sourceType.isVarArg() || !sourceType.hasVoidReturn()) {
    kernel.emitError("CIR Tile kernel must be non-variadic and return void");
    return mlir::failure();
  }
  if (!kernel.getBody().hasOneBlock()) {
    kernel.emitError("CIR Tile kernel must have one flattened SSA block");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type> inputTypes;
  for (mlir::Type input : sourceType.getInputs()) {
    auto converted = convertArgumentType(input, kernel);
    if (mlir::failed(converted))
      return mlir::failure();
    inputTypes.push_back(*converted);
  }

  builder.setInsertionPointToEnd(&targetModule.getBody().front());
  auto targetType = mlir::FunctionType::get(context, inputTypes, {});
  auto optimizationHints = mlir::cuda_tile::OptimizationHintsAttr::get(
      context, builder.getDictionaryAttr({}));
  auto entry = mlir::cuda_tile::EntryOp::create(
      builder, kernel.getLoc(), kernel.getSymName(), targetType,
      mlir::ArrayAttr(), mlir::ArrayAttr(), optimizationHints);
  mlir::Block *targetBlock = entry.addEntryBlock();

  mlir::Block &sourceBlock = kernel.getBody().front();
  for (auto [sourceArgument, targetArgument] :
       llvm::zip(sourceBlock.getArguments(), targetBlock->getArguments()))
    values[sourceArgument] = targetArgument;

  builder.setInsertionPointToStart(targetBlock);
  blockIds = {};
  for (mlir::Operation &operation : sourceBlock)
    if (mlir::failed(convertOperation(operation)))
      return mlir::failure();

  if (targetBlock->empty() ||
      !mlir::isa<mlir::cuda_tile::ReturnOp>(targetBlock->back())) {
    kernel.emitError("CIR Tile kernel has no return terminator");
    return mlir::failure();
  }
  return mlir::success();
}

mlir::OwningOpRef<mlir::cuda_tile::ModuleOp> Converter::convert() {
  context->getOrLoadDialect<mlir::cuda_tile::CudaTileDialect>();
  llvm::SmallVector<cir::FuncOp> kernels;
  for (cir::FuncOp function : sourceModule.getOps<cir::FuncOp>()) {
    auto annotation = decodeAnnotation(function);
    if (mlir::failed(annotation))
      return {};
    if (*annotation && (*annotation)->kind == IntrinsicKind::Kernel)
      kernels.push_back(function);
  }

  if (kernels.empty()) {
    sourceModule.emitError("contains no cir_tile.v1.kernel function");
    return {};
  }

  targetModule = mlir::cuda_tile::ModuleOp::create(
      builder, sourceModule.getLoc(), "cir_module", "clang-cir");
  mlir::OwningOpRef<mlir::cuda_tile::ModuleOp> result(targetModule);
  for (cir::FuncOp kernel : kernels)
    if (mlir::failed(convertKernel(kernel)))
      return {};
  if (mlir::failed(mlir::verify(targetModule)))
    return {};
  return result;
}

} // namespace

mlir::OwningOpRef<mlir::cuda_tile::ModuleOp>
convertToCudaTile(mlir::ModuleOp module) {
  return Converter(module).convert();
}

} // namespace clang::CIRTile
