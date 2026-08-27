//===- Conversion.cpp - CIR to CUDA Tile conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIRTile/Conversion.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "clang/CIRTile/Annotation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

namespace clang::CIRTile {
namespace {

class Converter {
public:
  explicit Converter(mlir::ModuleOp module)
      : sourceModule(module), context(module.getContext()),
        symbols(module.getOperation()), builder(context) {}

  mlir::OwningOpRef<mlir::cuda_tile::ModuleOp> convert();

private:
  using AnnotatedCallee = std::pair<cir::FuncOp, IntrinsicAnnotation>;
  using ViewKey = std::pair<mlir::Value, mlir::Operation *>;

  mlir::Type convertScalarType(mlir::Type type);
  mlir::FailureOr<mlir::Type> convertArgumentType(mlir::Type type,
                                                  cir::FuncOp kernel);
  mlir::FailureOr<AnnotatedCallee> getAnnotatedCallee(cir::CallOp call);
  mlir::FailureOr<mlir::Value> lookupValue(mlir::Value source,
                                           mlir::Operation &operation);
  mlir::FailureOr<mlir::Value>
  getOrCreatePartitionView(cir::CallOp call, cir::FuncOp callee,
                           const IntrinsicAnnotation &annotation);
  mlir::LogicalResult materializeViews(mlir::Block &sourceBlock);
  mlir::LogicalResult convertConstant(cir::ConstantOp constant);
  mlir::LogicalResult convertZero(cir::CallOp call,
                                  const IntrinsicAnnotation &annotation);
  mlir::LogicalResult convertLoad(cir::CallOp call, cir::FuncOp callee,
                                  const IntrinsicAnnotation &annotation);
  mlir::LogicalResult convertMma(cir::CallOp call,
                                 const IntrinsicAnnotation &annotation);
  mlir::LogicalResult convertStore(cir::CallOp call, cir::FuncOp callee,
                                   const IntrinsicAnnotation &annotation);
  mlir::LogicalResult convertKernel(cir::FuncOp kernel);
  mlir::LogicalResult convertCall(cir::CallOp call);
  mlir::LogicalResult convertOperation(mlir::Operation &operation);

  mlir::ModuleOp sourceModule;
  mlir::MLIRContext *context;
  mlir::SymbolTable symbols;
  mlir::OpBuilder builder;
  mlir::cuda_tile::ModuleOp targetModule;
  llvm::DenseMap<mlir::Value, mlir::Value> values;
  llvm::DenseMap<ViewKey, mlir::Value> partitionViews;
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

mlir::FailureOr<Converter::AnnotatedCallee>
Converter::getAnnotatedCallee(cir::CallOp call) {
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
  return AnnotatedCallee(callee, **annotation);
}

mlir::FailureOr<mlir::Value>
Converter::lookupValue(mlir::Value source, mlir::Operation &operation) {
  auto found = values.find(source);
  if (found == values.end()) {
    operation.emitError("uses a value that has not been converted");
    return mlir::failure();
  }
  return found->second;
}

mlir::FailureOr<mlir::Value>
Converter::getOrCreatePartitionView(cir::CallOp call, cir::FuncOp callee,
                                    const IntrinsicAnnotation &annotation) {
  if (call.getArgs().empty()) {
    call.emitError("CIR Tile view call requires a base pointer");
    return mlir::failure();
  }

  ViewKey key(call.getArgs().front(), callee.getOperation());
  if (auto found = partitionViews.find(key); found != partitionViews.end())
    return found->second;

  auto base = lookupValue(call.getArgs().front(), *call);
  if (mlir::failed(base))
    return mlir::failure();
  auto baseTile = mlir::dyn_cast<mlir::cuda_tile::TileType>((*base).getType());
  auto pointer = baseTile ? mlir::dyn_cast<mlir::cuda_tile::PointerType>(
                                baseTile.getElementType())
                          : mlir::cuda_tile::PointerType();
  if (!pointer) {
    call.emitError("CIR Tile view base must be a scalar pointer tile");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 2> tensorShape{annotation.arguments[2],
                                            annotation.arguments[3]};
  llvm::SmallVector<int64_t, 2> tensorStrides{annotation.arguments[3], 1};
  auto tensorType = mlir::cuda_tile::TensorViewType::get(
      context, pointer.getPointeeType(), tensorShape, tensorStrides);
  auto tensor = mlir::cuda_tile::MakeTensorViewOp::create(
      builder, call.getLoc(), tensorType, *base, mlir::ValueRange(),
      mlir::ValueRange());

  llvm::SmallVector<int32_t, 2> tileShape{
      static_cast<int32_t>(annotation.arguments[0]),
      static_cast<int32_t>(annotation.arguments[1])};
  auto partitionType = mlir::cuda_tile::PartitionViewType::get(
      context, builder.getDenseI32ArrayAttr(tileShape), tensorType, {0, 1},
      mlir::cuda_tile::PaddingValueAttr());
  auto partition = mlir::cuda_tile::MakePartitionViewOp::create(
      builder, call.getLoc(), partitionType, tensor.getResult());
  partitionViews[key] = partition.getResult();
  return partition.getResult();
}

mlir::LogicalResult Converter::materializeViews(mlir::Block &sourceBlock) {
  for (cir::CallOp call : sourceBlock.getOps<cir::CallOp>()) {
    auto annotatedCallee = getAnnotatedCallee(call);
    if (mlir::failed(annotatedCallee))
      return mlir::failure();
    if (annotatedCallee->second.kind != IntrinsicKind::Load &&
        annotatedCallee->second.kind != IntrinsicKind::Store)
      continue;
    if (mlir::failed(getOrCreatePartitionView(call, annotatedCallee->first,
                                              annotatedCallee->second)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult Converter::convertConstant(cir::ConstantOp constant) {
  auto integerType = mlir::dyn_cast<cir::IntType>(constant.getRes().getType());
  if (!integerType || integerType.isBitInt() ||
      !constant.getValueAttr<cir::IntAttr>()) {
    constant.emitError("only fixed-width integer constants are supported");
    return mlir::failure();
  }
  mlir::Type elementType = convertScalarType(integerType);
  if (!elementType) {
    constant.emitError() << "does not support integer constant type "
                         << integerType;
    return mlir::failure();
  }

  auto tileType = mlir::cuda_tile::TileType::get({}, elementType);
  auto value =
      mlir::DenseIntElementsAttr::get(tileType, constant.getIntValue());
  auto target = mlir::cuda_tile::ConstantOp::create(builder, constant.getLoc(),
                                                    tileType, value);
  values[constant.getResult()] = target.getResult();
  return mlir::success();
}

mlir::LogicalResult
Converter::convertZero(cir::CallOp call,
                       const IntrinsicAnnotation &annotation) {
  if (!call.getArgs().empty() || call.getNumResults() != 1) {
    call.emitError("CIR Tile zero call must have no arguments and one result");
    return mlir::failure();
  }

  auto tileType = mlir::cuda_tile::TileType::get(
      {annotation.arguments[0], annotation.arguments[1]}, builder.getF32Type());
  auto value = mlir::DenseFPElementsAttr::get(
      tileType, builder.getF32FloatAttr(0.0).getValue());
  auto zero = mlir::cuda_tile::ConstantOp::create(builder, call.getLoc(),
                                                  tileType, value);
  values[call.getResult()] = zero.getResult();
  return mlir::success();
}

mlir::LogicalResult
Converter::convertLoad(cir::CallOp call, cir::FuncOp callee,
                       const IntrinsicAnnotation &annotation) {
  if (call.getArgs().size() != 3 || call.getNumResults() != 1) {
    call.emitError(
        "CIR Tile load call must have three arguments and one result");
    return mlir::failure();
  }
  auto view = getOrCreatePartitionView(call, callee, annotation);
  if (mlir::failed(view))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 2> indices;
  for (mlir::Value source : call.getArgs().drop_front()) {
    auto target = lookupValue(source, *call);
    if (mlir::failed(target))
      return mlir::failure();
    indices.push_back(*target);
  }
  auto viewType =
      mlir::cast<mlir::cuda_tile::PartitionViewType>((*view).getType());
  auto load = mlir::cuda_tile::LoadViewTkoOp::create(
      builder, call.getLoc(), viewType.getViewTileType(),
      mlir::cuda_tile::TokenType::get(context),
      mlir::cuda_tile::MemoryOrderingSemantics::WEAK,
      mlir::cuda_tile::MemoryScopeAttr(), *view, indices, mlir::Value(),
      mlir::cuda_tile::OptimizationHintsAttr());
  values[call.getResult()] = load.getTile();
  return mlir::success();
}

mlir::LogicalResult
Converter::convertMma(cir::CallOp call, const IntrinsicAnnotation &annotation) {
  if (call.getArgs().size() != 3 || call.getNumResults() != 1) {
    call.emitError(
        "CIR Tile MMA call must have three arguments and one result");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 3> operands;
  for (mlir::Value source : call.getArgs()) {
    auto target = lookupValue(source, *call);
    if (mlir::failed(target))
      return mlir::failure();
    operands.push_back(*target);
  }
  auto resultType = mlir::cuda_tile::TileType::get(
      {annotation.arguments[0], annotation.arguments[1]}, builder.getF32Type());
  auto mma = mlir::cuda_tile::MmaFOp::create(builder, call.getLoc(), resultType,
                                             operands[0], operands[1],
                                             operands[2], false);
  values[call.getResult()] = mma.getResult();
  return mlir::success();
}

mlir::LogicalResult
Converter::convertStore(cir::CallOp call, cir::FuncOp callee,
                        const IntrinsicAnnotation &annotation) {
  if (call.getArgs().size() != 4 || call.getNumResults() != 0) {
    call.emitError(
        "CIR Tile store call must have four arguments and no results");
    return mlir::failure();
  }
  auto view = getOrCreatePartitionView(call, callee, annotation);
  if (mlir::failed(view))
    return mlir::failure();
  auto tile = lookupValue(call.getArgs()[1], *call);
  if (mlir::failed(tile))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 2> indices;
  for (mlir::Value source : call.getArgs().drop_front(2)) {
    auto target = lookupValue(source, *call);
    if (mlir::failed(target))
      return mlir::failure();
    indices.push_back(*target);
  }
  mlir::cuda_tile::StoreViewTkoOp::create(
      builder, call.getLoc(), mlir::cuda_tile::TokenType::get(context),
      mlir::cuda_tile::MemoryOrderingSemantics::WEAK,
      mlir::cuda_tile::MemoryScopeAttr(), *tile, *view, indices, mlir::Value(),
      mlir::cuda_tile::OptimizationHintsAttr());
  return mlir::success();
}

mlir::LogicalResult Converter::convertCall(cir::CallOp call) {
  auto annotatedCallee = getAnnotatedCallee(call);
  if (mlir::failed(annotatedCallee))
    return mlir::failure();
  cir::FuncOp callee = annotatedCallee->first;
  const IntrinsicAnnotation &annotation = annotatedCallee->second;

  if (annotation.kind == IntrinsicKind::Zero)
    return convertZero(call, annotation);
  if (annotation.kind == IntrinsicKind::Load)
    return convertLoad(call, callee, annotation);
  if (annotation.kind == IntrinsicKind::Mma)
    return convertMma(call, annotation);
  if (annotation.kind == IntrinsicKind::Store)
    return convertStore(call, callee, annotation);
  if (annotation.kind != IntrinsicKind::BlockId) {
    call.emitError() << "lowering for annotation '"
                     << getAnnotationName(annotation.kind)
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
      blockIds->getResult(static_cast<unsigned>(annotation.arguments[0]));
  return mlir::success();
}

mlir::LogicalResult Converter::convertOperation(mlir::Operation &operation) {
  if (auto constant = mlir::dyn_cast<cir::ConstantOp>(operation))
    return convertConstant(constant);
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
  values.clear();
  partitionViews.clear();
  blockIds = {};
  for (auto [sourceArgument, targetArgument] :
       llvm::zip(sourceBlock.getArguments(), targetBlock->getArguments()))
    values[sourceArgument] = targetArgument;

  builder.setInsertionPointToStart(targetBlock);
  if (mlir::failed(materializeViews(sourceBlock)))
    return mlir::failure();
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
    if (*annotation && mlir::failed(validateIntrinsic(function, **annotation)))
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
