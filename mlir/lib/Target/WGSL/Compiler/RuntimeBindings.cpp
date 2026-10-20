//===- RuntimeBindings.cpp - WGSL runtime bindings ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RuntimeBindings.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
Value storageRoot(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value;
}
}

LogicalResult wgsl::lowerPackedStorageBindings(ModuleOp module) {
  IRRewriter b(module.getContext());
  SmallVector<gpu::GPUFuncOp> kernels;
  module.walk([&](gpu::GPUFuncOp op) { kernels.push_back(op); });
  for (auto kernel : kernels) {
    SmallVector<std::pair<BlockArgument, MemRefType>> bindings;
    DenseMap<Value, Value> packed;
    for (unsigned i = 0; i < kernel.getNumArguments(); ++i) {
      auto width = kernel.getArgAttrOfType<IntegerAttr>(i, "wgsl.pack_width");
      if (!width)
        continue;
      auto arg = kernel.getArgument(i);
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (width.getInt() != 4 || !type || type.getRank() != 1 ||
          !type.getLayout().isIdentity() || !type.getElementType().isIntOrFloat())
        return kernel.emitError("invalid packed WGSL storage binding");
      int64_t size = type.getDimSize(0);
      if (!ShapedType::isDynamic(size)) {
        if (size % 4 != 0)
          return kernel.emitError("packed storage extent must be divisible by four");
        size /= 4;
      }
      auto packedType = MemRefType::get(
          {size}, VectorType::get({4}, type.getElementType()),
          type.getLayout(), type.getMemorySpace());
      bindings.emplace_back(arg, packedType);
      if (arg.use_empty())
        continue;
      b.setInsertionPointToStart(&kernel.getBody().front());
      auto bridge = UnrealizedConversionCastOp::create(
          b, kernel.getLoc(), TypeRange{packedType}, ValueRange{arg});
      packed[arg] = bridge.getResult(0);
    }
    if (bindings.empty())
      continue;
    SmallVector<Operation *> loads;
    kernel.walk([&](Operation *op) {
      if (isa<memref::LoadOp, vector::LoadOp, vector::StoreOp, memref::StoreOp>(op))
        loads.push_back(op);
    });
    for (Operation *op : loads) {
      Value source;
      ValueRange indices;
      VectorType vectorType;
      Value stored;
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        source = load.getMemRef();
        indices = load.getIndices();
      } else if (auto load = dyn_cast<vector::LoadOp>(op)) {
        source = load.getBase();
        indices = load.getIndices();
        vectorType = load.getVectorType();
      } else if (auto store = dyn_cast<vector::StoreOp>(op)) {
        source = store.getBase();
        indices = store.getIndices();
        stored = store.getValueToStore();
        vectorType = cast<VectorType>(stored.getType());
      } else {
        auto scalarStore = cast<memref::StoreOp>(op);
        source = scalarStore.getMemRef();
        indices = scalarStore.getIndices();
        stored = scalarStore.getValueToStore();
      }
      Value physical = packed.lookup(storageRoot(source));
      if (!physical)
        continue;
      auto sourceType = cast<MemRefType>(source.getType());
      auto storageType = cast<MemRefType>(storageRoot(source).getType());
      if (sourceType.getElementType() != storageType.getElementType())
        return op->emitError("packed storage views must preserve element type");
      if (stored && !vectorType)
        return op->emitError("scalar stores to packed WGSL storage would race");
      if (vectorType && (vectorType.getRank() != 1 || vectorType.getDimSize(0) != 4))
        return op->emitError("packed storage requires scalar or vec4 loads");
      b.setInsertionPoint(op);
      Location loc = op->getLoc();
      auto metadata = memref::ExtractStridedMetadataOp::create(b, loc, source);
      unsigned bits = sourceType.getElementTypeBitWidth();
      auto [info, relative] = memref::getLinearizedMemRefOffsetAndSize(
          b, loc, bits, bits, metadata.getConstifiedMixedOffset(),
          metadata.getConstifiedMixedSizes(), metadata.getConstifiedMixedStrides(),
          getAsOpFoldResult(indices));
      auto materialize = [&](OpFoldResult value) {
        return getValueOrCreateConstantIndexOp(b, loc, value);
      };
      Value offset = arith::AddIOp::create(
          b, loc, materialize(info.linearizedOffset), materialize(relative));
      Value four = arith::ConstantIndexOp::create(b, loc, 4);
      // Scalar reads pick a lane. Scalar writes would race on adjacent lanes.
      Value index = arith::DivUIOp::create(b, loc, offset, four);
      if (stored) {
        memref::StoreOp::create(b, loc, stored, physical, index);
        b.eraseOp(op);
        continue;
      }
      Value value = memref::LoadOp::create(b, loc, physical, index);
      if (!vectorType) {
        Value lane = arith::RemUIOp::create(b, loc, offset, four);
        value = vector::ExtractOp::create(
            b, loc, value, ArrayRef<OpFoldResult>{lane});
      }
      b.replaceOp(op, value);
    }
    // The temporary cast is an ABI bridge and never reaches the shader.
    PassManager pm(module.getContext());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(pm.run(module)))
      return failure();
    SmallVector<Type> inputTypes(kernel.getArgumentTypes());
    for (auto [argument, packedType] : bindings) {
      auto emitUnloweredAccess = [&]() -> LogicalResult {
        auto diagnostic = kernel.emitError("unlowered access to packed WGSL storage binding ");
        diagnostic << argument.getArgNumber();
        for (Operation *user : argument.getUsers())
          diagnostic.attachNote(user->getLoc()) << "remaining use: " << *user;
        return failure();
      };
      // Canonicalization may erase the bridge, so recover it from the argument.
      UnrealizedConversionCastOp bridge;
      if (!argument.use_empty()) {
        if (!argument.hasOneUse())
          return emitUnloweredAccess();
        bridge = dyn_cast<UnrealizedConversionCastOp>(
            *argument.getUsers().begin());
        if (!bridge || bridge.getNumResults() != 1 ||
            bridge.getResult(0).getType() != packedType)
          return emitUnloweredAccess();
      }
      argument.setType(packedType);
      inputTypes[argument.getArgNumber()] = packedType;
      if (bridge)
        b.replaceOp(bridge, argument);
      kernel.removeArgAttr(argument.getArgNumber(), "wgsl.pack_width");
    }
    kernel.setFunctionType(b.getFunctionType(inputTypes, kernel.getResultTypes()));
  }
  return success();
}
