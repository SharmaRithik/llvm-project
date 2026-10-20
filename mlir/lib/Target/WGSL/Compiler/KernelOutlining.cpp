//===- KernelOutlining.cpp - Outline WGSL kernels -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelOutlining.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

wgsl::KernelLaunch wgsl::createKernelLaunch(
    IRRewriter &builder, ModuleOp module, Location loc, TypeRange bindingTypes,
    TypeRange workgroupTypes, ArrayRef<int32_t> blockSize) {
  assert(blockSize.size() == 3 && "expected three workgroup dimensions");
  SmallVector<Type> hostTypes(bindingTypes);
  hostTypes.append(3, builder.getIndexType());
  builder.setInsertionPointToEnd(module.getBody());
  auto host = func::FuncOp::create(builder, loc, "__wgsl_launch",
                                  builder.getFunctionType(hostTypes, {}));
  host.addEntryBlock();
  builder.setInsertionPointToStart(&host.getBody().front());
  SmallVector<Value> block;
  for (int32_t size : blockSize)
    block.push_back(arith::ConstantIndexOp::create(builder, loc, size));
  unsigned numBindings = bindingTypes.size();
  // Dynamic grid sizes keep range analysis from folding block IDs.
  auto launch = gpu::LaunchOp::create(
      builder, loc, host.getArgument(numBindings),
      host.getArgument(numBindings + 1), host.getArgument(numBindings + 2),
      block[0], block[1], block[2], Value(), Type(), ValueRange{}, Value(),
      workgroupTypes);
  launch.setModuleAttr(FlatSymbolRefAttr::get(builder.getContext(), "kernels"));
  launch.setFunctionAttr(FlatSymbolRefAttr::get(builder.getContext(), "main"));
  func::ReturnOp::create(builder, loc);
  builder.setInsertionPointToStart(&launch.getBody().front());
  return {host, launch, numBindings};
}

FailureOr<gpu::GPUFuncOp>
wgsl::outlineKernelLaunch(ModuleOp module, KernelLaunch kernel) {
  PassManager pm(module.getContext());
  pm.addPass(createGpuKernelOutliningPass());
  if (failed(pm.run(module)))
    return failure();

  SmallVector<gpu::LaunchFuncOp> launches;
  kernel.host.walk([&](gpu::LaunchFuncOp op) { launches.push_back(op); });
  if (launches.size() != 1)
    return kernel.host.emitError("expected one outlined WGSL kernel"), failure();
  gpu::LaunchFuncOp launch = launches.front();
  auto gpuModule = module.lookupSymbol<gpu::GPUModuleOp>(
      launch.getKernelModuleName().getValue());
  auto function = gpuModule
                      ? gpuModule.lookupSymbol<gpu::GPUFuncOp>(
                            launch.getKernelName().getValue())
                      : gpu::GPUFuncOp();
  if (!function || launch.getKernelOperands().size() != function.getNumArguments())
    return launch.emitError("could not resolve outlined WGSL bindings"), failure();

  Builder builder(module.getContext());
  for (auto [index, capture] : llvm::enumerate(launch.getKernelOperands())) {
    auto binding = dyn_cast<BlockArgument>(capture);
    if (!binding || binding.getOwner() != &kernel.host.getBody().front() ||
        binding.getArgNumber() >= kernel.numBindings)
      return launch.emitError("outlined WGSL kernel captured a non-binding value"),
             failure();
    unsigned bindingNumber = binding.getArgNumber();
    if (auto attrs = kernel.host.getArgAttrDict(bindingNumber))
      for (NamedAttribute attr : attrs)
        function.setArgAttr(index, attr.getName(), attr.getValue());
    function.setArgAttr(index, "wgsl.binding",
                        builder.getI64IntegerAttr(bindingNumber));
    function.setArgAttr(index, "wgsl.name",
                        builder.getStringAttr(("arg" + Twine(bindingNumber)).str()));
  }
  kernel.host.erase();
  return function;
}
