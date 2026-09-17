// RUN: mlir-opt -split-input-file -map-memref-spirv-storage-class='client-api=webgpu' -verify-diagnostics %s -o - | FileCheck %s

// WebGPU mappings: Vulkan numbering restricted to the storage classes WebGPU
// has, plus the gpu dialect address space attribute.

// CHECK-LABEL: func @numeric_spaces
// CHECK-SAME: memref<4xf32, #spirv.storage_class<StorageBuffer>>
// CHECK-SAME: memref<4xf32, #spirv.storage_class<Workgroup>>
// CHECK-SAME: memref<4xf32, #spirv.storage_class<Uniform>>
// CHECK-SAME: memref<4xf32, #spirv.storage_class<Function>>
func.func @numeric_spaces(%a: memref<4xf32>, %b: memref<4xf32, 3>, %c: memref<4xf32, 4>, %d: memref<4xf32, 6>) {
  return
}

// CHECK-LABEL: func @gpu_address_spaces
// CHECK-SAME: memref<4xf32, #spirv.storage_class<StorageBuffer>>
// CHECK-SAME: memref<4xf32, #spirv.storage_class<Workgroup>>
// CHECK-SAME: memref<4xf32, #spirv.storage_class<Function>>
func.func @gpu_address_spaces(%a: memref<4xf32, #gpu.address_space<global>>,
                              %b: memref<4xf32, #gpu.address_space<workgroup>>,
                              %c: memref<4xf32, #gpu.address_space<private>>) {
  return
}

// CHECK-LABEL: func @workgroup_alloc
// CHECK: memref.alloc() : memref<64xf32, #spirv.storage_class<Workgroup>>
func.func @workgroup_alloc() {
  %0 = memref.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// PushConstant (Vulkan space 7) has no WebGPU counterpart.

// expected-error @+1 {{failed to legalize memory space}}
func.func @push_constant(%a: memref<4xf32, 7>) {
  return
}
