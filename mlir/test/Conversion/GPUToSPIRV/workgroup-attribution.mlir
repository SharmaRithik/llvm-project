// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s

// Workgroup attributions on a gpu.func are not converted. They used to trip an
// assertion inside the signature conversion; now they are a diagnostic.

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // expected-error @below {{gpu.func workgroup and private memory attributions are not supported when converting to SPIR-V; use a memref.alloc in the Workgroup storage class instead}}
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    gpu.func @with_attribution(%arg0 : memref<64xf32, #spirv.storage_class<StorageBuffer>>)
        workgroup(%sh : memref<64xf32, #spirv.storage_class<Workgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>} {
      %t = gpu.thread_id x
      %x = memref.load %arg0[%t] : memref<64xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %x, %sh[%t] : memref<64xf32, #spirv.storage_class<Workgroup>>
      gpu.return
    }
  }
}
