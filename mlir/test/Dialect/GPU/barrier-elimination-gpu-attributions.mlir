// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

module attributes {gpu.container_module, transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %kernels = transform.structured.match ops{["gpu.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %kernels {
      transform.apply_patterns.gpu.eliminate_barriers
    } : !transform.any_op
    transform.yield
  }

  gpu.module @kernels {
    // Attributions are block arguments past the signature and may alias.
    // CHECK-LABEL: gpu.func @workgroup_attribution
    // CHECK: memref.load
    // CHECK: gpu.barrier
    // CHECK: memref.store
    gpu.func @workgroup_attribution(%out: memref<1xf32> {llvm.noalias})
        workgroup(%shared: memref<1xf32, #gpu.address_space<workgroup>>) kernel {
      %c0 = arith.constant 0 : index
      %v = memref.load %shared[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      memref.store %v, %out[%c0] : memref<1xf32>
      gpu.return
    }

    // CHECK-LABEL: gpu.func @private_attribution
    // CHECK: memref.load
    // CHECK: gpu.barrier
    // CHECK: memref.store
    gpu.func @private_attribution(%out: memref<1xf32> {llvm.noalias})
        private(%scratch: memref<1xf32, #gpu.address_space<private>>) kernel {
      %c0 = arith.constant 0 : index
      %v = memref.load %scratch[%c0] : memref<1xf32, #gpu.address_space<private>>
      gpu.barrier
      memref.store %v, %out[%c0] : memref<1xf32>
      gpu.return
    }
  }
}
