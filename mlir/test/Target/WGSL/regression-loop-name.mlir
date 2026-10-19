// RUN: mlir-translate --mlir-to-wgsl %s | FileCheck %s
// A loop counter must not shadow the i32 type used by signed comparisons.
// CHECK: for (var iv32: u32 = 0u; bitcast<i32>(iv32)
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @main(%out: memref<?xindex>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %tid = gpu.thread_id x
      %v0 = arith.addi %tid, %c1 : index
      %v1 = arith.addi %v0, %c1 : index
      %v2 = arith.addi %v1, %c1 : index
      %v3 = arith.addi %v2, %c1 : index
      %v4 = arith.addi %v3, %c1 : index
      %v5 = arith.addi %v4, %c1 : index
      %v6 = arith.addi %v5, %c1 : index
      %v7 = arith.addi %v6, %c1 : index
      %v8 = arith.addi %v7, %c1 : index
      %v9 = arith.addi %v8, %c1 : index
      %v10 = arith.addi %v9, %c1 : index
      %v11 = arith.addi %v10, %c1 : index
      %v12 = arith.addi %v11, %c1 : index
      %v13 = arith.addi %v12, %c1 : index
      %v14 = arith.addi %v13, %c1 : index
      %v15 = arith.addi %v14, %c1 : index
      %v16 = arith.addi %v15, %c1 : index
      %v17 = arith.addi %v16, %c1 : index
      %v18 = arith.addi %v17, %c1 : index
      %v19 = arith.addi %v18, %c1 : index
      %v20 = arith.addi %v19, %c1 : index
      %v21 = arith.addi %v20, %c1 : index
      %v22 = arith.addi %v21, %c1 : index
      %v23 = arith.addi %v22, %c1 : index
      %v24 = arith.addi %v23, %c1 : index
      %v25 = arith.addi %v24, %c1 : index
      %v26 = arith.addi %v25, %c1 : index
      %v27 = arith.addi %v26, %c1 : index
      %v28 = arith.addi %v27, %c1 : index
      %v29 = arith.addi %v28, %c1 : index
      %v30 = arith.addi %v29, %c1 : index
      %v31 = arith.addi %v30, %c1 : index
      scf.for %i = %c0 to %v31 step %c1 {
        memref.store %i, %out[%tid] : memref<?xindex>
      }
      gpu.return
    }
  }
}
