// RUN: mlir-translate --mlir-to-wgsl %s -o %t.wgsl
// RUN: FileCheck %s --input-file=%t.wgsl

// CHECK: local_id.x * 8u

module attributes {gpu.container_module} { gpu.module @kernels { gpu.func @main(%src: memref<4x8xf32>, %dst: memref<?xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {

%c0 = arith.constant 0 : index
%tid = gpu.thread_id x
%view = memref.cast %src : memref<4x8xf32> to memref<?x?xf32>
%v = memref.load %view[%tid, %c0] : memref<?x?xf32>
memref.store %v, %dst[%c0] : memref<?xf32>

gpu.return
}}}
