// RUN: mlir-translate --mlir-to-wgsl %s -o %t.wgsl
// RUN: FileCheck %s --input-file=%t.wgsl

// CHECK: select(u32(0u), u32(4294967295u), true)

module attributes {gpu.container_module} { gpu.module @kernels { gpu.func @main(%dst: memref<?xi32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {

%c0 = arith.constant 0 : index
%t = arith.constant true
%x = arith.index_cast %t : i1 to index
%v = arith.index_cast %x : index to i32
memref.store %v, %dst[%c0] : memref<?xi32>

gpu.return
}}}
