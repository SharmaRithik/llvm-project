// RUN: not mlir-translate --mlir-to-wgsl %s 2>&1 | FileCheck %s
// CHECK: error: {{.*}}i8

module attributes {gpu.container_module} { gpu.module @kernels { gpu.func @main(%dst: memref<?xi32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {

%c0 = arith.constant 0 : index
%neg = arith.constant -1 : i8
%zero = arith.constant 0 : i8
%p = arith.cmpi slt, %neg, %zero : i8
%v = arith.extui %p : i1 to i32
memref.store %v, %dst[%c0] : memref<?xi32>

gpu.return
}}}
