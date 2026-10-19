// RUN: mlir-translate --mlir-to-wgsl %s -o %t.wgsl
// RUN: FileCheck %s --input-file=%t.wgsl

// CHECK: : vec2<bool> = {{.*}} != {{.*}};

module attributes {gpu.container_module} { gpu.module @kernels { gpu.func @main(%dst: memref<?xvector<2xi32>>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {

%c0 = arith.constant 0 : index
%a = arith.constant dense<[true, false]> : vector<2xi1>
%b = arith.constant dense<[false, true]> : vector<2xi1>
%x = arith.xori %a, %b : vector<2xi1>
%v = arith.extui %x : vector<2xi1> to vector<2xi32>
memref.store %v, %dst[%c0] : memref<?xvector<2xi32>>

gpu.return
}}}
