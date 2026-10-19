// RUN: mlir-translate --mlir-to-wgsl %s -o %t.wgsl
// RUN: FileCheck %s --input-file=%t.wgsl

// CHECK: let [[OLD:v[0-9]+]]: f32 = [[A:a[0-9]+]].x;
// CHECK: [[A]] = vec2<f32>(9.0f);
// CHECK: {{a[0-9]+}} = [[OLD]];

module attributes {gpu.container_module} { gpu.module @kernels { gpu.func @main(%dst: memref<?xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%v0 = arith.constant dense<[1.0, 2.0]> : vector<2xf32>
%v1 = arith.constant dense<9.0> : vector<2xf32>
%f0 = arith.constant 0.0 : f32
%r:2 = scf.for unsigned %i = %c0 to %c1 step %c1 iter_args(%a = %v0, %b = %f0) -> (vector<2xf32>, f32) {
  %old = vector.extract %a[0] : f32 from vector<2xf32>
  scf.yield %v1, %old : vector<2xf32>, f32
}
memref.store %r#1, %dst[%c0] : memref<?xf32>

gpu.return
}}}
