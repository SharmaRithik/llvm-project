// RUN: mlir-wgsl-compile %s --opt-level=0 --dump-prefix=%t | FileCheck %s
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=known_grid_size --implicit-check-not=gpu.launch --implicit-check-not=func.func < %t.scheduled.mlir
// RUN: FileCheck %s --check-prefix=IDS < %t.lowered.mlir
// RUN: mlir-wgsl-compile %s --opt-level=0 --merged-inputs | FileCheck %s --check-prefix=MERGED
// Binding numbers and names survive outlining. The grid stays dynamic.
// CHECK-DAG: @binding(0) var<storage, read_write> arg0: array<f16>;
// CHECK-DAG: @binding(1) var<storage, read_write> arg1: array<f16>;
// CHECK-DAG: @binding(2) var<storage, read_write> arg2: array<f32>;
// CHECK-DAG: @binding(3) var<uniform> arg3: array<vec4<u32>, 4>;
// CHECK: @compute @workgroup_size(64, 1, 1)
// CHECK: fn main(
// IR: gpu.module @kernels
// IR: gpu.func @main(%{{.*}}: memref<16xi32> {wgsl.address_space = "uniform", wgsl.binding = 3 : i64, wgsl.name = "arg3"}
// IR-SAME: workgroup(
// IR-SAME: known_block_size = array<i32: 64, 1, 1>
// IDS: gpu.block_id x
// IDS: gpu.grid_dim x
// MERGED-DAG: @binding(0) var<storage, read_write> arg0: array<f16>;
// MERGED-DAG: @binding(1) var<storage, read_write> arg1: array<f32>;
// MERGED-DAG: @binding(2) var<uniform> arg2: array<vec4<u32>, 4>;
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}
