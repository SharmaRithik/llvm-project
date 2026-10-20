// RUN: mlir-wgsl-compile %s --nvidia-native --storage-alignment=4 --dump-prefix=%t | FileCheck %s --implicit-check-not=storageBarrier
// RUN: FileCheck %s --check-prefix=TILED < %t.scheduled.mlir
// RUN: FileCheck %s --check-prefix=MAPPED --implicit-check-not=scf.forall < %t.lowered.mlir
// RUN: mlir-wgsl-compile %s --mma --nvidia-native --storage-alignment=4 | FileCheck %s --check-prefix=MMA --implicit-check-not=storageBarrier
// CHECK: var<storage, read_write> arg0: array<vec4<f16>>;
// CHECK-NOT: subgroup_matrix
// CHECK: @compute @workgroup_size(64, 1, 1)
// CHECK: fma(
// MMA: subgroup_matrix_result<f32, 16, 16>
// MMA: subgroupMatrixMultiplyAccumulate
// TILED: scf.forall
// TILED: memref.subview
// TILED: mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]
// MAPPED: gpu.thread_id x
// MAPPED: vector.fma
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<512x512xf16>, %b: tensor<256x512xf32>) -> tensor<512x256xf32> {
  %tmp = tensor.empty() : tensor<256x512xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<256x512xf32>) outs(%tmp : tensor<256x512xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<256x512xf16>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<512x256xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<512x256xf32>) -> tensor<512x256xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %half : tensor<512x512xf16>, tensor<256x512xf16>) outs(%init : tensor<512x256xf32>) -> tensor<512x256xf32>
  return %product : tensor<512x256xf32>
}
