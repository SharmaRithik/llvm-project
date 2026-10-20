// RUN: mlir-wgsl-compile %s --nvidia-native --storage-alignment=4 --max-shared-memory=49152 | FileCheck %s
// RUN: mlir-wgsl-compile %s --nvidia-native --storage-alignment=4 --max-shared-memory=49152 --batch-count=3 | FileCheck %s
//
// A single N workgroup has a constant-zero N coordinate even when batches
// share the flattened dispatch grid. Upstream affine delinearization exposes
// this fact, so copying eight valid rows into a sixteen-row shared tile does
// not retain a dynamic serial copy loop. The reduction itself remains counted.
// CHECK: var<storage, read_write> arg1: array<vec4<f32>>;
// CHECK: for (var [[K:iv[0-9]+]]: u32 = 0u; bitcast<i32>([[K]]) < bitcast<i32>(2048u); [[K]] += 32u)
// CHECK-NOT: for (
// CHECK: workgroupBarrier();
// CHECK: fma(
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<2048x2048xf16>, %b: tensor<8x2048xf32>) -> tensor<2048x8xf32> {
  %tmp = tensor.empty() : tensor<8x2048xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<8x2048xf32>) outs(%tmp : tensor<8x2048xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<8x2048xf16>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2048x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048x8xf32>) -> tensor<2048x8xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %half : tensor<2048x2048xf16>, tensor<8x2048xf16>) outs(%init : tensor<2048x8xf32>) -> tensor<2048x8xf32>
  return %product : tensor<2048x8xf32>
}
