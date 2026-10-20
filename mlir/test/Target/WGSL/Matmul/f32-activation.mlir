// RUN: mlir-wgsl-compile %s --mma | FileCheck %s
// CHECK: enable chromium_experimental_subgroup_matrix;
// CHECK: var<storage, read_write> arg1: array<f32>;
// CHECK: f16(
// CHECK: subgroupMatrixMultiplyAccumulate
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<32x32xf16>, %b: tensor<17x32xf32>) -> tensor<32x17xf32> {
  %tmp = tensor.empty() : tensor<17x32xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<17x32xf32>) outs(%tmp : tensor<17x32xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<17x32xf16>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x17xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x17xf32>) -> tensor<32x17xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %half : tensor<32x32xf16>, tensor<17x32xf16>) outs(%init : tensor<32x17xf32>) -> tensor<32x17xf32>
  return %product : tensor<32x17xf32>
}
