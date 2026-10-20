// RUN: not mlir-wgsl-compile %s --mma 2>&1 | FileCheck %s
// CHECK: expected static rank-2
func.func @matmul(%a: tensor<?x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<?x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}
