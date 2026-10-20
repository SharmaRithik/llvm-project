// RUN: split-file %s %t
// RUN: mlir-wgsl-compile %t/full.mlir --mma --nvidia-native --storage-alignment=4 | FileCheck %s --check-prefix=FULL
// RUN: mlir-wgsl-compile %t/full.mlir --nvidia-native --storage-alignment=4 | FileCheck %s --check-prefix=FULL
// RUN: mlir-wgsl-compile %t/partial.mlir --mma --nvidia-native --storage-alignment=4 | FileCheck %s --check-prefix=PARTIAL
// RUN: mlir-wgsl-compile %t/partial.mlir --nvidia-native --storage-alignment=4 | FileCheck %s --check-prefix=PARTIAL

// Complete tiles drop every bounds check except the workgroup guard.
// FULL: if (
// FULL-NOT: if (
// FULL: return;

// Partial M/N/K tiles must retain guarded loads and output stores.
// PARTIAL: if (
// PARTIAL: if (
// PARTIAL: if (
// PARTIAL: return;

//--- full.mlir
func.func @matmul(%a: tensor<128x64xf16>, %b: tensor<64x64xf16>) -> tensor<128x64xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x64xf32>) -> tensor<128x64xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<128x64xf16>, tensor<64x64xf16>) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %product : tensor<128x64xf32>
}

//--- partial.mlir
func.func @matmul(%a: tensor<132x68xf16>, %b: tensor<65x68xf16>) -> tensor<132x65xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<132x65xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<132x65xf32>) -> tensor<132x65xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<132x68xf16>, tensor<65x68xf16>) outs(%init : tensor<132x65xf32>) -> tensor<132x65xf32>
  return %product : tensor<132x65xf32>
}
