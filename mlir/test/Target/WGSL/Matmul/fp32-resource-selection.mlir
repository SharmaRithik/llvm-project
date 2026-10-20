// RUN: mlir-wgsl-compile %s --mma --nvidia-native --storage-alignment=4 --max-shared-memory=28672 2>&1 | FileCheck %s --check-prefix=MEDIUM
// RUN: mlir-wgsl-compile %s --mma --nvidia-native --storage-alignment=4 --max-shared-memory=28671 2>&1 | FileCheck %s --check-prefix=SMALL
// RUN: mlir-wgsl-compile %s --mma --nvidia-native --storage-alignment=4 --max-shared-memory=8192 2>&1 | FileCheck %s --check-prefix=MINIMAL
// RUN: mlir-wgsl-compile %s --mma --nvidia-native --storage-alignment=4 --max-workgroup-size=64 2>&1 | FileCheck %s --check-prefix=THREADS
// Metadata on stderr can precede the buffered WGSL on stdout.
// MEDIUM-DAG: subgroup_matrix_result<f32, 16, 16>
// MEDIUM-DAG: tile=64x64 threads=128 shared_bytes=28672
// SMALL-DAG: subgroup_matrix_result<f32, 16, 16>
// SMALL-DAG: tile=32x32 threads=128 shared_bytes=10240
// MINIMAL-DAG: subgroup_matrix_result<f32, 16, 16>
// MINIMAL-DAG: tile=32x32 threads=128 shared_bytes=6144
// THREADS-DAG: subgroup_matrix_result<f32, 16, 16>
// THREADS-DAG: tile=16x16 threads=32 shared_bytes=2048
func.func @matmul(%a: tensor<4096x4096xf16>, %b: tensor<512x4096xf16>) -> tensor<4096x512xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4096x512xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4096x512xf32>) -> tensor<4096x512xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<4096x4096xf16>, tensor<512x4096xf16>) outs(%init : tensor<4096x512xf32>) -> tensor<4096x512xf32>
  return %product : tensor<4096x512xf32>
}
