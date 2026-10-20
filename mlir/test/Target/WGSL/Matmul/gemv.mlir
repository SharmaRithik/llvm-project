// RUN: split-file %s %t
// RUN: mlir-wgsl-compile %t/aligned.mlir --storage-alignment=4 --dump-prefix=%t/gemv | FileCheck %s --check-prefix=VECTOR --implicit-check-not=Barrier --implicit-check-not=subgroup_matrix
// RUN: mlir-wgsl-compile %t/aligned.mlir --mma --storage-alignment=4 | FileCheck %s --check-prefix=VECTOR --implicit-check-not=Barrier --implicit-check-not=subgroup_matrix
// RUN: FileCheck %s --check-prefix=IR < %t/gemv.scheduled.mlir
// RUN: mlir-wgsl-compile %t/partial.mlir | FileCheck %s --check-prefix=SCALAR --implicit-check-not=Barrier
// RUN: mlir-wgsl-compile %t/half.mlir --merged-inputs --batch-count=6 | FileCheck %s --check-prefix=MERGED
// RUN: not mlir-wgsl-compile %t/aligned.mlir --max-workgroup-size=16 2>&1 | FileCheck %s --check-prefix=LIMIT
// RUN: not mlir-wgsl-compile %t/wide.mlir 2>&1 | FileCheck %s --check-prefix=WIDE
// RUN: mlir-wgsl-compile %t/large-k.mlir --dump-prefix=%t/large -o %t/large.wgsl
// RUN: FileCheck %s --check-prefix=LARGE < %t/large.scheduled.mlir

// VECTOR: enable subgroups;
// VECTOR: var<storage, read_write> arg0: array<vec4<f16>>;
// VECTOR: var<storage, read_write> arg1: array<vec4<f32>>;
// VECTOR: var<storage, read_write> arg2: array<f32>;
// VECTOR: @compute @workgroup_size(128, 1, 1)
// VECTOR: fma(
// VECTOR: subgroupAdd(
// IR: scf.for
// IR: arith.extf {{.*}} : vector<4xf16> to vector<4xf32>
// IR-NOT: arith.truncf
// IR: vector.fma
// IR: gpu.subgroup_reduce add
// SCALAR: var<storage, read_write> arg0: array<f16>;
// SCALAR: if (
// SCALAR: for (
// SCALAR: if (
// SCALAR-COUNT-4: subgroupAdd(
// MERGED: var<storage, read_write> arg0: array<f16>;
// MERGED: var<storage, read_write> arg1: array<f32>;
// MERGED: @group(0) @binding(2) var<uniform>
// LIMIT: GEMV requires a supported fixed subgroup size
// WIDE: unconverted f32 RHS requires N <= 4 and f32 accumulation
// LARGE: %[[CHUNKS:.*]] = arith.constant 67108864 : index
// LARGE: %[[STEP:.*]] = arith.constant 1 : index
// LARGE: scf.for {{.*}} to %[[CHUNKS]] step %[[STEP]]

//--- aligned.mlir
func.func @matmul(%a: tensor<2048x2048xf16>, %b: tensor<1x2048xf32>) -> tensor<2048x1xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2048x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2048x1xf32>) -> tensor<2048x1xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<2048x2048xf16>, tensor<1x2048xf32>) outs(%init : tensor<2048x1xf32>) -> tensor<2048x1xf32>
  return %product : tensor<2048x1xf32>
}

//--- partial.mlir
func.func @matmul(%a: tensor<5x35xf16>, %b: tensor<4x35xf32>) -> tensor<5x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<5x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<5x4xf32>) -> tensor<5x4xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<5x35xf16>, tensor<4x35xf32>) outs(%init : tensor<5x4xf32>) -> tensor<5x4xf32>
  return %product : tensor<5x4xf32>
}

//--- half.mlir
func.func @matmul(%a: tensor<5x3xf16>, %b: tensor<2x3xf16>) -> tensor<5x2xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<5x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<5x2xf32>) -> tensor<5x2xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<5x3xf16>, tensor<2x3xf16>) outs(%init : tensor<5x2xf32>) -> tensor<5x2xf32>
  return %product : tensor<5x2xf32>
}

//--- wide.mlir
func.func @matmul(%a: tensor<32x32xf16>, %b: tensor<5x32xf32>) -> tensor<32x5xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x5xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x5xf32>) -> tensor<32x5xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<32x32xf16>, tensor<5x32xf32>) outs(%init : tensor<32x5xf32>) -> tensor<32x5xf32>
  return %product : tensor<32x5xf32>
}

//--- large-k.mlir
func.func @matmul(%a: tensor<1x2147483647xf16>, %b: tensor<1x2147483647xf32>) -> tensor<1x1xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<1x2147483647xf16>, tensor<1x2147483647xf32>) outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %product : tensor<1x1xf32>
}
