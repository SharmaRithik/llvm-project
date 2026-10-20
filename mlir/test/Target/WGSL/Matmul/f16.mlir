// RUN: mlir-wgsl-compile %s --opt-level=0 --mma | FileCheck %s --check-prefix=MMA
// RUN: mlir-wgsl-compile %s --opt-level=0 --dump-prefix=%t | FileCheck %s --check-prefix=REG
// RUN: mlir-opt %t.scheduled.mlir --cse | FileCheck %s --check-prefix=REDUCTION --implicit-check-not=tensor.
// MMA: enable chromium_experimental_subgroup_matrix;
// MMA: @compute @workgroup_size(32, 1, 1)
// MMA: subgroupMatrixMultiplyAccumulate
// REG-NOT: subgroupMatrix
// REG: @compute @workgroup_size(64, 1, 1)
// REG: fma(
// The partial K tile keeps its min(K - k, BK) bound and the K loop.
// REDUCTION: #[[KBOUND:.*]] = affine_map<(d0) -> (-d0 + 19, 16)>
// REDUCTION-DAG: %[[LB:.*]] = arith.constant 0 : index
// REDUCTION-DAG: %[[UB:.*]] = arith.constant 19 : index
// REDUCTION-DAG: %[[STEP:.*]] = arith.constant 16 : index
// REDUCTION: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<4x4xf32>
// REDUCTION: scf.for %[[K:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}} = %[[INIT]]) -> (vector<4x4xf32>)
// REDUCTION: affine.min #[[KBOUND]](%[[K]])
// REDUCTION: vector.contract
// REDUCTION: scf.yield {{.*}} : vector<4x4xf32>
func.func @matmul(%a: tensor<33x19xf16>, %b: tensor<21x19xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<33x19xf16>, tensor<21x19xf16>) outs(%init : tensor<33x21xf32>) -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}
