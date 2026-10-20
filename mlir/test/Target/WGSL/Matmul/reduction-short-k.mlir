// RUN: mlir-wgsl-compile %s --opt-level=0 --dump-prefix=%t > /dev/null
// RUN: mlir-opt %t.scheduled.mlir --cse | FileCheck %s --implicit-check-not=tensor.
// RUN: mlir-wgsl-compile %s --opt-level=0 --mma --dump-prefix=%t.mma > /dev/null
// RUN: mlir-opt %t.mma.scheduled.mlir --cse | FileCheck %s --check-prefix=MMA --implicit-check-not=tensor.
// A K shorter than one tile zero-fills the rest of the shared tile.
// CHECK-DAG: %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 16 : index
// CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<4x4xf32>
// CHECK: scf.for {{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}} = %[[INIT]]) -> (vector<4x4xf32>)
// CHECK: vector.contract
// CHECK: scf.yield {{.*}} : vector<4x4xf32>
// MMA-DAG: %[[LB:.*]] = arith.constant 0 : index
// MMA-DAG: %[[UB:.*]] = arith.constant 7 : index
// MMA-DAG: %[[STEP:.*]] = arith.constant 16 : index
// MMA: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
// MMA: scf.for {{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}} = %[[INIT]]) -> (vector<16x16xf32>)
// MMA: vector.contract
// MMA: scf.yield {{.*}} : vector<16x16xf32>
func.func @matmul(%a: tensor<33x7xf16>, %b: tensor<21x7xf16>) -> tensor<33x21xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<33x21xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<33x21xf32>) -> tensor<33x21xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<33x7xf16>, tensor<21x7xf16>) outs(%init : tensor<33x21xf32>) -> tensor<33x21xf32>
  return %product : tensor<33x21xf32>
}
