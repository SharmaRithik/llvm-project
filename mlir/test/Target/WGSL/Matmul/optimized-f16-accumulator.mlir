// RUN: mlir-wgsl-compile %s --mma --nvidia-native --storage-alignment=4 --max-shared-memory=49152 --dump-prefix=%t | FileCheck %s --implicit-check-not=storageBarrier
// RUN: mlir-opt %t.scheduled.mlir --canonicalize --cse | FileCheck %s --check-prefix=SCHEDULE --implicit-check-not=tensor.
// RUN: FileCheck %s --check-prefix=MAPPED --implicit-check-not=scf.forall --implicit-check-not=arith.select --implicit-check-not=tensor. < %t.lowered.mlir
// RUN: not mlir-wgsl-compile %s --mma --mma-shapes=2 2>&1 | FileCheck %s --check-prefix=CAPABILITY
// CHECK: var<storage, read_write> arg0: array<vec4<f16>>;
// CHECK: @compute @workgroup_size(128, 1, 1)
// CHECK: subgroup_matrix_result<f16, 16, 16>
// CHECK: subgroupMatrixMultiplyAccumulate
// The K loop stays counted and pipelining peels one BK=32 iteration.
// SCHEDULE-DAG: %[[LB:.*]] = arith.constant 0 : index
// SCHEDULE-DAG: %[[UB:.*]] = arith.constant 512 : index
// SCHEDULE-DAG: %[[STEP:.*]] = arith.constant 32 : index
// SCHEDULE: scf.forall
// SCHEDULE: scf.for {{.*}} = %[[LB]] to %[[UB]] step %[[STEP]]
// SCHEDULE: gpu.barrier memfence [#gpu.address_space<workgroup>]
// SCHEDULE: scf.yield
// SCHEDULE: mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]
// MAPPED-DAG: %[[LB:.*]] = arith.constant 0 : index
// MAPPED-DAG: %[[UB:.*]] = arith.constant 480 : index
// MAPPED-DAG: %[[STEP:.*]] = arith.constant 32 : index
// MAPPED: gpu.subgroup_id
// MAPPED: scf.for {{.*}} = %[[LB]] to %[[UB]] step %[[STEP]]
// MAPPED: gpu.subgroup_mma_compute
// MAPPED: scf.yield
// MAPPED: gpu.subgroup_mma_compute
// CAPABILITY: requested schedule exceeds target capabilities
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<512x512xf16>, %b: tensor<256x512xf32>) -> tensor<512x256xf32> {
  %tmp = tensor.empty() : tensor<256x512xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<256x512xf32>) outs(%tmp : tensor<256x512xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<256x512xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<512x256xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<512x256xf16>) -> tensor<512x256xf16>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %half : tensor<512x512xf16>, tensor<256x512xf16>) outs(%init : tensor<512x256xf16>) -> tensor<512x256xf16>
  %out = tensor.empty() : tensor<512x256xf32>
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%product : tensor<512x256xf16>) outs(%out : tensor<512x256xf32>) {
  ^bb0(%x: f16, %unused: f32):
    %f = arith.extf %x : f16 to f32
    linalg.yield %f : f32
  } -> tensor<512x256xf32>
  return %result : tensor<512x256xf32>
}
