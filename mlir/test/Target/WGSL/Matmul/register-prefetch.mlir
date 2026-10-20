// RUN: mlir-wgsl-compile %s --nvidia-native --storage-alignment=4 --max-shared-memory=49152 --dump-prefix=%t | FileCheck %s --implicit-check-not=storageBarrier --implicit-check-not=subgroup_matrix
// RUN: mlir-opt %t.scheduled.mlir --cse | FileCheck %s --check-prefix=SCHEDULE --implicit-check-not=tensor.
// RUN: FileCheck %s --check-prefix=LOWERED --implicit-check-not=scf.forall --implicit-check-not=tensor. < %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=VALUE-STATE < %t.lowered.mlir
// Copy views are hoisted before pipelining and the K loop stays counted.
// CHECK-DAG: @binding(0) var<storage, read_write> arg0: array<vec4<f16>>;
// CHECK-DAG: @binding(1) var<storage, read_write> arg1: array<vec4<f32>>;
// CHECK-DAG: @binding(2) var<storage, read_write> arg2: array<vec4<f32>>;
// CHECK-DAG: @binding(3) var<uniform> arg3:
// CHECK: @compute @workgroup_size(256, 1, 1)
// CHECK: for (var [[K:iv[0-9]+]]: u32 = 0u; bitcast<i32>([[K]]) < bitcast<i32>(112u); [[K]] += 16u)
// CHECK: workgroupBarrier();
// CHECK: fma(
// SCHEDULE-DAG: %[[LB:.*]] = arith.constant 0 : index
// SCHEDULE-DAG: %[[UB:.*]] = arith.constant 128 : index
// SCHEDULE-DAG: %[[BK:.*]] = arith.constant 16 : index
// SCHEDULE: scf.forall
// SCHEDULE: scf.for {{.*}} = %[[LB]] to %[[UB]] step %[[BK]]
// SCHEDULE: vector.contract
// SCHEDULE: wgsl.prefetch
// LOWERED-DAG: %[[LB:.*]] = arith.constant 0 : index
// LOWERED-DAG: %[[UB:.*]] = arith.constant 112 : index
// LOWERED-DAG: %[[BK:.*]] = arith.constant 16 : index
// LOWERED: memref.reinterpret_cast {{.*}} to memref<1x4xf16, strided<[1, 64], offset: ?>, #gpu.address_space<workgroup>>
// LOWERED: %[[STATE:[a-zA-Z0-9_]+]]:{{[0-9]+}} = scf.for {{.*}} = %[[LB]] to %[[UB]] step %[[BK]]
// LOWERED: gpu.barrier memfence [#gpu.address_space<workgroup>]
// LOWERED: vector.fma
// LOWERED: scf.yield
// LOWERED: memref.store %[[STATE]]#
// LOWERED: gpu.barrier memfence [#gpu.address_space<workgroup>]
// LOWERED: vector.fma
// VALUE-STATE-LABEL: gpu.func @main
// VALUE-STATE-NOT: scf.for{{.*}}memref<
// VALUE-STATE: gpu.return
#id = affine_map<(i, j) -> (i, j)>
func.func @matmul(%a: tensor<128x128xf16>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tmp = tensor.empty() : tensor<128x128xf16>
  %half = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%b : tensor<128x128xf32>) outs(%tmp : tensor<128x128xf16>) {
  ^bb0(%x: f32, %unused: f16):
    %h = arith.truncf %x : f32 to f16
    linalg.yield %h : f16
  } -> tensor<128x128xf16>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %product = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %half : tensor<128x128xf16>, tensor<128x128xf16>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %product : tensor<128x128xf32>
}
