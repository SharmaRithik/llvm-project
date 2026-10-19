// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-gpu),canonicalize)" --split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// A prefetched vector carried next to the accumulator stays a vector.
// CHECK-LABEL: func @matmul_loop_with_carried_arithmetic
//       CHECK:   %[[R:.+]]:2 = scf.for {{.*}} iter_args(%[[BUF:.+]] = %{{.*}}, %[[ACC:.+]] = %{{.*}}) -> (vector<8x4xf16>, !gpu.mma_matrix<16x16xf16, "COp">)
//       CHECK:     vector.store {{.*}} : memref<128x128xf16>, vector<8x4xf16>
//       CHECK:     %[[RAW:.+]] = vector.load {{.*}} : memref<128x128xf16>, vector<8x4xf16>
//       CHECK:     %[[NEXT:.+]] = arith.addf %[[RAW]], %[[RAW]] : vector<8x4xf16>
//       CHECK:     %[[D:.+]] = gpu.subgroup_mma_compute {{.*}}, %[[ACC]]
//       CHECK:     scf.yield %[[NEXT]], %[[D]] : vector<8x4xf16>, !gpu.mma_matrix<16x16xf16, "COp">
//       CHECK:   gpu.subgroup_mma_store_matrix %[[R]]#1
func.func @matmul_loop_with_carried_arithmetic(%arg0: memref<128x128xf16>, %arg1: memref<128x128xf16>, %arg2: memref<128x128xf16>, %stage: memref<128x128xf16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f16
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
  %buf0 = vector.load %arg0[%c0, %c0] : memref<128x128xf16>, vector<8x4xf16>
  %r:2 = scf.for %iv = %c0 to %c128 step %c32 iter_args(%buf = %buf0, %acc = %C) -> (vector<8x4xf16>, vector<16x16xf16>) {
    vector.store %buf, %stage[%c0, %iv] : memref<128x128xf16>, vector<8x4xf16>
    %raw = vector.load %arg0[%iv, %c0] : memref<128x128xf16>, vector<8x4xf16>
    %next = arith.addf %raw, %raw : vector<8x4xf16>
    %a = vector.transfer_read %stage[%c0, %iv], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
    %b = vector.transfer_read %arg1[%iv, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
    %d = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %acc : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    scf.yield %next, %d : vector<8x4xf16>, vector<16x16xf16>
  }
  vector.transfer_write %r#1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<128x128xf16>
  return
}


// A nested loop must follow the same carried position at both levels.
// CHECK-LABEL: func @matmul_nested_carried_arithmetic
// CHECK-COUNT-2: scf.for {{.*}} -> (vector<8x4xf16>, !gpu.mma_matrix<16x16xf16, "COp">)
// CHECK: arith.addf {{.*}} : vector<8x4xf16>
// CHECK: gpu.subgroup_mma_compute
// CHECK: gpu.subgroup_mma_store_matrix
func.func @matmul_nested_carried_arithmetic(%arg0: memref<128x128xf16>, %arg1: memref<128x128xf16>, %arg2: memref<128x128xf16>, %stage: memref<128x128xf16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f16
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
  %buf0 = vector.load %arg0[%c0, %c0] : memref<128x128xf16>, vector<8x4xf16>
  %outer:2 = scf.for %j = %c0 to %c128 step %c32 iter_args(%outer_buf = %buf0, %outer_acc = %C) -> (vector<8x4xf16>, vector<16x16xf16>) {
    %r:2 = scf.for %iv = %c0 to %c128 step %c32 iter_args(%buf = %outer_buf, %acc = %outer_acc) -> (vector<8x4xf16>, vector<16x16xf16>) {
      vector.store %buf, %stage[%c0, %iv] : memref<128x128xf16>, vector<8x4xf16>
      %raw = vector.load %arg0[%iv, %c0] : memref<128x128xf16>, vector<8x4xf16>
      %next = arith.addf %raw, %raw : vector<8x4xf16>
      %a = vector.transfer_read %stage[%c0, %iv], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
      %b = vector.transfer_read %arg1[%iv, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
      %d = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %acc : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
      scf.yield %next, %d : vector<8x4xf16>, vector<16x16xf16>
    }
    scf.yield %r#0, %r#1 : vector<8x4xf16>, vector<16x16xf16>
  }
  vector.transfer_write %outer#1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<128x128xf16>
  return
}

// An unsupported use of the accumulator still prevents conversion.
// CHECK-LABEL: func @matmul_with_unsupported_accumulator_use
// CHECK-NOT: gpu.subgroup_mma
// CHECK: vector.contract
// CHECK-NOT: gpu.subgroup_mma
// CHECK: return
func.func @matmul_with_unsupported_accumulator_use(%arg0: memref<128x128xf16>, %arg1: memref<128x128xf16>, %arg2: memref<128x128xf16>, %stage: memref<128x128xf16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f16
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
  %buf0 = vector.load %arg0[%c0, %c0] : memref<128x128xf16>, vector<8x4xf16>
  %r:2 = scf.for %iv = %c0 to %c128 step %c32 iter_args(%buf = %buf0, %acc = %C) -> (vector<8x4xf16>, vector<16x16xf16>) {
    %element = vector.extract %acc[0, 0] : f16 from vector<16x16xf16>
    memref.store %element, %arg2[%iv, %c0] : memref<128x128xf16>
    vector.store %buf, %stage[%c0, %iv] : memref<128x128xf16>, vector<8x4xf16>
    %raw = vector.load %arg0[%iv, %c0] : memref<128x128xf16>, vector<8x4xf16>
    %next = arith.addf %raw, %raw : vector<8x4xf16>
    %a = vector.transfer_read %stage[%c0, %iv], %cst {in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
    %b = vector.transfer_read %arg1[%iv, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<128x128xf16>, vector<16x16xf16>
    %d = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %acc : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    scf.yield %next, %d : vector<8x4xf16>, vector<16x16xf16>
  }
  vector.transfer_write %r#1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<128x128xf16>
  return
}
