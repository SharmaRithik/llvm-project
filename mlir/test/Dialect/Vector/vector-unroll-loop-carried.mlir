// RUN: mlir-opt %s --vector-unroll-loop-carried="target-shape=16,16" --split-input-file | FileCheck %s
// RUN: mlir-opt %s --vector-unroll-loop-carried="target-shape=1,4" --split-input-file | FileCheck %s --check-prefix=ROWS

// The accumulator is carried as four 16x16 tiles and the glue folds away.
// CHECK-LABEL: func.func @split_for
//  CHECK-SAME:   %[[INIT:.*]]: vector<32x32xf32>
//   CHECK-DAG:   %[[I0:.*]] = vector.extract_strided_slice %[[INIT]] offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
//   CHECK-DAG:   %[[I3:.*]] = vector.extract_strided_slice %[[INIT]] offsets = [16, 16], sizes = [16, 16], strides = [1, 1]
//       CHECK:   %[[R:.*]]:4 = scf.for unsigned {{.*}} iter_args(%[[A0:.*]] = %[[I0]], %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A3:.*]] = %[[I3]]) -> (vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>, vector<16x16xf32>)
//       CHECK:     %[[N0:.*]] = arith.addf %[[A0]], %[[A0]] : vector<16x16xf32>
//       CHECK:     %[[N3:.*]] = arith.addf %[[A3]], %[[A3]] : vector<16x16xf32>
//       CHECK:     scf.yield %[[N0]], %{{.*}}, %{{.*}}, %[[N3]]
//       CHECK:   return %[[R]]#3 : vector<16x16xf32>

// ROWS-LABEL: func.func @split_for
//       ROWS:   scf.for unsigned {{.*}} -> (vector<4xf32>, vector<4xf32>
func.func @split_for(%init: vector<32x32xf32>, %n: index) -> vector<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for unsigned %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (vector<32x32xf32>) {
    %t0 = vector.extract_strided_slice %acc offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : vector<32x32xf32> to vector<16x16xf32>
    %t1 = vector.extract_strided_slice %acc offsets = [0, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf32> to vector<16x16xf32>
    %t2 = vector.extract_strided_slice %acc offsets = [16, 0], sizes = [16, 16], strides = [1, 1] : vector<32x32xf32> to vector<16x16xf32>
    %t3 = vector.extract_strided_slice %acc offsets = [16, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf32> to vector<16x16xf32>
    %n0 = arith.addf %t0, %t0 : vector<16x16xf32>
    %n1 = arith.addf %t1, %t1 : vector<16x16xf32>
    %n2 = arith.addf %t2, %t2 : vector<16x16xf32>
    %n3 = arith.addf %t3, %t3 : vector<16x16xf32>
    %z = ub.poison : vector<32x32xf32>
    %s0 = vector.insert_strided_slice %n0, %z offsets = [0, 0], strides = [1, 1] : vector<16x16xf32> into vector<32x32xf32>
    %s1 = vector.insert_strided_slice %n1, %s0 offsets = [0, 16], strides = [1, 1] : vector<16x16xf32> into vector<32x32xf32>
    %s2 = vector.insert_strided_slice %n2, %s1 offsets = [16, 0], strides = [1, 1] : vector<16x16xf32> into vector<32x32xf32>
    %s3 = vector.insert_strided_slice %n3, %s2 offsets = [16, 16], strides = [1, 1] : vector<16x16xf32> into vector<32x32xf32>
    scf.yield %s3 : vector<32x32xf32>
  }
  %out = vector.extract_strided_slice %r offsets = [16, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf32> to vector<16x16xf32>
  return %out : vector<16x16xf32>
}

// -----

// Vectors that are not a multiple of the target shape are left alone.
// CHECK-LABEL: func.func @not_divisible
//       CHECK:   scf.for {{.*}} -> (vector<24x8xf32>)
func.func @not_divisible(%init: vector<24x8xf32>, %n: index) -> vector<24x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (vector<24x8xf32>) {
    %d = arith.addf %acc, %acc : vector<24x8xf32>
    scf.yield %d : vector<24x8xf32>
  }
  return %r : vector<24x8xf32>
}

// -----

// A vector that is a single tile still drops its leading unit dims.
// ROWS-LABEL: func.func @single_tile
//       ROWS:   scf.for {{.*}} -> (vector<4xf16>)
func.func @single_tile(%init: vector<1x4xf16>, %n: index) -> vector<1x4xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (vector<1x4xf16>) {
    %d = arith.addf %acc, %acc : vector<1x4xf16>
    scf.yield %d : vector<1x4xf16>
  }
  return %r : vector<1x4xf16>
}
