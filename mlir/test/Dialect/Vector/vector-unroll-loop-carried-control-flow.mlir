// RUN: mlir-opt %s --vector-unroll-loop-carried="target-shape=1,4" --split-input-file | FileCheck %s

// Branch results keep their row order when a vector becomes two tiles.
// CHECK-LABEL: func.func @split_if
// CHECK-SAME: %[[COND:.*]]: i1, %[[LHS:.*]]: vector<2x4xf32>, %[[RHS:.*]]: vector<2x4xf32>
// CHECK-DAG: %[[LV0:.*]] = vector.extract %[[LHS]][0] : vector<4xf32> from vector<2x4xf32>
// CHECK-DAG: %[[LV1:.*]] = vector.extract %[[LHS]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK-DAG: %[[RV0:.*]] = vector.extract %[[RHS]][0] : vector<4xf32> from vector<2x4xf32>
// CHECK-DAG: %[[RV1:.*]] = vector.extract %[[RHS]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK: %[[RESULT:.*]]:2 = scf.if %[[COND]] -> (vector<4xf32>, vector<4xf32>)
// CHECK: scf.yield %[[LV0]], %[[LV1]] : vector<4xf32>, vector<4xf32>
// CHECK: } else {
// CHECK: scf.yield %[[RV0]], %[[RV1]] : vector<4xf32>, vector<4xf32>
// CHECK: return %[[RESULT]]#1 : vector<4xf32>
func.func @split_if(%cond: i1, %lhs: vector<2x4xf32>, %rhs: vector<2x4xf32>) -> vector<4xf32> {
  %r = scf.if %cond -> (vector<2x4xf32>) {
    scf.yield %lhs : vector<2x4xf32>
  } else {
    scf.yield %rhs : vector<2x4xf32>
  }
  %row = vector.extract_strided_slice %r offsets = [1, 0], sizes = [1, 4], strides = [1, 1] : vector<2x4xf32> to vector<1x4xf32>
  %out = vector.shape_cast %row : vector<1x4xf32> to vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// The condition and backedge carry both tiles beside an unchanged scalar.
// CHECK-LABEL: func.func @split_while
// CHECK: %[[RESULT:.*]]:3 = scf.while (%[[IV:.*]] = %{{.*}}, %[[A0:.*]] = %{{.*}}, %[[A1:.*]] = %{{.*}}) : (index, vector<4xf32>, vector<4xf32>) -> (index, vector<4xf32>, vector<4xf32>)
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[IV]], %{{.*}} : index
// CHECK: scf.condition(%[[COND]]) %[[IV]], %[[A0]], %[[A1]] : index, vector<4xf32>, vector<4xf32>
// CHECK: } do {
// CHECK: ^bb0(%[[I:.*]]: index, %[[B0:.*]]: vector<4xf32>, %[[B1:.*]]: vector<4xf32>):
// CHECK-DAG: %[[NEXT:.*]] = arith.addi %[[I]], %{{.*}} : index
// CHECK-DAG: %[[ROW0:.*]] = vector.shape_cast %[[B0]] : vector<4xf32> to vector<1x4xf32>
// CHECK-DAG: %[[ROW1:.*]] = vector.shape_cast %[[B1]] : vector<4xf32> to vector<1x4xf32>
// CHECK: %[[SUM:.*]] = arith.addf %[[ROW0]], %[[ROW1]] : vector<1x4xf32>
// CHECK: %[[S:.*]] = vector.shape_cast %[[SUM]] : vector<1x4xf32> to vector<4xf32>
// CHECK: scf.yield %[[NEXT]], %[[S]], %[[B1]] : index, vector<4xf32>, vector<4xf32>
// CHECK: return %[[RESULT]]#1 : vector<4xf32>
func.func @split_while(%init: vector<2x4xf32>, %n: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.while (%i = %c0, %acc = %init) : (index, vector<2x4xf32>) -> (index, vector<2x4xf32>) {
    %cond = arith.cmpi slt, %i, %n : index
    scf.condition(%cond) %i, %acc : index, vector<2x4xf32>
  } do {
  ^bb0(%i: index, %acc: vector<2x4xf32>):
    %next = arith.addi %i, %c1 : index
    %row0 = vector.extract_strided_slice %acc offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : vector<2x4xf32> to vector<1x4xf32>
    %row1 = vector.extract_strided_slice %acc offsets = [1, 0], sizes = [1, 4], strides = [1, 1] : vector<2x4xf32> to vector<1x4xf32>
    %sum = arith.addf %row0, %row1 : vector<1x4xf32>
    %updated = vector.insert_strided_slice %sum, %acc offsets = [0, 0], strides = [1, 1] : vector<1x4xf32> into vector<2x4xf32>
    scf.yield %next, %updated : index, vector<2x4xf32>
  }
  %row = vector.extract_strided_slice %r#1 offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : vector<2x4xf32> to vector<1x4xf32>
  %out = vector.shape_cast %row : vector<1x4xf32> to vector<4xf32>
  return %out : vector<4xf32>
}
