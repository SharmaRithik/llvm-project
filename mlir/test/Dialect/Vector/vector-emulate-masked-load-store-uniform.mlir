// RUN: mlir-opt %s --test-vector-emulate-masked-load-store | FileCheck %s
// RUN: mlir-opt %s --test-vector-emulate-masked-load-store --canonicalize | FileCheck %s --check-prefix=FOLD

// A uniform mask becomes an scf.if that keeps passthrough and alignment.
// CHECK-LABEL: func.func @load_broadcast(
// CHECK-SAME: %[[BASE:.*]]: memref<?x?xf32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[COND:.*]]: i1, %[[PASS:.*]]: vector<4xf32>
// CHECK-NEXT: %[[RESULT:.*]] = scf.if %[[COND]] -> (vector<4xf32>) {
// CHECK-NEXT:   %[[LOADED:.*]] = vector.load %[[BASE]][%[[I]], %[[J]]] alignment = 16 : memref<?x?xf32>, vector<4xf32>
// CHECK-NEXT:   scf.yield %[[LOADED]] : vector<4xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[PASS]] : vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : vector<4xf32>
func.func @load_broadcast(%base: memref<?x?xf32>, %i: index, %j: index, %cond: i1, %pass: vector<4xf32>) -> vector<4xf32> {
  %mask = vector.broadcast %cond : i1 to vector<4xi1>
  %result = vector.maskedload %base[%i, %j], %mask, %pass alignment = 16 : memref<?x?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @load_select(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[COND:.*]]: i1, %[[PASS:.*]]: vector<4xf32>
// CHECK-NEXT: %[[RESULT:.*]] = scf.if %[[COND]] -> (vector<4xf32>) {
// CHECK-NEXT:   %[[LOADED:.*]] = vector.load %[[BASE]][%[[I]]] : memref<?xf32>, vector<4xf32>
// CHECK-NEXT:   scf.yield %[[LOADED]] : vector<4xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[PASS]] : vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : vector<4xf32>
func.func @load_select(%base: memref<?xf32>, %i: index, %cond: i1, %pass: vector<4xf32>) -> vector<4xf32> {
  %all_true = arith.constant dense<true> : vector<4xi1>
  %all_false = arith.constant dense<false> : vector<4xi1>
  %mask = arith.select %cond, %all_true, %all_false : i1, vector<4xi1>
  %result = vector.maskedload %base[%i], %mask, %pass : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @load_select_inverted(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[COND:.*]]: i1, %[[PASS:.*]]: vector<4xf32>
// CHECK-NEXT: %[[RESULT:.*]] = scf.if %[[COND]] -> (vector<4xf32>) {
// CHECK-NEXT:   scf.yield %[[PASS]] : vector<4xf32>
// CHECK-NEXT: } else {
// CHECK-NEXT:   %[[LOADED:.*]] = vector.load %[[BASE]][%[[I]]] : memref<?xf32>, vector<4xf32>
// CHECK-NEXT:   scf.yield %[[LOADED]] : vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : vector<4xf32>
func.func @load_select_inverted(%base: memref<?xf32>, %i: index, %cond: i1, %pass: vector<4xf32>) -> vector<4xf32> {
  %all_true = arith.constant dense<true> : vector<4xi1>
  %all_false = arith.constant dense<false> : vector<4xi1>
  %mask = arith.select %cond, %all_false, %all_true : i1, vector<4xi1>
  %result = vector.maskedload %base[%i], %mask, %pass : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

// No store, including a read-modify-write, is executed on the false path.
// CHECK-LABEL: func.func @store_broadcast(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[COND:.*]]: i1, %[[VALUE:.*]]: vector<4xf32>
// CHECK-NEXT: scf.if %[[COND]] {
// CHECK-NEXT:   vector.store %[[VALUE]], %[[BASE]][%[[I]]] alignment = 16 : memref<?xf32>, vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @store_broadcast(%base: memref<?xf32>, %i: index, %cond: i1, %value: vector<4xf32>) {
  %mask = vector.broadcast %cond : i1 to vector<4xi1>
  vector.maskedstore %base[%i], %mask, %value alignment = 16 : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @store_select(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[COND:.*]]: i1, %[[VALUE:.*]]: vector<4xf32>
// CHECK-NEXT: scf.if %[[COND]] {
// CHECK-NEXT:   vector.store %[[VALUE]], %[[BASE]][%[[I]]] : memref<?xf32>, vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @store_select(%base: memref<?xf32>, %i: index, %cond: i1, %value: vector<4xf32>) {
  %all_true = arith.constant dense<true> : vector<4xi1>
  %all_false = arith.constant dense<false> : vector<4xi1>
  %mask = arith.select %cond, %all_true, %all_false : i1, vector<4xi1>
  vector.maskedstore %base[%i], %mask, %value : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @store_select_inverted(
// CHECK-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[COND:.*]]: i1, %[[VALUE:.*]]: vector<4xf32>
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[INVERSE:.*]] = arith.xori %[[COND]], %[[TRUE]] : i1
// CHECK-NEXT: scf.if %[[INVERSE]] {
// CHECK-NEXT:   vector.store %[[VALUE]], %[[BASE]][%[[I]]] : memref<?xf32>, vector<4xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @store_select_inverted(%base: memref<?xf32>, %i: index, %cond: i1, %value: vector<4xf32>) {
  %all_true = arith.constant dense<true> : vector<4xi1>
  %all_false = arith.constant dense<false> : vector<4xi1>
  %mask = arith.select %cond, %all_false, %all_true : i1, vector<4xi1>
  vector.maskedstore %base[%i], %mask, %value : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return
}

// A scalar condition does not make a select of nonuniform masks uniform.
// CHECK-LABEL: func.func @nonuniform_select(
// CHECK-NOT: vector.load
// CHECK: %[[MASK:.*]] = arith.select
// CHECK: vector.extract %[[MASK]][0]
// CHECK: scf.if
// CHECK: memref.load
// CHECK-COUNT-3: memref.load
// CHECK-NOT: vector.load
// CHECK: return
func.func @nonuniform_select(%base: memref<?xf32>, %i: index, %cond: i1, %pass: vector<4xf32>) -> vector<4xf32> {
  %some = arith.constant dense<[true, false, true, false]> : vector<4xi1>
  %all_false = arith.constant dense<false> : vector<4xi1>
  %mask = arith.select %cond, %some, %all_false : i1, vector<4xi1>
  %result = vector.maskedload %base[%i], %mask, %pass : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

// A vector condition can select different values in different lanes.
// CHECK-LABEL: func.func @vector_select_condition(
// CHECK-NOT: vector.store
// CHECK: vector.extract
// CHECK: scf.if
// CHECK: memref.store
// CHECK-COUNT-3: memref.store
// CHECK-NOT: vector.store
// CHECK: return
func.func @vector_select_condition(%base: memref<?xf32>, %i: index, %cond: vector<4xi1>, %value: vector<4xf32>) {
  %all_true = arith.constant dense<true> : vector<4xi1>
  %all_false = arith.constant dense<false> : vector<4xi1>
  %mask = arith.select %cond, %all_true, %all_false : vector<4xi1>, vector<4xi1>
  vector.maskedstore %base[%i], %mask, %value : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return
}

// Keep the existing scalable-mask behavior, even for a scalar broadcast.
// CHECK-LABEL: func.func @scalable_broadcast(
// CHECK: vector.maskedload
// CHECK: vector.maskedstore
func.func @scalable_broadcast(%base: memref<?xf32>, %i: index, %cond: i1, %pass: vector<[4]xf32>) -> vector<[4]xf32> {
  %mask = vector.broadcast %cond : i1 to vector<[4]xi1>
  %result = vector.maskedload %base[%i], %mask, %pass : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
  vector.maskedstore %base[%i], %mask, %result : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32>
  return %result : vector<[4]xf32>
}

// Constant false must not access memory at all; %i need not be in bounds.
// FOLD-LABEL: func.func @constant_false(
// FOLD-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[PASS:.*]]: vector<4xf32>
// FOLD-NEXT: return %[[PASS]] : vector<4xf32>
func.func @constant_false(%base: memref<?xf32>, %i: index, %pass: vector<4xf32>) -> vector<4xf32> {
  %mask = arith.constant dense<false> : vector<4xi1>
  %result = vector.maskedload %base[%i], %mask, %pass : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  vector.maskedstore %base[%i], %mask, %result : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// FOLD-LABEL: func.func @constant_true(
// FOLD-SAME: %[[BASE:.*]]: memref<?xf32>, %[[I:.*]]: index, %[[VALUE:.*]]: vector<4xf32>
// FOLD-NEXT: %[[LOADED:.*]] = vector.load %[[BASE]][%[[I]]]
// FOLD-NEXT: vector.store %[[VALUE]], %[[BASE]][%[[I]]]
// FOLD-NEXT: return %[[LOADED]] : vector<4xf32>
func.func @constant_true(%base: memref<?xf32>, %i: index, %value: vector<4xf32>) -> vector<4xf32> {
  %mask = arith.constant dense<true> : vector<4xi1>
  %result = vector.maskedload %base[%i], %mask, %value : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  vector.maskedstore %base[%i], %mask, %value : memref<?xf32>, vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}
