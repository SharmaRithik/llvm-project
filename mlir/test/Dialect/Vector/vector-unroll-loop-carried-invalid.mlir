// RUN: not mlir-opt %s --vector-unroll-loop-carried=target-shape=0,16 2>&1 | FileCheck %s
// RUN: not mlir-opt %s --vector-unroll-loop-carried=target-shape=-1,16 2>&1 | FileCheck %s
// RUN: not mlir-opt %s --vector-unroll-loop-carried=target-shape= 2>&1 | FileCheck %s

// CHECK: error: target-shape must contain positive dimensions
module {
  func.func @invalid_shape() {
    return
  }
}
