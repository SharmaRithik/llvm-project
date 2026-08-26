// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -std=c++20 -fclangir \
// RUN:   -fno-clangir-call-conv-lowering -I %S/../../../lib/Headers \
// RUN:   -emit-cir %S/../../../examples/CIRTileMatmul/matmul.cpp -o %t.cir
// RUN: cir-opt %t.cir -cir-flatten-cfg -mem2reg -o %t.ssa.cir
// RUN: FileCheck %s --input-file=%t.ssa.cir

// CHECK-LABEL: cir.func no_inline dso_local @matmul(
// CHECK-SAME: !cir.ptr<!cir.f16> {llvm.noalias, llvm.noundef}
// CHECK-SAME: !cir.ptr<!cir.float> {llvm.noalias, llvm.noundef}
// CHECK-SAME: [#cir.annotation<"cir_tile.v1.kernel">]
// CHECK: %[[BLOCK_M:.*]] = cir.call @{{.*}}bid{{.*}}()
// CHECK: %[[BLOCK_N:.*]] = cir.call @{{.*}}bid{{.*}}()
// CHECK: %[[ZERO:.*]] = cir.call @{{.*}}zeros{{.*}}()
// CHECK: %[[A0:.*]] = cir.call @{{.*}}load
// CHECK: %[[B0:.*]] = cir.call @{{.*}}load
// CHECK: %[[ACC0:.*]] = cir.call @{{.*}}mma{{.*}}(%[[A0]], %[[B0]], %[[ZERO]])
// CHECK: %[[A1:.*]] = cir.call @{{.*}}load
// CHECK: %[[B1:.*]] = cir.call @{{.*}}load
// CHECK: %[[ACC1:.*]] = cir.call @{{.*}}mma{{.*}}(%[[A1]], %[[B1]], %[[ACC0]])
// CHECK: %[[A2:.*]] = cir.call @{{.*}}load
// CHECK: %[[B2:.*]] = cir.call @{{.*}}load
// CHECK: %[[ACC2:.*]] = cir.call @{{.*}}mma{{.*}}(%[[A2]], %[[B2]], %[[ACC1]])
// CHECK: %[[A3:.*]] = cir.call @{{.*}}load
// CHECK: %[[B3:.*]] = cir.call @{{.*}}load
// CHECK: %[[ACC3:.*]] = cir.call @{{.*}}mma{{.*}}(%[[A3]], %[[B3]], %[[ACC2]])
// CHECK: cir.call @{{.*}}store{{.*}}(%arg2, %[[ACC3]], %[[BLOCK_M]], %[[BLOCK_N]])
// CHECK-NEXT: cir.return
