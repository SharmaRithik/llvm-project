// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -std=c++20 -fclangir \
// RUN:   -fno-clangir-call-conv-lowering -I %S/../../../lib/Headers \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

#include <cir_tile.hpp>

namespace ct = cuda::tiles;

// CHECK-DAG: cir.func private @{{.*}}bid{{.*}}() -> (!s32i {{.*}}) [#cir.annotation<"cir_tile.v1.bid", [0 : i32]>]
// CHECK-DAG: cir.func private @{{.*}}zeros{{.*}}() -> (!cir.vector<1024 x !cir.float> {{.*}}) [#cir.annotation<"cir_tile.v1.zero", [32 : i32, 32 : i32]>]
// CHECK-DAG: cir.func private @{{.*}}load{{.*}}(!cir.ptr<!cir.f16> {{.*}}, !s32i {{.*}}, !s32i {{.*}}) -> (!cir.vector<512 x !cir.f16> {{.*}}) [#cir.annotation<"cir_tile.v1.load", [32 : i32, 16 : i32, 64 : i32, 64 : i32]>]
// CHECK-DAG: cir.func private @{{.*}}mma{{.*}}(!cir.vector<512 x !cir.f16> {{.*}}, !cir.vector<512 x !cir.f16> {{.*}}, !cir.vector<1024 x !cir.float> {{.*}}) -> (!cir.vector<1024 x !cir.float> {{.*}}) [#cir.annotation<"cir_tile.v1.mma", [32 : i32, 32 : i32, 16 : i32]>]
// CHECK-DAG: cir.func private @{{.*}}store{{.*}}(!cir.ptr<!cir.float> {{.*}}, !cir.vector<1024 x !cir.float> {{.*}}, !s32i {{.*}}, !s32i {{.*}}) [#cir.annotation<"cir_tile.v1.store", [32 : i32, 32 : i32, 64 : i32, 64 : i32]>]
// CHECK: cir.func no_inline dso_local @tile_frontend(
// CHECK-SAME: !cir.ptr<!cir.f16> {llvm.noalias, llvm.noundef}
// CHECK-SAME: !cir.ptr<!cir.float> {llvm.noalias, llvm.noundef}
// CHECK-SAME: [#cir.annotation<"cir_tile.v1.kernel">]

extern "C" __tile_global__
void tile_frontend(const _Float16 *__restrict a,
                   const _Float16 *__restrict b, float *__restrict c) {
  int block_m = ct::bid<0>();
  auto acc = ct::zeros<32, 32>();
  auto lhs = ct::load<32, 16, 64, 64>(a, block_m, 0);
  auto rhs = ct::load<16, 32, 64, 64>(b, 0, 0);
  acc = ct::mma<32, 32, 16>(lhs, rhs, acc);
  ct::store<32, 32, 64, 64>(c, acc, block_m, 0);
}
