// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DDEFINE_MEMCMP -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.x86.cir
// RUN: FileCheck %s --input-file=%t.x86.cir --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -DDEFINE_MEMCMP -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.a64.cir
// RUN: FileCheck %s --input-file=%t.a64.cir --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm -disable-llvm-passes %s -o %t.x86.ll
// RUN: FileCheck %s --input-file=%t.x86.ll --check-prefix=LLVM64
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm -disable-llvm-passes %s -o %t.a64.ll
// RUN: FileCheck %s --input-file=%t.a64.ll --check-prefix=LLVM64
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.opt.ll
// RUN: FileCheck %s --input-file=%t.opt.ll --check-prefix=OPT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnux32 -DDEFINE_MEMCMP -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.x32.cir
// RUN: FileCheck %s --input-file=%t.x32.cir --check-prefix=ILP32
// RUN: %clang_cc1 -std=c++20 -triple powerpc64le-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm -disable-llvm-passes %s -o %t.ppc.ll
// RUN: FileCheck %s --input-file=%t.ppc.ll --check-prefix=PPC-CIR
// RUN: %clang_cc1 -std=c++20 -triple powerpc64le-unknown-linux-gnu -DDEFINE_MEMCMP -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm -disable-llvm-passes %s -o %t.ppc.define.ll
// RUN: FileCheck %s --input-file=%t.ppc.define.ll --check-prefix=PPC-LOCAL
// RUN: %clang_cc1 -std=c++20 -triple powerpc64le-unknown-linux-gnu -DCLASSIC_MEMCMP -emit-llvm -disable-llvm-passes %s -o %t.ppc.classic.ll
// RUN: FileCheck %s --input-file=%t.ppc.classic.ll --check-prefix=PPC-CLASSIC
// RUN: %clang_cc1 -std=c++20 -triple amdgcn-amd-amdhsa -DDEFINE_MEMCMP -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.amdgcn.cir
// RUN: FileCheck %s --input-file=%t.amdgcn.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memcmp --implicit-check-not=cir.std.equal
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DDEFINE_MEMCMP -ffreestanding -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.free.cir
// RUN: FileCheck %s --input-file=%t.free.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memcmp --implicit-check-not=cir.std.equal
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DDEFINE_MEMCMP -fno-builtin-memcmp -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.no.cir
// RUN: FileCheck %s --input-file=%t.no.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memcmp --implicit-check-not=cir.std.equal

// CIR-LABEL: cir.func dso_local @_Z10byte_equalPhS_S_
// CIR: %[[IS_EMPTY:[0-9]+]] = cir.cmp eq %[[FIRST:[0-9]+]], %[[LAST:[0-9]+]] : !cir.ptr<!u8i>
// CIR-NEXT: %{{[0-9]+}} = cir.ternary(%[[IS_EMPTY]], true {
// CIR-NEXT: %[[TRUE:[0-9]+]] = cir.const #true
// CIR-NEXT: cir.yield %[[TRUE]] : !cir.bool
// CIR: }, false {
// CIR-NEXT: %[[LHS:[0-9]+]] = cir.cast bitcast %[[FIRST]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR-NEXT: %[[RHS:[0-9]+]] = cir.cast bitcast %[[SECOND:[0-9]+]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR-NEXT: %[[LEN:[0-9]+]] = cir.ptr_diff %[[LAST]], %[[FIRST]] : !cir.ptr<!u8i> -> !u64i
// CIR-NEXT: %[[MEMCMP:[0-9]+]] = cir.libc.memcmp(%[[LHS]], %[[RHS]], %[[LEN]]) : !cir.ptr<!void>, !cir.ptr<!void>, !u64i -> !s32i
// CIR-NEXT: %[[ZERO:[0-9]+]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: %[[EQUAL:[0-9]+]] = cir.cmp eq %[[MEMCMP]], %[[ZERO]] : !s32i
// CIR-NEXT: cir.yield %[[EQUAL]] : !cir.bool

// The proven capture free equality lambda licenses the same rewrite,
// including the emptiness guard around the call.
// CIR-LABEL: cir.func dso_local @_Z15byte_equal_predPhS_S_
// CIR: %[[PRED_EMPTY:[0-9]+]] = cir.cmp eq %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!u8i>
// CIR: cir.ternary(%[[PRED_EMPTY]], true {
// CIR: %[[PRED_MEMCMP:[0-9]+]] = cir.libc.memcmp(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.ptr<!void>, !cir.ptr<!void>, !u64i -> !s32i
// CIR: cir.cmp eq %[[PRED_MEMCMP]], %{{[0-9]+}} : !s32i

// CIR-LABEL: cir.func dso_local @_Z11empty_equalPhS_
// CIR: %[[EMPTY:[0-9]+]] = cir.cmp eq %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!u8i>
// CIR-NEXT: %{{[0-9]+}} = cir.ternary(%[[EMPTY]], true {
// CIR-NEXT: %[[EMPTY_TRUE:[0-9]+]] = cir.const #true
// CIR-NEXT: cir.yield %[[EMPTY_TRUE]] : !cir.bool

// CIR-LABEL: cir.func dso_local @_Z14unmarked_equalPcS_Ph
// CIR: cir.call @_ZSt5equalIPcPhEbT_S2_T0_
// CIR-LABEL: cir.func dso_local @_Z9int_equalPiS_S_
// CIR: cir.call @_ZSt5equalIPiS0_EbT_S1_T0_
// CIR-LABEL: cir.func dso_local @_Z14volatile_equalPVhS0_S0_
// CIR: cir.call @_ZSt5equalIPVhS1_EbT_S2_T0_
// CIR-LABEL: cir.func dso_local @memcmp
// CIR-NOT: cir.libc.memcmp
// CIR: cir.call @_ZSt5equalIPKhS1_EbT_S2_T0_
// CIR-NOT: cir.libc.memcmp

// LLVM64-DAG: declare i32 @memcmp(ptr noundef, ptr noundef, i64 noundef)
// LLVM64-LABEL: define dso_local {{.*}}i1 @_Z10byte_equalPhS_S_
// LLVM64: call i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
// LLVM64-NEXT: %{{[0-9]+}} = icmp eq i32 %{{[0-9]+}}, 0

// OPT-LABEL: define dso_local {{.*}}i1 @_Z11empty_equalPhS_
// OPT-NEXT: ret i1 true

// ILP32: cir.libc.memcmp({{.*}}) : !cir.ptr<!void>, !cir.ptr<!void>, !u32i -> !s32i

// PPC-CIR: declare i32 @memcmp(ptr noundef, ptr noundef, i64 noundef){{$}}
// PPC-CIR-LABEL: define dso_local {{.*}}i1 @_Z10byte_equalPhS_S_
// PPC-CIR: call i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}){{$}}

// PPC-LOCAL-LABEL: define dso_local {{.*}}i1 @_Z10byte_equalPhS_S_
// PPC-LOCAL: call i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}){{$}}
// PPC-LOCAL-LABEL: define dso_local i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}}) {

// PPC-CLASSIC-LABEL: define dso_local signext i32 @source_memcmp
// PPC-CLASSIC: call signext i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
// PPC-CLASSIC: declare signext i32 @memcmp(ptr noundef, ptr noundef, i64 noundef)

// NOXFORM-LABEL: cir.func dso_local @_Z10byte_equalPhS_S_
// NOXFORM: cir.call @_ZSt5equalIPhS0_EbT_S1_T0_
// NOXFORM-LABEL: cir.func dso_local @_Z15byte_equal_predPhS_S_
// NOXFORM: cir.call @_ZSt5equalIPhS0_

namespace std {
template <class Iter1, class Iter2>
bool equal(Iter1 first1, Iter1 last1, Iter2 first2);
template <class Iter1, class Iter2, class Pred>
bool equal(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred) {
  for (; first1 != last1; ++first1, ++first2)
    if (!pred(*first1, *first2))
      return false;
  return true;
}
}

bool byte_equal(unsigned char *first1, unsigned char *last1,
                unsigned char *first2) {
  return std::equal(first1, last1, first2);
}

bool byte_equal_pred(unsigned char *first1, unsigned char *last1,
                     unsigned char *first2) {
  return std::equal(first1, last1, first2,
                    [](auto a, auto b) { return a == b; });
}

bool empty_equal(unsigned char *first1, unsigned char *first2) {
  return std::equal(first1, first1, first2);
}

bool unmarked_equal(char *first1, char *last1, unsigned char *first2) {
  return std::equal(first1, last1, first2);
}

bool int_equal(int *first1, int *last1, int *first2) {
  return std::equal(first1, last1, first2);
}

bool volatile_equal(volatile unsigned char *first1,
                    volatile unsigned char *last1,
                    volatile unsigned char *first2) {
  return std::equal(first1, last1, first2);
}

#ifdef CLASSIC_MEMCMP
extern "C" int memcmp(const void *lhs, const void *rhs, __SIZE_TYPE__ len);

extern "C" int source_memcmp(const void *lhs, const void *rhs,
                             __SIZE_TYPE__ len) {
  return memcmp(lhs, rhs, len);
}
#endif

#ifdef DEFINE_MEMCMP
extern "C" int memcmp(const void *lhs, const void *rhs, __SIZE_TYPE__ len) {
  const auto *first1 = static_cast<const unsigned char *>(lhs);
  const auto *first2 = static_cast<const unsigned char *>(rhs);
  return std::equal(first1, first1 + len, first2);
}
#endif
