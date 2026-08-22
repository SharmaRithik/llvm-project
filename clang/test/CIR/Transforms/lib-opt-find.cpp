// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.a64.cir
// RUN: FileCheck %s --input-file=%t.a64.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnux32 -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.x32.cir
// RUN: FileCheck %s --input-file=%t.x32.cir --check-prefix=XFORM32
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnux32 -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.x32.ll
// RUN: FileCheck %s --input-file=%t.x32.ll --check-prefix=LLVM32
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.lp64.ll
// RUN: FileCheck %s --input-file=%t.lp64.ll --check-prefix=LLVM64
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.a64.ll
// RUN: FileCheck %s --input-file=%t.a64.ll --check-prefix=LLVM64
// RUN: %clang_cc1 -std=c++20 -triple powerpc64le-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.ppc64le.ll
// RUN: FileCheck %s --input-file=%t.ppc64le.ll --check-prefix=LLVMEXT
// RUN: %clang_cc1 -std=c++20 -triple s390x-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.s390x.ll
// RUN: FileCheck %s --input-file=%t.s390x.ll --check-prefix=LLVMEXT
// RUN: %clang_cc1 -std=c++20 -triple i686-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.i686.cir
// RUN: FileCheck %s --input-file=%t.i686.cir --check-prefix=XFORM32
// RUN: %clang_cc1 -std=c++20 -triple spirv-unknown-vulkan-compute -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.spirv.cir
// RUN: FileCheck %s --input-file=%t.spirv.cir --check-prefix=XFORM32
// RUN: %clang_cc1 -std=c++20 -triple msp430-unknown-unknown -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.msp430.cir
// RUN: FileCheck %s --input-file=%t.msp430.cir --check-prefix=XFORM16
// RUN: %clang_cc1 -std=c++20 -triple bpfel -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.bpf.cir
// RUN: FileCheck %s --input-file=%t.bpf.cir --check-prefix=XFORM
// RUN: %clang_cc1 -std=c++20 -triple spir64-unknown-unknown -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.spir64.cir
// RUN: FileCheck %s --input-file=%t.spir64.cir --check-prefix=XFORM
// RUN: %clang_cc1 -std=c++20 -triple powerpc64le-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.ppc64le.cir
// RUN: FileCheck %s --input-file=%t.ppc64le.cir --check-prefix=XFORM
// RUN: %clang_cc1 -std=c++20 -triple s390x-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.s390x.cir
// RUN: FileCheck %s --input-file=%t.s390x.cir --check-prefix=XFORM
// TargetLibraryInfo reports no memchr on these targets, so no call may be
// introduced.
// RUN: %clang_cc1 -std=c++20 -triple amdgcn-amd-amdhsa -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.amdgcn.cir
// RUN: FileCheck %s --input-file=%t.amdgcn.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nvptx.cir
// RUN: FileCheck %s --input-file=%t.nvptx.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -ffreestanding -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.free.cir
// RUN: FileCheck %s --input-file=%t.free.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fno-builtin-memchr -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nomemchr.cir
// RUN: FileCheck %s --input-file=%t.nomemchr.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fno-builtin-strlen -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nostrlen.cir
// RUN: FileCheck %s --input-file=%t.nostrlen.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DMEMCHR_NORETURN -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.noreturn.cir
// RUN: FileCheck %s --input-file=%t.noreturn.cir --check-prefixes=XFORM,NORETURN
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DMEMCHR_NONNULL -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nonnull.cir
// RUN: FileCheck %s --input-file=%t.nonnull.cir --check-prefixes=XFORM,NONNULL
// NORETURN: cir.func private @memchr({{.*}}) -> !cir.ptr<!void> {{.*}}noreturn
// NONNULL: cir.func private @memchr({{.*}}) -> (!cir.ptr<!void> {llvm.nonnull})
// XFORM: cir.libc.memchr
// XFORM32: cir.libc.memchr({{.*}}) : !cir.ptr<!void>, !s32i, !u32i
// XFORM16: cir.libc.memchr({{.*}}) : !cir.ptr<!void>, !s16i, !u16i
// LLVM32: declare ptr @memchr(ptr noundef, i32 noundef, i32 noundef)
// LLVM32: call ptr @memchr(ptr noundef {{.*}}, i32 noundef {{.*}}, i32 noundef {{.*}})
// LLVM64: declare ptr @memchr(ptr noundef, i32 noundef, i64 noundef)
// LLVM64: call ptr @memchr(ptr noundef {{.*}}, i32 noundef {{.*}}, i64 noundef {{.*}})
// LLVMEXT: declare ptr @memchr(ptr noundef, i32 noundef signext, i64 noundef)
// LLVMEXT: call ptr @memchr(ptr noundef {{.*}}, i32 noundef signext {{.*}}, i64 noundef {{.*}})
// NOXFORM: cir.call @_ZNKSt6ranges6__findclIPhS2_hSt8identityEET_S4_T0_RKT1_T2_
// NOXFORM: cir.call @_ZNKSt6ranges6__findclIPiS2_iSt8identityEET_S4_T0_RKT1_T2_
// NOXFORM: cir.call @_ZSt4findIPhhET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPccET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPaaET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPaaET_S1_S1_RKT0_

#ifdef MEMCHR_NORETURN
extern "C" __attribute__((noreturn)) void *memchr(const void *, int,
                                                  unsigned long);
#endif
#ifdef MEMCHR_NONNULL
extern "C" __attribute__((returns_nonnull)) void *memchr(const void *, int,
                                                         unsigned long);
#endif
#if defined(MEMCHR_NORETURN) || defined(MEMCHR_NONNULL)
const void *force_memchr(const void *p) { return memchr(p, 'x', 4); }
#endif

namespace std {
template <class Iter, class T> Iter find(Iter, Iter, const T &);
struct identity {
  template <class T> T &&operator()(T &&t) const;
};
namespace ranges {
struct __find {
  template <class Iter, class Sent, class T, class Proj = identity>
  Iter operator()(Iter first, Sent last, const T &value,
                  Proj proj = {}) const;
};
inline namespace __cpo {
inline constexpr __find find{};
}
}
}

unsigned char *test_byte_ranges_find(unsigned char *first, unsigned char *last,
                                     const unsigned char &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_byte_ranges_findPhS_RKh
// CHECK-NOT: cir.call @_ZNKSt6ranges
// CHECK: cir.libc.memchr
// CHECK-NOT: cir.call @_ZNKSt6ranges

int *test_wide_ranges_find(int *first, int *last, const int &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_wide_ranges_findPiS_RKi
// CHECK-NOT: cir.libc.memchr
// CHECK: cir.call @_ZNKSt6ranges6__findclIPiS2_iSt8identityEET_S4_T0_RKT1_T2_(
// CHECK-NOT: cir.libc.memchr

unsigned char *test_byte_find(unsigned char *first, unsigned char *last,
                              const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_byte_findPhS_RKh
// CHECK: %[[FIRSTSLOT:.*]] = cir.alloca "first" align(8) init : !cir.ptr<!cir.ptr<!u8i>>
// CHECK: %[[LASTSLOT:.*]] = cir.alloca "last" align(8) init : !cir.ptr<!cir.ptr<!u8i>>
// CHECK: %[[VALUESLOT:.*]] = cir.alloca "value" align(8) init const : !cir.ptr<!cir.ptr<!u8i>>
// CHECK: %[[FIRST:.*]] = cir.load align(8) %[[FIRSTSLOT]] : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CHECK: %[[LAST:.*]] = cir.load align(8) %[[LASTSLOT]] : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CHECK-NOT: cir.call
// CHECK-NOT: cir.load %arg2
// CHECK-NOT: cir.ptr_diff
// CHECK-NOT: cir.libc.memchr
// CHECK: %[[EMPTY:.*]] = cir.cmp eq %[[FIRST]], %[[LAST]] : !cir.ptr<!u8i>
// CHECK: cir.ternary(%[[EMPTY]], true {
// CHECK-NEXT: cir.yield %[[LAST]] : !cir.ptr<!u8i>
// CHECK-NEXT: }, false {
// CHECK: %[[SRC:.*]] = cir.cast bitcast %[[FIRST]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CHECK: %[[BYTE:.*]] = cir.load %arg2 : !cir.ptr<!u8i>, !u8i
// CHECK: %[[PATTERN:.*]] = cir.cast integral %[[BYTE]] : !u8i -> !s32i
// CHECK: %[[LEN:.*]] = cir.ptr_diff %[[LAST]], %[[FIRST]] : !cir.ptr<!u8i> -> !u64i
// CHECK: %[[PTR:.*]] = cir.libc.memchr(%[[SRC]], %[[PATTERN]], %[[LEN]]) : !cir.ptr<!void>, !s32i, !u64i
// CHECK: %[[RES:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CHECK: %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CHECK: %[[MISS:.*]] = cir.cmp eq %[[RES]], %[[NULL]]
// CHECK: %[[SELECT:.*]] = cir.select if %[[MISS]] then %[[LAST]] else %[[RES]]
// CHECK-NEXT: cir.yield %[[SELECT]] : !cir.ptr<!u8i>
// CHECK-NEXT: }) : (!cir.bool) -> !cir.ptr<!u8i>

char *test_char_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_char_findPcS_RKc
// CHECK-NOT: cir.call
// CHECK: cir.cast integral %{{.*}} : !s8i -> !s32i
// CHECK: %[[CLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s8i> -> !u64i
// CHECK: %[[CPTR:.*]] = cir.libc.memchr(%{{.*}}, %{{.*}}, %[[CLEN]]) : !cir.ptr<!void>, !s32i, !u64i
// CHECK: %[[CRES:.*]] = cir.cast bitcast %[[CPTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
// CHECK: %[[CMISS:.*]] = cir.cmp eq %[[CRES]], %{{.*}}
// CHECK: cir.select if %[[CMISS]] then %{{.*}} else %[[CRES]]

signed char *test_high_bit_find(signed char *first, signed char *last) {
  signed char value = -128;
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_high_bit_findPaS_
// CHECK: %[[SLOT:.*]] = cir.alloca "value" align(1) init : !cir.ptr<!s8i>
// CHECK: %[[HIGH:.*]] = cir.const #cir.int<-128> : !s8i
// CHECK: cir.store align(1) %[[HIGH]], %[[SLOT]] : !s8i, !cir.ptr<!s8i>
// CHECK-NOT: cir.call
// CHECK: %[[HBYTE:.*]] = cir.load %[[SLOT]] : !cir.ptr<!s8i>, !s8i
// CHECK: %[[HPATTERN:.*]] = cir.cast integral %[[HBYTE]] : !s8i -> !s32i
// CHECK: cir.libc.memchr(%{{.*}}, %[[HPATTERN]], %{{.*}}) : !cir.ptr<!void>, !s32i, !u64i

signed char *test_signed_char_find(signed char *first, signed char *last,
                                   const signed char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_signed_char_find
// CHECK-NOT: cir.call
// CHECK: cir.cast integral %{{.*}} : !s8i -> !s32i
// CHECK: %[[SLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s8i> -> !u64i
// CHECK: %[[SPTR:.*]] = cir.libc.memchr(%{{.*}}, %{{.*}}, %[[SLEN]]) : !cir.ptr<!void>, !s32i, !u64i
// CHECK: %[[SRES:.*]] = cir.cast bitcast %[[SPTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
// CHECK: %[[SMISS:.*]] = cir.cmp eq %[[SRES]], %{{.*}}
// CHECK: cir.select if %[[SMISS]] then %{{.*}} else %[[SRES]]

int *test_int_find(int *first, int *last, const int &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z13test_int_findPiS_RKi
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

char *test_sign_mismatch(char *first, char *last, const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_sign_mismatchPcS_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

char *test_char_flavors(char *first, char *last, const signed char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z17test_char_flavorsPcS_RKa
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

volatile unsigned char *test_volatile_find(volatile unsigned char *first,
                                           volatile unsigned char *last,
                                           const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_volatile_findPVhS0_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

unsigned char *test_volatile_value(unsigned char *first, unsigned char *last,
                                   const volatile unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z19test_volatile_valuePhS_RVKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

__attribute__((no_builtin("memchr")))
unsigned char *test_fn_no_builtin(unsigned char *first, unsigned char *last,
                                  const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_fn_no_builtinPhS_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

enum class Tri : unsigned char { A, B, C };
bool operator==(Tri, Tri);
Tri *test_enum_find(Tri *first, Tri *last, const Tri &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_enum_findP3TriS0_RKS_
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

typedef _Atomic(unsigned char) AByte;
AByte *test_atomic_find(AByte *first, AByte *last,
                        const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z16test_atomic_find
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

struct Byte {
  unsigned char b;
  bool operator==(const Byte &) const;
};
Byte *test_record_find(Byte *first, Byte *last, const Byte &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z16test_record_findP4ByteS0_RKS_
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

typedef __attribute__((address_space(1))) unsigned char ASByte;
ASByte *test_addr_space_find(ASByte *first, ASByte *last,
                             const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z20test_addr_space_findPU3AS1hS0_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

char8_t *test_char8_find(char8_t *first, char8_t *last, const char8_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z15test_char8_findPDuS_RKDu
// CHECK-NOT: cir.call
// CHECK: cir.libc.memchr({{.*}}) : !cir.ptr<!void>, !s32i, !u64i
