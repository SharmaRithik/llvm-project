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
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fwchar-type=short -fno-signed-wchar -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.short-wchar.cir
// RUN: FileCheck %s --input-file=%t.short-wchar.cir --check-prefix=SHORTWCHAR
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
// RUN: FileCheck %s --input-file=%t.amdgcn.cir --check-prefixes=NOXFORM,NOXFORMW --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.libc.wmemchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nvptx.cir
// RUN: FileCheck %s --input-file=%t.nvptx.cir --check-prefixes=NOXFORM,NOXFORMW --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.libc.wmemchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -ffreestanding -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.free.cir
// RUN: FileCheck %s --input-file=%t.free.cir --check-prefixes=NOXFORM,NOXFORMW --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.libc.wmemchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fno-builtin-memchr -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nomemchr.cir
// RUN: FileCheck %s --input-file=%t.nomemchr.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fno-builtin-wmemchr -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nowmemchr.cir
// RUN: FileCheck %s --input-file=%t.nowmemchr.cir --check-prefix=NOWMEMCHR --implicit-check-not=cir.libc.wmemchr --implicit-check-not=cir.std.
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
// NOXFORMW: cir.call @_ZNKSt6ranges6__findclIPiS2_iSt8identityEET_S4_T0_RKT1_T2_
// NOXFORM: cir.call @_ZSt4findIPhhET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPccET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPaaET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt7find_ifIPhZ20test_literal_find_ifS0_S0_E3$_0ET_S2_S2_T0_
// NOXFORM: cir.call @_ZSt11find_if_notIPaZ24test_literal_find_if_notS0_S0_E3$_0ET_S2_S2_T0_
// NOXFORM: cir.call @_ZSt4findIPaaET_S1_S1_RKT0_
// NOXFORMW: cir.call @_ZSt4findIPwwET_S1_S1_RKT0_
// NOXFORMW: cir.call @_ZSt4findIPwwET_S1_S1_RKT0_
// NOWMEMCHR: cir.libc.memchr
// NOWMEMCHR: cir.call @_ZSt4findIPwwET_S1_S1_RKT0_
// NOWMEMCHR: cir.call @_ZSt4findIPwwET_S1_S1_RKT0_

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
template <class Iter, class Pred> Iter find_if(Iter first, Iter last, Pred pred) {
  for (; first != last; ++first)
    if (pred(*first))
      break;
  return first;
}
template <class Iter, class Pred>
Iter find_if_not(Iter first, Iter last, Pred pred) {
  for (; first != last; ++first)
    if (!pred(*first))
      break;
  return first;
}
struct identity {
  template <class T> T &&operator()(T &&t) const;
};

struct bit_alloc {
  using word_type = unsigned long;
  using pointer = word_type *;
  using const_pointer = const word_type *;
};

template <class T, class Alloc = bit_alloc> class vector;
template <class Alloc> class vector<bool, Alloc> {
public:
  using __storage_type = typename Alloc::word_type;
  using __storage_pointer = typename Alloc::pointer;
  using __const_storage_pointer = typename Alloc::const_pointer;
};

template <bool, class True, class False> struct __choose;
template <class True, class False> struct __choose<true, True, False> {
  using type = True;
};
template <class True, class False> struct __choose<false, True, False> {
  using type = False;
};

template <class Cp, bool IsConst> class __bit_iterator {
  using storage_pointer =
      typename __choose<IsConst, typename Cp::__const_storage_pointer,
                        typename Cp::__storage_pointer>::type;
  storage_pointer __seg_;
  unsigned __ctz_;

public:
  bool operator!=(const __bit_iterator &) const;
  __bit_iterator &operator++();
  bool operator*() const;
};

namespace ranges {
struct __find {
  template <class Iter, class Sent, class T, class Proj = identity>
  Iter operator()(Iter first, Sent last, const T &value,
                  Proj proj = {}) const;
};
struct __find_if {
  template <class Iter, class Sent, class Pred, class Proj = identity>
  Iter operator()(Iter first, Sent last, Pred pred, Proj proj = {}) const {
    return std::find_if(first, last, pred);
  }
};
inline namespace __cpo {
inline constexpr __find find{};
inline constexpr __find_if find_if{};
}
}
}

using bit_iterator = std::__bit_iterator<std::vector<bool>, false>;

unsigned char *test_byte_ranges_find(unsigned char *first, unsigned char *last,
                                     const unsigned char &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_byte_ranges_findPhS_RKh
// CHECK-NOT: cir.call @_ZNKSt6ranges
// CHECK: cir.libc.memchr
// CHECK-NOT: cir.call @_ZNKSt6ranges

unsigned char *test_literal_ranges_find_if(unsigned char *first,
                                           unsigned char *last) {
  return std::ranges::find_if(
      first, last, [](unsigned char element) { return element == 0x80; });
}
// CHECK-LABEL: @_Z27test_literal_ranges_find_ifPhS_
// CHECK: %[[RANGES_LITERAL_BYTE:.*]] = cir.const #cir.int<128> : !u8i
// CHECK: %[[RANGES_LITERAL_PATTERN:.*]] = cir.cast integral %[[RANGES_LITERAL_BYTE]] : !u8i -> !s32i
// CHECK: cir.libc.memchr(%{{.*}}, %[[RANGES_LITERAL_PATTERN]], %{{.*}}) : !cir.ptr<!void>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZNKSt6ranges9__find_if

bit_iterator test_ranges_bool_find_if(bit_iterator first,
                                      bit_iterator last) {
  return std::ranges::find_if(
      first, last, [](bool element) { return element; });
}
// CHECK-LABEL: @_Z24test_ranges_bool_find_if
// CHECK: cir.alloca "find_bit_word"
// CHECK: cir.ctz
// CHECK-NOT: cir.call @_ZNKSt6ranges9__find_if

int *test_wide_ranges_find(int *first, int *last, const int &value) {
  return std::ranges::find(first, last, value);
}
// An int of the target wchar_t width rewrites through wmemchr here too.
// CHECK-LABEL: @_Z21test_wide_ranges_findPiS_RKi
// CHECK-NOT: cir.libc.memchr
// CHECK: cir.libc.wmemchr({{.*}}) : !cir.ptr<!s32i>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZNKSt6ranges6__find
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
// CHECK: %[[BYTE:.*]] = cir.load %arg2 : !cir.ptr<!u8i>, !u8i
// CHECK: %[[LEN:.*]] = cir.ptr_diff %[[LAST]], %[[FIRST]] : !cir.ptr<!u8i> -> !u64i
// CHECK: %[[SRC:.*]] = cir.cast bitcast %[[FIRST]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CHECK: %[[PATTERN:.*]] = cir.cast integral %[[BYTE]] : !u8i -> !s32i
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
// CHECK: %[[CLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s8i> -> !u64i
// CHECK: cir.cast integral %{{.*}} : !s8i -> !s32i
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

unsigned char *test_literal_find_if(unsigned char *first,
                                    unsigned char *last) {
  return std::find_if(
      first, last, [](unsigned char element) { return element == 0x80; });
}
// CHECK-LABEL: @_Z20test_literal_find_ifPhS_
// CHECK: %[[LITERAL_BYTE:.*]] = cir.const #cir.int<128> : !u8i
// CHECK: %[[LITERAL_PATTERN:.*]] = cir.cast integral %[[LITERAL_BYTE]] : !u8i -> !s32i
// CHECK: cir.libc.memchr(%{{.*}}, %[[LITERAL_PATTERN]], %{{.*}}) : !cir.ptr<!void>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZSt7find_if

signed char *test_literal_find_if_not(signed char *first, signed char *last) {
  return std::find_if_not(
      first, last, [](signed char element) { return element != -128; });
}
// CHECK-LABEL: @_Z24test_literal_find_if_notPaS_
// CHECK: %[[NOT_LITERAL_BYTE:.*]] = cir.const #cir.int<-128> : !s8i
// CHECK: %[[NOT_LITERAL_PATTERN:.*]] = cir.cast integral %[[NOT_LITERAL_BYTE]] : !s8i -> !s32i
// CHECK: cir.libc.memchr(%{{.*}}, %[[NOT_LITERAL_PATTERN]], %{{.*}}) : !cir.ptr<!void>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZSt11find_if_not

signed char *test_signed_char_find(signed char *first, signed char *last,
                                   const signed char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_signed_char_find
// CHECK-NOT: cir.call
// CHECK: %[[SLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s8i> -> !u64i
// CHECK: cir.cast integral %{{.*}} : !s8i -> !s32i
// CHECK: %[[SPTR:.*]] = cir.libc.memchr(%{{.*}}, %{{.*}}, %[[SLEN]]) : !cir.ptr<!void>, !s32i, !u64i
// CHECK: %[[SRES:.*]] = cir.cast bitcast %[[SPTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
// CHECK: %[[SMISS:.*]] = cir.cmp eq %[[SRES]], %{{.*}}
// CHECK: cir.select if %[[SMISS]] then %{{.*}} else %[[SRES]]

int *test_int_find(int *first, int *last, const int &value) {
  return std::find(first, last, value);
}
// An int of the target wchar_t width rewrites to wmemchr, whose count is
// elements rather than bytes, so no width scaling appears here.
// CHECK-LABEL: @_Z13test_int_findPiS_RKi
// CHECK-NOT: cir.call
// CHECK: %[[ILEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s32i> -> !u64i
// CHECK: %[[IPTR:.*]] = cir.libc.wmemchr(%{{.*}}, %{{.*}}, %[[ILEN]]) : !cir.ptr<!s32i>, !s32i, !u64i
// CHECK: %[[IMISS:.*]] = cir.cmp eq %[[IPTR]], %{{.*}}
// CHECK: cir.select if %[[IMISS]] then %{{.*}} else %[[IPTR]]
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

wchar_t *test_wide_find(wchar_t *first, wchar_t *last, const wchar_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_wide_findPwS_RKw
// CHECK-NOT: cir.call
// CHECK: %[[WBYTE:.*]] = cir.load %arg2 : !cir.ptr<![[WCH:[su]32i]]>, ![[WCH]]
// CHECK: %[[WLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<![[WCH]]> -> !u64i
// CHECK: %[[WPTR:.*]] = cir.libc.wmemchr(%{{.*}}, %[[WBYTE]], %[[WLEN]]) : !cir.ptr<![[WCH]]>, ![[WCH]], !u64i
// CHECK: %[[WMISS:.*]] = cir.cmp eq %[[WPTR]], %{{.*}}
// CHECK: cir.select if %[[WMISS]] then %{{.*}} else %[[WPTR]]
// LLVMEXT: declare ptr @wmemchr(ptr noundef, i32 noundef signext, i64 noundef)
// LLVMEXT: call ptr @wmemchr(ptr noundef {{.*}}, i32 noundef signext {{.*}}, i64 noundef {{.*}})
// LLVM64: declare ptr @wmemchr(ptr noundef, i32 noundef, i64 noundef)
// LLVM64: call ptr @wmemchr(ptr noundef {{.*}}, i32 noundef {{.*}}, i64 noundef {{.*}})
// SHORTWCHAR-LABEL: @_Z14test_wide_findPwS_RKw
// SHORTWCHAR-NOT: cir.libc.wmemchr
// SHORTWCHAR: cir.call @_ZSt4findIPwwET_S1_S1_RKT0_
// SHORTWCHAR-NOT: cir.libc.wmemchr

wchar_t *test_wide_find_if(wchar_t *first, wchar_t *last,
                           const wchar_t &value) {
  return std::find_if(first, last,
                      [&](wchar_t element) { return element == value; });
}
// The wide predicate marker now licenses this form like the byte one. The
// element type is the triple's wchar_t, signed or unsigned by target.
// CHECK-LABEL: @_Z17test_wide_find_ifPwS_RKw
// CHECK: cir.libc.wmemchr(
// CHECK-NOT: cir.call @_ZSt7find_ifIPw

volatile wchar_t *test_volatile_wide_find(volatile wchar_t *first,
                                          volatile wchar_t *last,
                                          const wchar_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z23test_volatile_wide_findPVwS0_RKw
// CHECK: cir.call @_ZSt4findIPVwwET_S2_S2_RKT0_
// CHECK-NOT: cir.libc.wmemchr

__attribute__((no_builtin("wmemchr")))
wchar_t *test_wide_no_builtin(wchar_t *first, wchar_t *last,
                              const wchar_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z20test_wide_no_builtinPwS_RKw
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.wmemchr
// CHECK-NOT: cir.std.find

int *test_int_find_if(int *first, int *last, const int &value) {
  return std::find_if(first, last, [&](int e) { return e == value; });
}
// CHECK-LABEL: @_Z16test_int_find_ifPiS_RKi
// CHECK: cir.libc.wmemchr({{.*}}) : !cir.ptr<!s32i>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZSt7find_ifIPi

int *test_int_find_if_not(int *first, int *last, const int &value) {
  return std::find_if_not(first, last, [&](int e) { return e != value; });
}
// CHECK-LABEL: @_Z20test_int_find_if_notPiS_RKi
// CHECK: cir.libc.wmemchr({{.*}}) : !cir.ptr<!s32i>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZSt11find_if_notIPi

int *test_int_find_if_literal(int *first, int *last) {
  return std::find_if(first, last, [](int e) { return e == 70000; });
}
// CHECK-LABEL: @_Z24test_int_find_if_literalPiS_
// CHECK: %[[WLIT:.*]] = cir.const #cir.int<70000> : !s32i
// CHECK: cir.libc.wmemchr({{.*}}, %[[WLIT]], %{{.*}}) : !cir.ptr<!s32i>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZSt7find_ifIPi

int *test_int_ranges_find_if(int *first, int *last, const int &value) {
  return std::ranges::find_if(first, last, [&](int e) { return e == value; });
}
// CHECK-LABEL: @_Z23test_int_ranges_find_ifPiS_RKi
// CHECK: cir.libc.wmemchr({{.*}}) : !cir.ptr<!s32i>, !s32i, !u64i
// CHECK-NOT: cir.call @_ZNKSt6ranges9__find_if
