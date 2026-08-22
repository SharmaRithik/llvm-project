// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.x86.cir
// RUN: FileCheck %s --input-file=%t.x86.cir --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.a64.cir
// RUN: FileCheck %s --input-file=%t.a64.cir --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.x86.ll
// RUN: FileCheck %s --input-file=%t.x86.ll --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm %s -o %t.a64.ll
// RUN: FileCheck %s --input-file=%t.a64.ll --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -ffreestanding -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.free.cir
// RUN: FileCheck %s --input-file=%t.free.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memmem --implicit-check-not=cir.std.search
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fno-builtin-memmem -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.no-memmem.cir
// RUN: FileCheck %s --input-file=%t.no-memmem.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memmem --implicit-check-not=cir.std.search
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DMEMMEM_RECURSION -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.recursion.cir
// RUN: FileCheck %s --input-file=%t.recursion.cir --check-prefix=RECURSION --implicit-check-not=cir.libc.memmem --implicit-check-not=cir.std.search

namespace std {
template <class Iter1, class Iter2>
Iter1 search(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);
struct contiguous_iterator_tag {};
template <class Pointer> struct __wrap_iter {
  Pointer ptr;
  using iterator_concept = contiguous_iterator_tag;
};
}

#ifndef MEMMEM_RECURSION
// CIRGen lowers the dynamic initializer to a function before LibOpt.
// CIR-LABEL: cir.func internal private @__cxx_global_var_init
// CIR: cir.libc.memmem(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : !cir.ptr<!void>, !u64i, !cir.ptr<!void>, !u64i
unsigned char *search_hit(unsigned char *first1, unsigned char *last1,
                          unsigned char *first2, unsigned char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CIR-LABEL: cir.func{{.*}} @_Z10search_hitPhS_S_S_
// CIR: %[[EMPTY:.*]] = cir.cmp eq %{{.*}}, %{{.*}} : !cir.ptr<!u8i>
// CIR: cir.ternary(%[[EMPTY]], true {
// CIR-NEXT: cir.yield %{{.*}} : !cir.ptr<!u8i>
// CIR: %[[HAYLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!u8i> -> !u64i
// CIR-NEXT: %[[NEEDLELEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!u8i> -> !u64i
// CIR-NEXT: %[[LONGER:.*]] = cir.cmp gt %[[NEEDLELEN]], %[[HAYLEN]] : !u64i
// CIR-NEXT: cir.ternary(%[[LONGER]], true {
// CIR-NEXT: cir.yield %{{.*}} : !cir.ptr<!u8i>
// CIR: cir.libc.memmem(%{{.*}}, %[[HAYLEN]], %{{.*}}, %[[NEEDLELEN]]) : !cir.ptr<!void>, !u64i, !cir.ptr<!void>, !u64i
// CIR: cir.select if %{{.*}} then %{{.*}} else %{{.*}}
// LLVM: declare ptr @memmem(ptr noundef, i64 noundef, ptr noundef, i64 noundef)
// LLVM-LABEL: define{{.*}} ptr @_Z10search_hitPhS_S_S_
// LLVM: call ptr @memmem(ptr noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}})
// NOXFORM: cir.call @_ZSt6searchIPhS0_ET_S1_S1_T0_S2_

const char *search_const(const char *first1, const char *last1,
                         const char *first2, const char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CIR-LABEL: cir.func{{.*}} @_Z12search_constPKcS0_S0_S0_
// CIR: cir.libc.memmem(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : !cir.ptr<!void>, !u64i, !cir.ptr<!void>, !u64i
// LLVM-LABEL: define{{.*}} ptr @_Z12search_constPKcS0_S0_S0_
// LLVM: call ptr @memmem(ptr noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}})

char *search_mixed_qualified(char *first1, char *last1, const char *first2,
                             const char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CIR-LABEL: cir.func{{.*}} @_Z22search_mixed_qualifiedPcS_PKcS1_
// CIR: cir.libc.memmem(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : !cir.ptr<!void>, !u64i, !cir.ptr<!void>, !u64i
// LLVM-LABEL: define{{.*}} ptr @_Z22search_mixed_qualifiedPcS_PKcS1_
// LLVM: call ptr @memmem(ptr noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}})

char *search_wrapped(std::__wrap_iter<char *> first1,
                     std::__wrap_iter<char *> last1,
                     std::__wrap_iter<char *> first2,
                     std::__wrap_iter<char *> last2) {
  return std::search(first1, last1, first2, last2).ptr;
}
// The libc++ style single member wrapper is recognized and rewritten.
// CIR-LABEL: cir.func{{.*}} @_Z14search_wrappedSt11__wrap_iterIPcES1_S1_S1_
// CIR: cir.extract_member %{{.*}}[0] : !rec_std3A3A__wrap_iter3Cchar_2A3E -> !cir.ptr<!s8i>
// CIR: cir.libc.memmem(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : !cir.ptr<!void>, !u64i, !cir.ptr<!void>, !u64i
// CIR: cir.insert_member %{{.*}}[0], %{{.*}} : !rec_std3A3A__wrap_iter3Cchar_2A3E, !cir.ptr<!s8i>
// LLVM-LABEL: define{{.*}} ptr @_Z14search_wrappedSt11__wrap_iterIPcES1_S1_S1_
// LLVM: call ptr @memmem(ptr noundef %{{.*}}, i64 noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}})

unsigned char global_haystack[4];
unsigned char global_needle[2];
unsigned char *global_search_result =
    std::search(global_haystack, global_haystack + 4, global_needle,
                global_needle + 2);

unsigned char *search_empty_needle(unsigned char *first1,
                                   unsigned char *last1,
                                   unsigned char *needle) {
  return std::search(first1, last1, needle, needle);
}
// CIR-LABEL: cir.func{{.*}} @_Z19search_empty_needlePhS_S_
// CIR: cir.cmp eq %{{.*}}, %{{.*}} : !cir.ptr<!u8i>
// CIR: cir.ternary
// CIR: cir.yield %{{.*}} : !cir.ptr<!u8i>
// CIR: cir.libc.memmem
// LLVM-LABEL: define{{.*}} ptr @_Z19search_empty_needlePhS_S_
// LLVM-NOT: @memmem
// LLVM: ret ptr %{{.*}}

unsigned char *search_short_haystack(unsigned char *haystack,
                                     unsigned char *needle) {
  return std::search(haystack, haystack + 1, needle, needle + 2);
}
// CIR-LABEL: cir.func{{.*}} @_Z21search_short_haystackPhS_
// CIR: cir.cmp gt %{{.*}}, %{{.*}} : !u64i
// CIR: cir.ternary
// CIR: cir.yield %{{.*}} : !cir.ptr<!u8i>
// CIR: cir.libc.memmem
// LLVM-LABEL: define{{.*}} ptr @_Z21search_short_haystackPhS_
// LLVM-NOT: @memmem
// LLVM: getelementptr i8, ptr %0, i64 1
// LLVM-NOT: @memmem
// LLVM: ret ptr

unsigned char *search(unsigned char *, unsigned char *, unsigned char *,
                      unsigned char *);
unsigned char *search_unmarked(unsigned char *first1, unsigned char *last1,
                               unsigned char *first2, unsigned char *last2) {
  return search(first1, last1, first2, last2);
}
// CIR-LABEL: cir.func{{.*}} @_Z15search_unmarkedPhS_S_S_
// CIR: cir.call @_Z6searchPhS_S_S_
// CIR-NOT: cir.libc.memmem

int *search_int(int *first1, int *last1, int *first2, int *last2) {
  return std::search(first1, last1, first2, last2);
}
// CIR-LABEL: cir.func{{.*}} @_Z10search_intPiS_S_S_
// CIR: cir.call @_ZSt6searchIPiS0_ET_S1_S1_T0_S2_
// CIR-NOT: cir.libc.memmem
// LLVM-LABEL: define internal void @_GLOBAL__sub_I_lib_opt_search.cpp
// LLVM: call ptr @memmem(ptr noundef nonnull @global_haystack, i64 noundef 4, ptr noundef nonnull @global_needle, i64 noundef 2)

#else
extern "C" unsigned char *
memmem(unsigned char *first1, unsigned char *last1, unsigned char *first2,
       unsigned char *last2) {
  return std::search(first1, last1, first2, last2);
}
// RECURSION-LABEL: cir.func{{.*}} @memmem
// RECURSION: cir.call @_ZSt6searchIPhS0_ET_S1_S1_T0_S2_
#endif
// RECURSION-NOT: cir.libc.memmem
