// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.x86.cir
// RUN: FileCheck %s --input-file=%t.x86.cir --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-llvm -disable-llvm-passes %s -o %t.x86.ll
// RUN: FileCheck %s --input-file=%t.x86.ll --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -ffreestanding -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.free.cir
// RUN: FileCheck %s --input-file=%t.free.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memcmp

// The loop carries its offset in an alloca, which has no home in a global
// initializer region, so the global case keeps the call and only function
// bodies rewrite.
// CIR-LABEL: cir.func internal private @__cxx_global_var_init
// CIR: cir.call @_ZSt8mismatchIPhS0_ESt4pairIT_T0_ES2_S2_S3_({{.*}}) {cir.narrow_char_params}

// CIR-LABEL: cir.func dso_local @_Z13byte_mismatchPhS_S_
// CIR: %{{[0-9]+}} = cir.ptr_diff %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!u8i> -> !u64i
// CIR: %{{[0-9]+}} = cir.alloca "mismatch_off"
// CIR: cir.while {
// CIR: cir.ternary
// CIR: %{{[0-9]+}} = cir.libc.memcmp(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.ptr<!void>, !cir.ptr<!void>, !u64i -> !s32i
// CIR: cir.condition
// CIR: cir.while {
// CIR: cir.condition
// CIR: %{{[0-9]+}} = cir.insert_member %{{[0-9]+}}[0], %{{[0-9]+}} : !rec_std3A3Apair3Cunsigned_char_2A2C_unsigned_char_2A3E, !cir.ptr<!u8i>
// CIR: %{{[0-9]+}} = cir.insert_member %{{[0-9]+}}[1], %{{[0-9]+}} : !rec_std3A3Apair3Cunsigned_char_2A2C_unsigned_char_2A3E, !cir.ptr<!u8i>
// CIR-NOT: cir.std.mismatch

// CIR-LABEL: cir.func dso_local @_Z18byte_mismatch_predPhS_S_
// CIR: cir.while {
// CIR: cir.libc.memcmp
// CIR: cir.while {
// CIR: cir.insert_member %{{[0-9]+}}[1]
// CIR-NOT: cir.std.mismatch

// CIR-LABEL: cir.func dso_local @_Z12int_mismatchPiS_S_
// CIR: cir.call @_ZSt8mismatchIPiS0_ESt4pairIT_T0_ES2_S2_S3_
// CIR-NOT: cir.libc.memcmp

// The bounded operation has no LibOpt rewrite and lowers back to the call
// CIR-LABEL: cir.func dso_local @_Z13four_iteratorSt11__wrap_iterIcES0_S0_S0_
// CIR: cir.call @_ZSt8mismatchISt11__wrap_iterIcES1_ESt4pairIT_T0_ES3_S3_S4_S4_({{.*}}) {cir.narrow_char_params}
// CIR-NOT: cir.libc.memcmp

// LLVM: declare i32 @memcmp(ptr noundef, ptr noundef, i64 noundef)
// LLVM-LABEL: define dso_local {{.*}}ptr @_Z13byte_mismatchPhS_S_
// LLVM: call i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}})
// LLVM-LABEL: define dso_local {{.*}}ptr @_Z18byte_mismatch_predPhS_S_
// LLVM: call i32 @memcmp(ptr noundef %{{[0-9]+}}, ptr noundef %{{[0-9]+}}, i64 noundef %{{[0-9]+}})

// NOXFORM-LABEL: cir.func dso_local @_Z13byte_mismatchPhS_S_
// NOXFORM: cir.call @_ZSt8mismatchIPhS0_ESt4pairIT_T0_ES2_S2_S3_
// NOXFORM-LABEL: cir.func dso_local @_Z18byte_mismatch_predPhS_S_
// NOXFORM: cir.call @_ZSt8mismatchIPhS0_

namespace std {
struct contiguous_iterator_tag {};
template <class T> struct __wrap_iter {
  typedef std::contiguous_iterator_tag iterator_concept;
  T *ptr;
  T &operator*() const { return *ptr; }
  __wrap_iter &operator++() {
    ++ptr;
    return *this;
  }
  bool operator!=(__wrap_iter o) const { return ptr != o.ptr; }
  bool operator==(__wrap_iter o) const { return ptr == o.ptr; }
};
template <class T1, class T2> struct pair {
  T1 first;
  T2 second;
};
template <class Iter1, class Iter2>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2);
template <class Iter1, class Iter2, class Pred>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Pred pred) {
  for (; first1 != last1; ++first1, ++first2)
    if (!pred(*first1, *first2))
      break;
  return {first1, first2};
}
}

unsigned char *byte_mismatch(unsigned char *first1, unsigned char *last1,
                             unsigned char *first2) {
  return std::mismatch(first1, last1, first2).first;
}

unsigned char *byte_mismatch_pred(unsigned char *first1,
                                  unsigned char *last1,
                                  unsigned char *first2) {
  return std::mismatch(first1, last1, first2,
                       [](auto a, auto b) { return a == b; })
      .first;
}

int *int_mismatch(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2).first;
}

namespace std {
template <class Iter1, class Iter2>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Iter2 last2) {
  for (; first1 != last1 && first2 != last2; ++first1, ++first2)
    if (!(*first1 == *first2))
      break;
  return {first1, first2};
}
}

bool four_iterator(std::__wrap_iter<char> f1, std::__wrap_iter<char> l1,
                   std::__wrap_iter<char> f2, std::__wrap_iter<char> l2) {
  auto r = std::mismatch(f1, l1, f2, l2);
  return r.first == l1;
}

extern unsigned char storage_a[4];
extern unsigned char storage_b[4];
unsigned char *global_mismatch_result =
    std::mismatch(storage_a, storage_a + 4, storage_b).first;
