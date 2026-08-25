// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2> %t.raised.cir
// RUN: FileCheck %s --check-prefix=RAISED --input-file=%t.raised.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.final.cir
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.final.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir -mmlir -mlir-pass-statistics -mmlir -mlir-pass-statistics-display=list %s -o /dev/null 2> %t.stats
// RUN: FileCheck %s --check-prefix=STATS --input-file=%t.stats
// STATS: (S) 5 equal-deque-to-memcmp
// STATS: (S) 5 mismatch-deque-to-memcmp

namespace std {

constexpr unsigned long __deque_buf_size(unsigned long size) {
  return size < 512 ? 512 / size : 1;
}

template <class Type, class Reference, class Pointer>
struct _Deque_iterator {
  static unsigned long _S_buffer_size() {
    return __deque_buf_size(sizeof(Type));
  }

  Pointer _M_cur;
  Pointer _M_first;
  Pointer _M_last;
  Type **_M_node;
};

inline namespace __1 {

template <class Type, class Pointer, class Reference, class MapPointer,
          class Difference, unsigned long BlockSize>
struct __deque_iterator {
  MapPointer __m_iter_;
  Pointer __ptr_;
};

}

template <class First, class Second> struct pair {
  First first;
  Second second;
};

template <class First, class Second>
bool equal(First first1, First last1, Second first2);

template <class First, class Second, class Predicate>
bool equal(First first1, First last1, Second first2, Predicate predicate) {
  return true;
}

template <class First, class Second>
pair<First, Second> mismatch(First first1, First last1, Second first2);

template <class First, class Second>
pair<First, Second> mismatch(First first1, First last1, Second first2,
                             Second last2);

template <class First, class Second, class Predicate>
pair<First, Second> mismatch(First first1, First last1, Second first2,
                             Predicate predicate) {
  return {first1, first2};
}

}

using GnuChar = std::_Deque_iterator<char, char &, char *>;
using GnuInt = std::_Deque_iterator<int, int &, int *>;
using LibcxxChar =
    std::__deque_iterator<char, char *, char &, char **, long, 4096>;
using LibcxxInt = std::__deque_iterator<int, int *, int &, int **, long, 1024>;

bool equal_gnu(GnuChar first1, GnuChar last1, GnuChar first2) {
  return std::equal(first1, last1, first2);
}

bool equal_libcxx(LibcxxChar first1, LibcxxChar last1, LibcxxChar first2) {
  return std::equal(first1, last1, first2);
}

std::pair<GnuInt, GnuInt>
mismatch_gnu(GnuInt first1, GnuInt last1, GnuInt first2) {
  return std::mismatch(first1, last1, first2);
}

std::pair<LibcxxInt, LibcxxInt>
mismatch_libcxx(LibcxxInt first1, LibcxxInt last1, LibcxxInt first2) {
  return std::mismatch(first1, last1, first2);
}

bool equal_mixed(GnuChar first1, GnuChar last1, char *first2) {
  return std::equal(first1, last1, first2);
}

std::pair<char *, LibcxxChar>
mismatch_mixed(char *first1, char *last1, LibcxxChar first2) {
  return std::mismatch(first1, last1, first2);
}

bool equal_pred(GnuChar first1, GnuChar last1, GnuChar first2) {
  return std::equal(first1, last1, first2,
                    [](char left, char right) { return left == right; });
}

std::pair<LibcxxChar, LibcxxChar>
mismatch_pred(LibcxxChar first1, LibcxxChar last1, LibcxxChar first2) {
  return std::mismatch(
      first1, last1, first2,
      [](char left, char right) { return left == right; });
}
bool equal_wide(GnuInt first1, GnuInt last1, GnuInt first2) {
  return std::equal(first1, last1, first2);
}

std::pair<LibcxxChar, LibcxxChar>
mismatch_bounded(LibcxxChar first1, LibcxxChar last1, LibcxxChar first2,
                 LibcxxChar last2) {
  return std::mismatch(first1, last1, first2, last2);
}

// RAISED-LABEL: @_Z9equal_gnu
// RAISED: cir.std.equal
// RAISED-SAME: cir.narrow_char_params
// FINAL-LABEL: @_Z9equal_gnu
// FINAL: cir.alloca "compare_deque_first_map"
// FINAL: cir.alloca "compare_deque_second_map"
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.equal

// RAISED-LABEL: @_Z12equal_libcxx
// RAISED: cir.std.equal
// RAISED-SAME: cir.narrow_char_params
// FINAL-LABEL: @_Z12equal_libcxx
// FINAL: cir.alloca "compare_deque_first_map"
// FINAL: cir.alloca "compare_deque_second_map"
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.equal

// RAISED-LABEL: @_Z12mismatch_gnu
// RAISED: cir.std.mismatch
// RAISED-SAME: cir.wide_char_params
// FINAL-LABEL: @_Z12mismatch_gnu
// FINAL: cir.alloca "compare_deque_first_map"
// FINAL: cir.alloca "compare_deque_second_map"
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.mismatch

// RAISED-LABEL: @_Z15mismatch_libcxx
// RAISED: cir.std.mismatch
// RAISED-SAME: cir.wide_char_params
// FINAL-LABEL: @_Z15mismatch_libcxx
// FINAL: cir.alloca "compare_deque_first_map"
// FINAL: cir.alloca "compare_deque_second_map"
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.mismatch

// RAISED-LABEL: @_Z11equal_mixed
// RAISED: cir.std.equal
// RAISED-SAME: cir.narrow_char_params
// FINAL-LABEL: @_Z11equal_mixed
// FINAL: cir.alloca "compare_deque_first_map"
// FINAL-NOT: cir.alloca "compare_deque_second_map"
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.equal

// RAISED-LABEL: @_Z14mismatch_mixed
// RAISED: cir.std.mismatch
// RAISED-SAME: cir.narrow_char_params
// FINAL-LABEL: @_Z14mismatch_mixed
// FINAL-NOT: cir.alloca "compare_deque_first_map"
// FINAL: cir.alloca "compare_deque_second_map"
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.mismatch

// RAISED-LABEL: @_Z10equal_pred
// RAISED: cir.std.equal_pred
// RAISED-SAME: cir.elem_eq_binary_pred
// FINAL-LABEL: @_Z10equal_pred
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.equal_pred

// RAISED-LABEL: @_Z13mismatch_pred
// RAISED: cir.std.mismatch_pred
// RAISED-SAME: cir.elem_eq_binary_pred
// FINAL-LABEL: @_Z13mismatch_pred
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.mismatch_pred

// RAISED-LABEL: @_Z10equal_wide
// RAISED: cir.std.equal
// RAISED-SAME: cir.wide_char_params
// FINAL-LABEL: @_Z10equal_wide
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.equal

// RAISED-LABEL: @_Z16mismatch_bounded
// RAISED: cir.std.mismatch_bounded
// RAISED-SAME: cir.narrow_char_params
// FINAL-LABEL: @_Z16mismatch_bounded
// FINAL: cir.libc.memcmp
// FINAL-NOT: cir.std.mismatch_bounded
