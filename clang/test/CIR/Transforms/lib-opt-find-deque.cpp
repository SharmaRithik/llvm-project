// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2> %t.raised.cir
// RUN: FileCheck %s --check-prefix=RAISED --input-file=%t.raised.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.final.cir
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.final.cir

namespace std {

constexpr unsigned long __deque_buf_size(unsigned long size) {
  return size < 512 ? 512 / size : 1;
}

template <class Type, class Reference, class Pointer>
struct _Deque_iterator {
  using element_pointer = Type *;
  using map_pointer = Type **;

  static unsigned long _S_buffer_size() {
    return __deque_buf_size(sizeof(Type));
  }

  element_pointer _M_cur;
  element_pointer _M_first;
  element_pointer _M_last;
  map_pointer _M_node;
};

template <class Type> struct deque {
  using iterator = _Deque_iterator<Type, Type &, Type *>;

  iterator begin();
  iterator end();
};

template <class Iterator, class Type>
Iterator find(Iterator first, Iterator last, const Type &value);

template <class Iterator, class Predicate>
Iterator find_if(Iterator first, Iterator last, Predicate predicate) {
  return first;
}

}

std::deque<char>::iterator find_char(std::deque<char> &values,
                                     const char &value) {
  return std::find(values.begin(), values.end(), value);
}
// RAISED-LABEL: @_Z9find_charRSt5dequeIcERKc
// RAISED: cir.std.find
// RAISED-SAME: cir.narrow_char_params
// FINAL-LABEL: @_Z9find_charRSt5dequeIcERKc
// FINAL: cir.libc.memchr
// FINAL: cir.while {
// FINAL-NOT: cir.call @_ZSt4find


std::deque<wchar_t>::iterator find_wide(std::deque<wchar_t> &values,
                                        const wchar_t &value) {
  return std::find(values.begin(), values.end(), value);
}

// RAISED-LABEL: @_Z9find_wideRSt5dequeIwERKw
// RAISED: cir.std.find
// RAISED-SAME: cir.wide_char_params
// FINAL-LABEL: @_Z9find_wideRSt5dequeIwERKw
// FINAL: cir.libc.wmemchr
// FINAL-NOT: cir.call @_ZSt4find

std::deque<char>::iterator find_char_if(std::deque<char> &values,
                                        const char &value) {
  return std::find_if(values.begin(), values.end(),
                      [&](char element) { return element == value; });
}
// RAISED-LABEL: @_Z12find_char_ifRSt5dequeIcERKc
// RAISED: cir.std.find_if
// RAISED-SAME: cir.byte_eq_pred
// FINAL-LABEL: @_Z12find_char_ifRSt5dequeIcERKc
// FINAL: cir.libc.memchr
// FINAL-NOT: cir.call @_ZSt7find_if


std::deque<short>::iterator keep_short(std::deque<short> &values,
                                       const short &value) {
  return std::find(values.begin(), values.end(), value);
}
// RAISED-LABEL: @_Z10keep_shortRSt5dequeIsERKs
// RAISED: cir.std.find
// RAISED-NOT: char_params
// FINAL-LABEL: @_Z10keep_shortRSt5dequeIsERKs
// FINAL: cir.call @_ZSt4find
// FINAL-NOT: cir.libc.memchr

