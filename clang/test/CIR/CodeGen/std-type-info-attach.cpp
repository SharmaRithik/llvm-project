// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-passes -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DGNU_SHAPE -fclangir -clangir-disable-passes -emit-cir %s -o %t.gnu.cir
// RUN: FileCheck --input-file=%t.gnu.cir --check-prefix=GNU %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DPARTIAL_SHAPE -fclangir -clangir-disable-passes -emit-cir %s -o %t.partial.cir
// RUN: FileCheck --input-file=%t.partial.cir --check-prefix=PARTIAL %s

// Each RUN mocks one real standard library layout of std::vector. The
// default shape nests the bounds in a layout record like current libc++
// and the GNU shape keeps them in a base class chain like libstdc++. The
// identity is attached by field name, so each shape proves its own path,
// and the wide element instantiation gets none.
// CHECK: std_type_info = #cir.std_type_info<"std::vector", !s8i, roles {begin = array<i32: 0, 0>, end = array<i32: 0, 1>}>
// CHECK-NOT: #cir.std_type_info<"std::vector", !s32i
// GNU: std_type_info = #cir.std_type_info<"std::vector", !s8i, roles {begin = array<i32: 0, 0, 0, 0>, end = array<i32: 0, 0, 0, 1>}>
// GNU-NOT: #cir.std_type_info<"std::vector", !s32i
// PARTIAL-NOT: std_type_info

namespace std {
inline namespace __1 {

#if defined(PARTIAL_SHAPE)

// A program may specialize std::vector for a program defined type, and
// such a specialization owes the primary template's layout nothing, so
// only a primary instantiation gets the identity even when the field
// names match.
struct my_alloc {};
template <class T, class A = my_alloc> class vector;
template <class T> class vector<T, my_alloc> {
  T *__begin_;
  T *__end_;
  T *__capacity_;
};

#elif defined(GNU_SHAPE)

template <class T> struct _Vector_impl_data {
  T *_M_start;
  T *_M_finish;
  T *_M_end_of_storage;
};
template <class T> struct _Vector_base {
  struct _Vector_impl : _Vector_impl_data<T> {};
  _Vector_impl _M_impl;
};
template <class T> class vector : protected _Vector_base<T> {};

#else

template <class T> struct __vector_layout {
  T *__begin_;
  T *__end_;
  T *__capacity_;
};
template <class T> class vector {
  __vector_layout<T> __layout_;
};

#endif

}
}

unsigned long sizes(std::vector<char> &c, std::vector<int> &w) {
  return sizeof(c) + sizeof(w);
}
