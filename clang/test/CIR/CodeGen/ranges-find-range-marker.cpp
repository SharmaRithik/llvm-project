// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DGNU_SHAPE -fclangir -emit-cir %s -o %t.gnu.cir
// RUN: FileCheck --input-file=%t.gnu.cir --check-prefix=GNU %s

// Each RUN mocks one real standard library layout of std::vector, the
// same shapes std-type-info-attach.cpp pins the attached paths for, and
// checks the whole range call gets the marker only over the identified
// container of the searched character.

namespace std {
inline namespace __1 {
struct identity {
  template <class T> T &&operator()(T &&t) const;
};
template <class T> struct remove_ref {
  typedef T type;
};
template <class T> struct remove_ref<T &> {
  typedef T type;
};

#if defined(GNU_SHAPE)

template <class T> struct _Vector_impl_data {
  T *_M_start;
  T *_M_finish;
  T *_M_end_of_storage;
};
template <class T> struct _Vector_base {
  struct _Vector_impl : _Vector_impl_data<T> {};
  _Vector_impl _M_impl;
};
template <class T> class vector : protected _Vector_base<T> {
public:
  typedef T *iterator;
};

#else

template <class T> struct __vector_layout {
  T *__begin_;
  T *__end_;
  T *__capacity_;
};
template <class T> class vector {
  __vector_layout<T> __layout_;

public:
  typedef T *iterator;
};

#endif

namespace ranges {
namespace __find {
struct __fn {
  template <class Range, class T, class Proj = identity>
  typename remove_ref<Range>::type::iterator
  operator()(Range &&r, const T &value, Proj proj = {}) const;
};
}
inline namespace __cpo {
inline constexpr __find::__fn find{};
}
}
}
}

char *range_find_eligible(std::vector<char> &v, const char &value) {
  return std::ranges::find(v, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z19range_find_eligible
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_6vectorIcEEcNS_8identityEEENS_10remove_refIT_E4type8iteratorEOS9_RKT0_T1_({{.*}}) {cir.narrow_char_params}
// GNU-LABEL: cir.func{{.*}} @_Z19range_find_eligible
// GNU: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_6vectorIcEEcNS_8identityEEENS_10remove_refIT_E4type8iteratorEOS9_RKT0_T1_({{.*}}) {cir.narrow_char_params}

int *range_find_wide(std::vector<int> &v, const int &value) {
  return std::ranges::find(v, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z15range_find_wide
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_6vectorIiEEiNS_8identityEEENS_10remove_refIT_E4type8iteratorEOS9_RKT0_T1_
// CHECK-NOT: cir.narrow_char_params
// GNU-LABEL: cir.func{{.*}} @_Z15range_find_wide
// GNU: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_6vectorIiEEiNS_8identityEEENS_10remove_refIT_E4type8iteratorEOS9_RKT0_T1_
// GNU-NOT: cir.narrow_char_params

#if !defined(GNU_SHAPE)

struct negate_proj {
  char operator()(char c) const;
};
char *range_find_projected(std::vector<char> &v, const char &value) {
  return std::ranges::find(v, value, negate_proj{});
}
// CHECK-LABEL: cir.func{{.*}} @_Z20range_find_projected
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_6vectorIcEEc11negate_projEENS_10remove_refIT_E4type8iteratorEOS9_RKT0_T1_
// CHECK-NOT: cir.narrow_char_params

namespace myns {
template <class T> class vector {
  T *__begin_;
  T *__end_;
  T *__capacity_;

public:
  typedef T *iterator;
};
}

char *range_find_outside_std(myns::vector<char> &v, const char &value) {
  return std::ranges::find(v, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z22range_find_outside_std
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIRN4myns6vectorIcEEcNS_8identityEEENS_10remove_refIT_E4type8iteratorEOSA_RKT0_T1_
// CHECK-NOT: cir.narrow_char_params

namespace std {
inline namespace __1 {
template <> class vector<signed char> {
  signed char *__data_;
  unsigned long __count_;

public:
  typedef signed char *iterator;
};
}
}

signed char *range_find_unknown_fields(std::vector<signed char> &v,
                                       const signed char &value) {
  return std::ranges::find(v, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z25range_find_unknown_fields
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_6vectorIaEEaNS_8identityEEENS_10remove_refIT_E4type8iteratorEOS9_RKT0_T1_
// CHECK-NOT: cir.narrow_char_params

#endif
