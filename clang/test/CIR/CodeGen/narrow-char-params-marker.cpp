// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

namespace std {
inline namespace __1 {
template <class Iter, class T> Iter find(Iter first, Iter last, const T &value);
}
}

char *eligible(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z8eligible
// CHECK: cir.call @_ZNSt3__14findIPccEET_S2_S2_RKT0_({{.*}}) {{{.*}}cir.narrow_char_params
// CHECK: cir.func private @_ZNSt3__14findIPccEET_S2_S2_RKT0_
// CHECK-NOT: cir.narrow_char_params

unsigned long unrelated(const char *s);
unsigned long calls_unrelated(const char *s) { return unrelated(s); }
// CHECK-LABEL: cir.func{{.*}} @_Z15calls_unrelated
// CHECK: cir.call @_Z9unrelatedPKc
// CHECK-NOT: cir.narrow_char_params

char8_t *char8_eligible(char8_t *first, char8_t *last, const char8_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z14char8_eligible
// CHECK: cir.call @_ZNSt3__14findIPDuDuEET_S2_S2_RKT0_({{.*}}) {{{.*}}cir.narrow_char_params

namespace std {
inline namespace __1 {
struct contiguous_iterator_tag {};
struct forward_iterator_tag {};
template <class P> struct span_iter {
  P ptr;
  typedef contiguous_iterator_tag iterator_concept;
};
template <class P> struct node_iter {
  P ptr;
  typedef forward_iterator_tag iterator_concept;
};
template <class P> struct fat_iter {
  P ptr;
  P end;
  typedef contiguous_iterator_tag iterator_concept;
};
}
}

namespace user {
template <class P> struct span_iter {
  P ptr;
  typedef std::contiguous_iterator_tag iterator_concept;
};
}

char *wrapped_eligible(std::span_iter<char *> first, std::span_iter<char *> last,
                       const char &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z16wrapped_eligible
// CHECK: cir.call @_ZNSt3__14findINS_9span_iterIPcEEcEET_S4_S4_RKT0_({{.*}}) {{{.*}}cir.narrow_char_params

int *wrapped_wide(std::span_iter<int *> first, std::span_iter<int *> last,
                  const int &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z12wrapped_wide
// CHECK: cir.call @_ZNSt3__14findINS_9span_iterIPiEEiEET_S4_S4_RKT0_
// CHECK-NOT: cir.narrow_char_params

char *wrapped_not_contiguous(std::node_iter<char *> first,
                             std::node_iter<char *> last, const char &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z22wrapped_not_contiguous
// CHECK: cir.call @_ZNSt3__14findINS_9node_iterIPcEEcEET_S4_S4_RKT0_
// CHECK-NOT: cir.narrow_char_params

char *wrapped_two_fields(std::fat_iter<char *> first, std::fat_iter<char *> last,
                         const char &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z18wrapped_two_fields
// CHECK: cir.call @_ZNSt3__14findINS_8fat_iterIPcEEcEET_S4_S4_RKT0_
// CHECK-NOT: cir.narrow_char_params

char *wrapped_outside_std(user::span_iter<char *> first,
                          user::span_iter<char *> last, const char &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z19wrapped_outside_std
// CHECK: cir.call @_ZNSt3__14findIN4user9span_iterIPcEEcEET_S5_S5_RKT0_
// CHECK-NOT: cir.narrow_char_params

namespace __gnu_cxx {
template <class P, class C> struct normal_iter {
  P ptr;
  typedef std::contiguous_iterator_tag iterator_concept;
};
}

char *gnu_wrapped_eligible(__gnu_cxx::normal_iter<char *, int> first,
                           __gnu_cxx::normal_iter<char *, int> last,
                           const char &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z20gnu_wrapped_eligible
// CHECK: cir.call @_ZNSt3__14findIN9__gnu_cxx11normal_iterIPciEEcEET_S5_S5_RKT0_({{.*}}) {{{.*}}cir.narrow_char_params

namespace std {
inline namespace __1 {
template <class Iter, class Pred> Iter find_if(Iter first, Iter last, Pred pred) {
  for (; first != last; ++first)
    if (pred(*first))
      break;
  return first;
}
template <class Iter, class Pred> Iter find_if_not(Iter first, Iter last, Pred pred) {
  for (; first != last; ++first)
    if (!pred(*first))
      break;
  return first;
}
}
}

char *find_if_ref_capture(char *first, char *last, const char &value) {
  return std::find_if(first, last, [&](char element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z19find_if_ref_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ19find_if_ref_capture{{.*}} {{{.*}}cir.byte_eq_pred

char *find_if_value_capture(char *first, char *last, char value) {
  return std::find_if(first, last, [value](char element) { return value == element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21find_if_value_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ21find_if_value_capture{{.*}} {{{.*}}cir.byte_eq_pred

char *find_if_not_ne(char *first, char *last, const char &value) {
  return std::find_if_not(first, last, [&](char element) { return element != value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z14find_if_not_ne
// CHECK: cir.call @_ZNSt3__111find_if_notIPcZ14find_if_not_ne{{.*}} {{{.*}}cir.byte_eq_pred

char *find_if_wrong_polarity(char *first, char *last, const char &value) {
  return std::find_if(first, last, [&](char element) { return element != value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z22find_if_wrong_polarity
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ22find_if_wrong_polarity
// CHECK-NOT: cir.byte_eq_pred

char *find_if_two_captures(char *first, char *last, const char &value, const char &other) {
  return std::find_if(first, last,
                      [&](char element) { return element == value && element == other; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20find_if_two_captures
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ20find_if_two_captures
// CHECK-NOT: cir.byte_eq_pred

char *find_if_wide_capture(char *first, char *last, const int &value) {
  return std::find_if(first, last, [&](char element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20find_if_wide_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ20find_if_wide_capture
// CHECK-NOT: cir.byte_eq_pred

int side_channel;
char *find_if_side_effect(char *first, char *last, const char &value) {
  return std::find_if(first, last, [&](char element) {
    ++side_channel;
    return element == value;
  });
}
// CHECK-LABEL: cir.func{{.*}} @_Z19find_if_side_effect
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ19find_if_side_effect
// CHECK-NOT: cir.byte_eq_pred

char *find_if_generic_lambda(char *first, char *last, const char &value) {
  return std::find_if(first, last, [&](auto element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z22find_if_generic_lambda
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ22find_if_generic_lambda{{.*}} {{{.*}}cir.byte_eq_pred

char *find_if_mutable_lambda(char *first, char *last, char value) {
  return std::find_if(first, last,
                      [value](char element) mutable { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z22find_if_mutable_lambda
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ22find_if_mutable_lambda{{.*}} {{{.*}}cir.byte_eq_pred

char *find_if_init_capture(char *first, char *last, const char &value) {
  return std::find_if(first, last,
                      [v = value](char element) { return element == v; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20find_if_init_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ20find_if_init_capture{{.*}} {{{.*}}cir.byte_eq_pred

struct EqFunctor {
  char value;
  bool operator()(char element) const { return element == value; }
};
char *find_if_functor(char *first, char *last, char value) {
  return std::find_if(first, last, EqFunctor{value});
}
// CHECK-LABEL: cir.func{{.*}} @_Z15find_if_functor
// CHECK: cir.call @_ZNSt3__17find_ifIPc9EqFunctorEET_S3_S3_T0_
// CHECK-NOT: cir.byte_eq_pred

namespace user_gnu {
namespace __gnu_cxx {
template <class P> struct nested_iter {
  P ptr;
  typedef std::contiguous_iterator_tag iterator_concept;
};
}
}

char *wrapped_nested_gnu(user_gnu::__gnu_cxx::nested_iter<char *> first,
                         user_gnu::__gnu_cxx::nested_iter<char *> last,
                         const char &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z18wrapped_nested_gnu
// CHECK: cir.call @_ZNSt3__14findIN8user_gnu9__gnu_cxx11nested_iterIPcEEcEET_S6_S6_RKT0_
// CHECK-NOT: cir.narrow_char_params
