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
