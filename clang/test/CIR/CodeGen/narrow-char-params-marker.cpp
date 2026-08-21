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
