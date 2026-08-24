// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DGNU_SHAPE -fclangir -emit-cir %s -o %t.gnu.cir
// RUN: FileCheck --input-file=%t.gnu.cir --check-prefix=GNU %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DMSVC_SHAPE -fclangir -emit-cir %s -o %t.msvc.cir
// RUN: FileCheck --input-file=%t.msvc.cir --check-prefix=MSVC %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DUSING_SHAPE -fclangir -emit-cir %s -o %t.using.cir
// RUN: FileCheck --input-file=%t.using.cir --check-prefix=USING %s

// Each RUN mocks one layout of the customization point object seen in a
// real standard library. The default shape nests the class in its own
// namespace with the object in an inline namespace, the historical
// libc++ layout. The MSVC standard library keeps the class in ranges
// and the object in an inline namespace, the structure current libc++
// shares since it removed the dedicated namespaces. libstdc++ keeps
// both directly in ranges. The last shape republishes the object
// through a using declaration.

#if defined(GNU_SHAPE)

namespace std {
struct identity {
  template <class T> T &&operator()(T &&t) const;
};
namespace ranges {
struct __find_fn {
  template <class Iter, class T, class Proj = identity>
  Iter operator()(Iter first, Iter last, const T &value, Proj proj = {}) const;
};
inline constexpr __find_fn find{};
}
}

#elif defined(MSVC_SHAPE)

namespace std {
struct identity {
  template <class T> T &&operator()(T &&t) const;
};
namespace ranges {
class _Find_fn {
public:
  template <class Iter, class T, class Proj = identity>
  Iter operator()(Iter first, Iter last, const T &value, Proj proj = {}) const;
};
inline namespace _Cpos {
inline constexpr _Find_fn find{};
}
}
}

#elif defined(USING_SHAPE)

namespace std {
struct identity {
  template <class T> T &&operator()(T &&t) const;
};
namespace ranges {
namespace __hidden {
struct __find_fn {
  template <class Iter, class T, class Proj = identity>
  Iter operator()(Iter first, Iter last, const T &value, Proj proj = {}) const;
};
inline constexpr __find_fn find{};
}
using __hidden::find;
}
}

#else

namespace std {
inline namespace __1 {
struct identity {
  template <class T> T &&operator()(T &&t) const;
};
struct contiguous_iterator_tag {};
template <class P> struct span_iter {
  P ptr;
  typedef contiguous_iterator_tag iterator_concept;
};
struct char_sentinel {};
struct char_range {
  char *b;
  char *e;
  char *begin() const;
  char *end() const;
};
namespace ranges {
namespace __find {
struct __fn {
  template <class Iter, class Sent, class T, class Proj = identity>
  Iter operator()(Iter first, Sent last, const T &value, Proj proj = {}) const;
  template <class Range, class T, class Proj = identity>
    requires requires(Range &&r) { r.begin(); r.end(); }
  auto operator()(Range &&r, const T &value, Proj proj = {}) const {
    return (*this)(r.begin(), r.end(), value, proj);
  }
  signed char *operator()(signed char *first, signed char *last,
                          const signed char &value) const;
};
}
inline namespace __cpo {
inline constexpr __find::__fn find{};
}
namespace __other {
struct __fn {
  template <class Iter, class T>
  Iter operator()(Iter first, Iter last, const T &value) const;
};
}
inline namespace __cpo {
inline constexpr __other::__fn mismatch{};
}
}
}
}

namespace myns {
namespace ranges {
struct __find_fn {
  template <class Iter, class T>
  Iter operator()(Iter first, Iter last, const T &value) const;
};
inline constexpr __find_fn find{};
}
}

#endif

char *ranges_find_eligible(char *first, char *last, const char &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z20ranges_find_eligible
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIPcS4_cNS_8identityEEET_S6_T0_RKT1_T2_({{.*}}) {cir.narrow_char_params}
// GNU-LABEL: cir.func{{.*}} @_Z20ranges_find_eligible
// GNU: cir.call @_ZNKSt6ranges9__find_fnclIPccSt8identityEET_S4_S4_RKT0_T1_({{.*}}) {cir.narrow_char_params}
// MSVC-LABEL: cir.func{{.*}} @_Z20ranges_find_eligible
// MSVC: cir.call @_ZNKSt6ranges8_Find_fnclIPccSt8identityEET_S4_S4_RKT0_T1_({{.*}}) {cir.narrow_char_params}
// USING-LABEL: cir.func{{.*}} @_Z20ranges_find_eligible
// USING: cir.call @_ZNKSt6ranges8__hidden9__find_fnclIPccSt8identityEET_S5_S5_RKT0_T1_({{.*}}) {cir.narrow_char_params}

int *ranges_find_wide(int *first, int *last, const int &value) {
  return std::ranges::find(first, last, value);
}
// An int of the target wchar_t width takes the wide marker, never the
// narrow one.
// CHECK-LABEL: cir.func{{.*}} @_Z16ranges_find_wide
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIPiS4_iNS_8identityEEET_S6_T0_RKT1_T2_({{.*}}) {cir.wide_char_params}
// CHECK-NOT: cir.narrow_char_params
// GNU-LABEL: cir.func{{.*}} @_Z16ranges_find_wide
// GNU: cir.call @_ZNKSt6ranges9__find_fnclIPiiSt8identityEET_S4_S4_RKT0_T1_({{.*}}) {cir.wide_char_params}
// GNU-NOT: cir.narrow_char_params
// MSVC-LABEL: cir.func{{.*}} @_Z16ranges_find_wide
// MSVC: cir.call @_ZNKSt6ranges8_Find_fnclIPiiSt8identityEET_S4_S4_RKT0_T1_({{.*}}) {cir.wide_char_params}
// MSVC-NOT: cir.narrow_char_params
// USING-LABEL: cir.func{{.*}} @_Z16ranges_find_wide
// USING: cir.call @_ZNKSt6ranges8__hidden9__find_fnclIPiiSt8identityEET_S5_S5_RKT0_T1_({{.*}}) {cir.wide_char_params}
// USING-NOT: cir.narrow_char_params

#if !defined(GNU_SHAPE) && !defined(MSVC_SHAPE) && !defined(USING_SHAPE)

char *ranges_find_wrapped(std::span_iter<char *> first,
                          std::span_iter<char *> last, const char &value) {
  return std::ranges::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z19ranges_find_wrapped
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclINS_9span_iterIPcEES6_cNS_8identityEEET_S8_T0_RKT1_T2_({{.*}}) {cir.narrow_char_params}

struct negate_proj {
  char operator()(char c) const;
};
char *ranges_find_projected(char *first, char *last, const char &value) {
  return std::ranges::find(first, last, value, negate_proj{});
}
// CHECK-LABEL: cir.func{{.*}} @_Z21ranges_find_projected
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIPcS4_c11negate_projEET_S6_T0_RKT1_T2_
// CHECK-NOT: cir.narrow_char_params

char *ranges_find_sentinel(char *first, std::char_sentinel last,
                           const char &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z20ranges_find_sentinel
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIPcNS_13char_sentinelEcNS_8identityEEET_S7_T0_RKT1_T2_
// CHECK-NOT: cir.narrow_char_params

char *ranges_find_whole_range(std::char_range r, const char &value) {
  return std::ranges::find(r, value);
}
// The whole range overload itself gets no marker, while the iterator
// pair call it delegates to inside its body does.
// CHECK-LABEL: cir.func{{.*}} @_Z23ranges_find_whole_range
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIRNS_10char_rangeEcNS_8identityE{{.*}}) : (
// CHECK: cir.func {{.*}}linkonce_odr @_ZNKSt3__16ranges6__find4__fnclIRNS_10char_rangeEcNS_8identityE
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclIPcS4_cNS_8identityEEET_S6_T0_RKT1_T2_({{.*}}) {cir.narrow_char_params}

char *ranges_other_cpo(char *first, char *last, const char &value) {
  return std::ranges::mismatch(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z16ranges_other_cpo
// CHECK: cir.call @_ZNKSt3__16ranges7__other4__fnclIPccEET_S5_S5_RKT0_
// CHECK-NOT: cir.narrow_char_params

char *ranges_find_outside_std(char *first, char *last, const char &value) {
  return myns::ranges::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z23ranges_find_outside_std
// CHECK: cir.call @_ZNK4myns6ranges9__find_fnclIPccEET_S4_S4_RKT0_
// CHECK-NOT: cir.narrow_char_params

signed char *ranges_find_three_params(signed char *first, signed char *last,
                                      const signed char &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z24ranges_find_three_params
// CHECK: cir.call @_ZNKSt3__16ranges6__find4__fnclEPaS3_RKa
// CHECK-NOT: cir.narrow_char_params

#endif
