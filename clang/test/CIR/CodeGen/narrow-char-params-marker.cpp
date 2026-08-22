// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple aarch64-unknown-linux-gnu -fno-signed-char -fclangir -emit-cir %s -o %t.uchar.cir
// RUN: FileCheck --input-file=%t.uchar.cir --check-prefix=UCHAR %s

namespace std {
inline namespace __1 {
template <class Iter, class T> Iter find(Iter first, Iter last, const T &value);
template <class Iter1, class Iter2>
Iter1 search(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);
template <class Iter1, class Iter2>
bool equal(Iter1 first1, Iter1 last1, Iter2 first2);
template <class Iter1, class Iter2>
bool equal(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);
template <class Iter1, class Iter2, class Pred>
bool equal(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred);
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

bool equal_char(char *first1, char *last1, char *first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z10equal_char
// CHECK: cir.call @_ZNSt3__15equalIPcS1_EEbT_S2_T0_({{.*}}) {{{.*}}cir.narrow_char_params

bool equal_unsigned_char(unsigned char *first1, unsigned char *last1,
                         unsigned char *first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z19equal_unsigned_char
// CHECK: cir.call @_ZNSt3__15equalIPhS1_EEbT_S2_T0_({{.*}}) {{{.*}}cir.narrow_char_params

bool equal_int(int *first1, int *last1, int *first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z9equal_int
// CHECK: cir.call @_ZNSt3__15equalIPiS1_EEbT_S2_T0_
// CHECK-NOT: cir.narrow_char_params

bool equal_mixed(char *first1, char *last1, unsigned char *first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z11equal_mixed
// CHECK: cir.call @_ZNSt3__15equalIPcPhEEbT_S3_T0_
// CHECK-NOT: cir.narrow_char_params

bool equal_volatile(volatile char *first1, volatile char *last1,
                    volatile char *first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z14equal_volatile
// CHECK: cir.call @_ZNSt3__15equalIPVcS2_EEbT_S3_T0_
// CHECK-NOT: cir.narrow_char_params

bool equal_wide(wchar_t *first1, wchar_t *last1, wchar_t *first2) {
  return std::equal(first1, last1, first2);
}

bool equal_four_ranges(char *first1, char *last1, char *first2, char *last2) {
  return std::equal(first1, last1, first2, last2);
}

struct EqualPredicate {
  char state;
  bool operator()(char lhs, char rhs) const;
};

bool equal_predicate(char *first1, char *last1, char *first2,
                     EqualPredicate pred) {
  return std::equal(first1, last1, first2, pred);
}
// CHECK-LABEL: cir.func{{.*}} @_Z10equal_wide
// CHECK: cir.call @_ZNSt3__15equalIPwS1_EEbT_S2_T0_
// CHECK-NOT: cir.narrow_char_params

// CHECK-LABEL: cir.func{{.*}} @_Z17equal_four_ranges
// CHECK: cir.call @_ZNSt3__15equalIPcS1_EEbT_S2_T0_S3_({{.*}}) {cir.narrow_char_params}
// CHECK-NOT: cir.std.equal

// CHECK-LABEL: cir.func{{.*}} @_Z15equal_predicate
// CHECK: cir.call @_ZNSt3__15equalIPcS1_14EqualPredicateEEbT_S3_T0_T1_
// CHECK-NOT: cir.narrow_char_params
// CHECK-NOT: cir.std.equal

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

bool equal_wrapped(std::span_iter<char *> first1,
                   std::span_iter<char *> last1,
                   std::span_iter<char *> first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z13equal_wrapped
// CHECK: cir.call @_ZNSt3__15equalINS_9span_iterIPcEES3_EEbT_S4_T0_({{.*}}) {{{.*}}cir.narrow_char_params

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

bool equal_gnu_wrapped(__gnu_cxx::normal_iter<unsigned char *, int> first1,
                       __gnu_cxx::normal_iter<unsigned char *, int> last1,
                       __gnu_cxx::normal_iter<unsigned char *, int> first2) {
  return std::equal(first1, last1, first2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z17equal_gnu_wrapped
// CHECK: cir.call @_ZNSt3__15equalIN9__gnu_cxx11normal_iterIPhiEES4_EEbT_S5_T0_({{.*}}) {{{.*}}cir.narrow_char_params

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
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ19find_if_ref_capture{{.*}} {{{.*}}cir.byte_eq_pred}

char *find_if_value_capture(char *first, char *last, char value) {
  return std::find_if(first, last, [value](char element) { return value == element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21find_if_value_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ21find_if_value_capture{{.*}} {{{.*}}cir.byte_eq_pred}
// CHECK-NOT: cir.byte_eq_pred_value

char *find_if_not_ne(char *first, char *last, const char &value) {
  return std::find_if_not(first, last, [&](char element) { return element != value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z14find_if_not_ne
// CHECK: cir.call @_ZNSt3__111find_if_notIPcZ14find_if_not_ne{{.*}} {{{.*}}cir.byte_eq_pred}

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
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ22find_if_generic_lambda{{.*}} {{{.*}}cir.byte_eq_pred}

char *find_if_mutable_lambda(char *first, char *last, char value) {
  return std::find_if(first, last,
                      [value](char element) mutable { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z22find_if_mutable_lambda
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ22find_if_mutable_lambda{{.*}} {{{.*}}cir.byte_eq_pred}

char *find_if_init_capture(char *first, char *last, const char &value) {
  return std::find_if(first, last,
                      [v = value](char element) { return element == v; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20find_if_init_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ20find_if_init_capture{{.*}} {{{.*}}cir.byte_eq_pred}

char *find_if_literal_char(char *first, char *last) {
  return std::find_if(first, last,
                      [](char element) { return element == 'x'; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20find_if_literal_charPcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ20find_if_literal_char{{.*}} {cir.byte_eq_pred_value = #cir.int<120> : !s8i}

unsigned char *find_if_not_literal_unsigned(unsigned char *first,
                                             unsigned char *last) {
  return std::find_if_not(
      first, last, [](unsigned char element) { return element != 0x80; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z28find_if_not_literal_unsignedPhS_
// CHECK: cir.call @_ZNSt3__111find_if_notIPhZ28find_if_not_literal_unsigned{{.*}} {cir.byte_eq_pred_value = #cir.int<128> : !u8i}

signed char *find_if_literal_negative(signed char *first, signed char *last) {
  return std::find_if(first, last,
                      [](signed char element) { return element == -128; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z24find_if_literal_negativePaS_
// CHECK: cir.call @_ZNSt3__17find_ifIPaZ24find_if_literal_negative{{.*}} {cir.byte_eq_pred_value = #cir.int<-128> : !s8i}

unsigned char *find_if_literal_generic(unsigned char *first,
                                       unsigned char *last) {
  return std::find_if(first, last,
                      [](auto element) { return 0x7f == element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z23find_if_literal_genericPhS_
// CHECK: cir.call @_ZNSt3__17find_ifIPhZ23find_if_literal_generic{{.*}} {cir.byte_eq_pred_value = #cir.int<127> : !u8i}

char *find_if_literal_mutable(char *first, char *last) {
  return std::find_if(
      first, last, [](char element) mutable { return element == 'm'; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z23find_if_literal_mutablePcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ23find_if_literal_mutable{{.*}} {cir.byte_eq_pred_value = #cir.int<109> : !s8i}

char *find_if_literal_wrong_polarity(char *first, char *last) {
  return std::find_if(first, last,
                      [](char element) { return element != 'x'; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z30find_if_literal_wrong_polarityPcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ30find_if_literal_wrong_polarity
// CHECK-NOT: cir.byte_eq_pred_value

unsigned char *find_if_literal_negative_unsigned(unsigned char *first,
                                                  unsigned char *last) {
  return std::find_if(first, last,
                      [](unsigned char element) { return element == -1; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z33find_if_literal_negative_unsignedPhS_
// CHECK: cir.call @_ZNSt3__17find_ifIPhZ33find_if_literal_negative_unsigned
// CHECK-NOT: cir.byte_eq_pred_value

unsigned char *find_if_literal_out_of_range(unsigned char *first,
                                            unsigned char *last) {
  return std::find_if(first, last,
                      [](unsigned char element) { return element == 256; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z28find_if_literal_out_of_rangePhS_
// CHECK: cir.call @_ZNSt3__17find_ifIPhZ28find_if_literal_out_of_range
// CHECK-NOT: cir.byte_eq_pred_value

char literal_call();
char *find_if_literal_two_calls(char *first, char *last) {
  return std::find_if(first, last, [](char) {
    return literal_call() == literal_call();
  });
}
// CHECK-LABEL: cir.func{{.*}} @_Z25find_if_literal_two_callsPcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ25find_if_literal_two_calls
// CHECK-NOT: cir.byte_eq_pred_value

constexpr char kSentinelChar = 'q';
char *find_if_literal_constexpr_var(char *first, char *last) {
  return std::find_if(first, last,
                      [](char element) { return element == kSentinelChar; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z29find_if_literal_constexpr_varPcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ29find_if_literal_constexpr_var{{.*}} {cir.byte_eq_pred_value = #cir.int<113> : !s8i}

enum ByteMarker : unsigned char { kByteMarker = 0x7e };
unsigned char *find_if_literal_enum(unsigned char *first,
                                    unsigned char *last) {
  return std::find_if(first, last, [](unsigned char element) {
    return element == kByteMarker;
  });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20find_if_literal_enumPhS_
// CHECK: cir.call @_ZNSt3__17find_ifIPhZ20find_if_literal_enum{{.*}} {cir.byte_eq_pred_value = #cir.int<126> : !u8i}

char8_t *find_if_literal_char8(char8_t *first, char8_t *last) {
  return std::find_if(first, last,
                      [](char8_t element) { return element == u8'z'; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21find_if_literal_char8PDuS_
// CHECK: cir.call @_ZNSt3__17find_ifIPDuZ21find_if_literal_char8{{.*}} {cir.byte_eq_pred_value = #cir.int<122> : !u8i}

char *find_if_literal_dependent(char *first, char *last) {
  return std::find_if(
      first, last, [](auto element) { return element == sizeof(element); });
}
// CHECK-LABEL: cir.func{{.*}} @_Z25find_if_literal_dependentPcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ25find_if_literal_dependent
// CHECK-NOT: cir.byte_eq_pred_value

signed char *find_if_literal_signed_overflow(signed char *first,
                                             signed char *last) {
  return std::find_if(first, last,
                      [](signed char element) { return element == 200; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z31find_if_literal_signed_overflowPaS_
// CHECK: cir.call @_ZNSt3__17find_ifIPaZ31find_if_literal_signed_overflow
// CHECK-NOT: cir.byte_eq_pred_value

char *find_if_literal_wrong_param(char *first, char *last) {
  return std::find_if(first, last,
                      [](int element) { return element == 'x'; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z27find_if_literal_wrong_paramPcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ27find_if_literal_wrong_param
// CHECK-NOT: cir.byte_eq_pred_value

char *find_if_plain_negative(char *first, char *last) {
  return std::find_if(first, last,
                      [](char element) { return element == -1; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z22find_if_plain_negativePcS_
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ22find_if_plain_negative{{.*}} {cir.byte_eq_pred_value = #cir.int<-1> : !s8i}
// UCHAR-LABEL: cir.func{{.*}} @_Z22find_if_plain_negativePcS_
// UCHAR: cir.call @_ZNSt3__17find_ifIPcZ22find_if_plain_negative
// UCHAR-NOT: cir.byte_eq_pred_value

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

wchar_t *wide_eligible(wchar_t *first, wchar_t *last, const wchar_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z13wide_eligible
// CHECK: cir.call @_ZNSt3__14findIPwwEET_S2_S2_RKT0_({{.*}}) {cir.wide_char_params}
// UCHAR-LABEL: cir.func{{.*}} @_Z13wide_eligible
// UCHAR: cir.call @_ZNSt3__14findIPwwEET_S2_S2_RKT0_({{.*}}) {cir.wide_char_params}

wchar_t *wide_wrapped(std::span_iter<wchar_t *> first,
                      std::span_iter<wchar_t *> last, const wchar_t &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z12wide_wrapped
// CHECK: cir.call @_ZNSt3__14findINS_9span_iterIPwEEwEET_S4_S4_RKT0_({{.*}}) {cir.wide_char_params}

char16_t *wide_char16(char16_t *first, char16_t *last, const char16_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z11wide_char16
// CHECK: cir.call @_ZNSt3__14findIPDsDsEET_S2_S2_RKT0_
// CHECK-NOT: cir.wide_char_params

char32_t *wide_char32(char32_t *first, char32_t *last, const char32_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z11wide_char32
// CHECK: cir.call @_ZNSt3__14findIPDiDiEET_S2_S2_RKT0_
// CHECK-NOT: cir.wide_char_params

wchar_t *wide_mixed(wchar_t *first, wchar_t *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z10wide_mixed
// CHECK: cir.call @_ZNSt3__14findIPwcEET_S2_S2_RKT0_
// CHECK-NOT: cir.wide_char_params

char *narrow_mixed_wide_value(char *first, char *last,
                              const wchar_t &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z23narrow_mixed_wide_value
// CHECK: cir.call @_ZNSt3__14findIPcwEET_S2_S2_RKT0_
// CHECK-NOT: cir.narrow_char_params

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

char *search_char(char *first1, char *last1, char *first2, char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z11search_charPcS_S_S_
// CHECK: cir.call @_ZNSt3__16searchIPcS1_EET_S2_S2_T0_S3_({{.*}}) {{{.*}}cir.narrow_char_params

unsigned char *search_unsigned(unsigned char *first1, unsigned char *last1,
                               unsigned char *first2,
                               unsigned char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z15search_unsignedPhS_S_S_
// CHECK: cir.call @_ZNSt3__16searchIPhS1_EET_S2_S2_T0_S3_({{.*}}) {{{.*}}cir.narrow_char_params

char *search_wrapped(std::span_iter<char *> first1,
                     std::span_iter<char *> last1,
                     std::span_iter<char *> first2,
                     std::span_iter<char *> last2) {
  return std::search(first1, last1, first2, last2).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z14search_wrapped
// CHECK: cir.call @_ZNSt3__16searchINS_9span_iterIPcEES3_EET_S4_S4_T0_S5_({{.*}}) {{{.*}}cir.narrow_char_params

int *search_int(int *first1, int *last1, int *first2, int *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z10search_intPiS_S_S_
// CHECK: cir.call @_ZNSt3__16searchIPiS1_EET_S2_S2_T0_S3_
// CHECK-NOT: cir.narrow_char_params

char *search_mixed(char *first1, char *last1, unsigned char *first2,
                   unsigned char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z12search_mixedPcS_PhS0_
// CHECK: cir.call @_ZNSt3__16searchIPcPhEET_S3_S3_T0_S4_
// CHECK-NOT: cir.narrow_char_params

wchar_t *search_wchar(wchar_t *first1, wchar_t *last1, wchar_t *first2,
                      wchar_t *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z12search_wcharPwS_S_S_
// CHECK: cir.call @_ZNSt3__16searchIPwS1_EET_S2_S2_T0_S3_
// CHECK-NOT: cir.narrow_char_params

volatile char *search_volatile(volatile char *first1, volatile char *last1,
                               volatile char *first2,
                               volatile char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z15search_volatilePVcS0_S0_S0_
// CHECK: cir.call @_ZNSt3__16searchIPVcS2_EET_S3_S3_T0_S4_
// CHECK-NOT: cir.narrow_char_params
