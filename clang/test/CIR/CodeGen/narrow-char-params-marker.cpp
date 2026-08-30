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
bool equal(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Pred pred) {
  for (; first1 != last1 && first2 != last2; ++first1, ++first2)
    if (!pred(*first1, *first2))
      return false;
  return first1 == last1 && first2 == last2;
}
template <class Iter1, class Iter2, class Pred>
bool equal(Iter1 first1, Iter1 last1, Iter2 first2, Pred pred) {
  for (; first1 != last1; ++first1, ++first2)
    if (!pred(*first1, *first2))
      return false;
  return true;
}
template <class T1, class T2> struct pair {
  T1 first;
  T2 second;
};
struct __equal_to {
  template <class T1, class T2>
  bool operator()(const T1 &lhs, const T2 &rhs) const {
    return lhs == rhs;
  }
};
template <class T> struct equal_to {
  bool operator()(const T &lhs, const T &rhs) const { return lhs == rhs; }
};
struct __stateful_equal_to {
  int state;
  bool operator()(int lhs, int rhs) const { return lhs == rhs; }
};
struct __stateful_base {
  int state;
};
struct __base_equal_to : __stateful_base {
  bool operator()(int lhs, int rhs) const { return lhs == rhs; }
};
struct __unavailable_equal_to {
  bool operator()(int lhs, int rhs) const;
};
struct __not_equal {
  bool operator()(int lhs, int rhs) const { return lhs != rhs; }
};
struct __converted_equal_to {
  bool operator()(int lhs, int rhs) const {
    return static_cast<unsigned>(lhs) == static_cast<unsigned>(rhs);
  }
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
template <class Iter1, class Iter2>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Iter2 last2) {
  for (; first1 != last1 && first2 != last2; ++first1, ++first2)
    if (!(*first1 == *first2))
      break;
  return {first1, first2};
}
template <class Iter1, class Iter2, class Pred>
pair<Iter1, Iter2> mismatch(Iter1 first1, Iter1 last1, Iter2 first2,
                            Iter2 last2, Pred pred) {
  for (; first1 != last1 && first2 != last2; ++first1, ++first2)
    if (!pred(*first1, *first2))
      break;
  return {first1, first2};
}
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

// Real std::identity is empty and call convention lowering drops an empty
// record argument, so this one carries a byte to keep the projection
// operand visible.
struct identity {
  unsigned char state;
  template <class T> T &&operator()(T &&value) const;
};

namespace ranges {
namespace __find_if {
struct __fn {
  template <class Iter, class Sent, class Pred, class Proj = identity>
  Iter operator()(Iter first, Sent last, Pred pred,
                  Proj proj = {}) const {
    return std::find_if(first, last, pred);
  }
};
}
namespace __find_if_not {
struct __fn {
  template <class Iter, class Sent, class Pred, class Proj = identity>
  Iter operator()(Iter first, Sent last, Pred pred,
                  Proj proj = {}) const {
    return std::find_if_not(first, last, pred);
  }
};
}
inline namespace __cpo {
inline constexpr __find_if::__fn find_if{};
inline constexpr __find_if_not::__fn find_if_not{};
}
}

struct bit_alloc {
  using word_type = unsigned long;
  using pointer = word_type *;
  using const_pointer = const word_type *;
};

template <class T, class Alloc = bit_alloc> class vector;
template <class Alloc> class vector<bool, Alloc> {
public:
  using __storage_type = typename Alloc::word_type;
  using __storage_pointer = typename Alloc::pointer;
  using __const_storage_pointer = typename Alloc::const_pointer;
};

template <bool, class True, class False> struct __choose;
template <class True, class False> struct __choose<true, True, False> {
  using type = True;
};
template <class True, class False> struct __choose<false, True, False> {
  using type = False;
};

template <class Cp, bool IsConst> class __bit_iterator {
  using storage_pointer =
      typename __choose<IsConst, typename Cp::__const_storage_pointer,
                        typename Cp::__storage_pointer>::type;
  storage_pointer __seg_;
  unsigned __ctz_;

public:
  bool operator!=(const __bit_iterator &) const;
  __bit_iterator &operator++();
  bool operator*() const;
};
}
}

using bit_iterator = std::__bit_iterator<std::vector<bool>, false>;

char *find_if_ref_capture(char *first, char *last, const char &value) {
  return std::find_if(first, last, [&](char element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z19find_if_ref_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPcZ19find_if_ref_capture{{.*}} {{{.*}}cir.byte_eq_pred}
// CHECK-NOT: cir.bool_eq_pred

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

char *ranges_byte_find_if_capture(char *first, char *last,
                                  const char &value) {
  return std::ranges::find_if(
      first, last, [&](char element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z27ranges_byte_find_if_capture
// CHECK: cir.call @_ZNKSt3__16ranges9__find_if4__fnclIPcS4_Z27ranges_byte_find_if_capture{{.*}} {cir.byte_eq_pred}

unsigned char *ranges_byte_find_if_not_literal(unsigned char *first,
                                                unsigned char *last) {
  return std::ranges::find_if_not(
      first, last, [](unsigned char element) { return element != 0x80; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z31ranges_byte_find_if_not_literal
// CHECK: cir.call @_ZNKSt3__16ranges13__find_if_not4__fnclIPhS4_Z31ranges_byte_find_if_not_literal{{.*}} {cir.byte_eq_pred_value = #cir.int<128> : !u8i}

char *ranges_byte_find_if_wrong_polarity(char *first, char *last,
                                         const char &value) {
  return std::ranges::find_if(
      first, last, [&](char element) { return element != value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z34ranges_byte_find_if_wrong_polarity
// CHECK: cir.call @_ZNKSt3__16ranges9__find_if4__fnclIPcS4_Z34ranges_byte_find_if_wrong_polarity
// CHECK-NOT: cir.byte_eq_pred

struct ByteProjection {
  char operator()(char) const;
};
char *ranges_byte_find_if_projected(char *first, char *last,
                                    const char &value) {
  return std::ranges::find_if(first, last,
                              [&](char element) { return element == value; },
                              ByteProjection{});
}
// CHECK-LABEL: cir.func{{.*}} @_Z29ranges_byte_find_if_projected
// CHECK: cir.call @_ZNKSt3__16ranges9__find_if4__fnclIPcS4_Z29ranges_byte_find_if_projected{{.*}}) : (

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

bit_iterator bool_find_if_capture(bit_iterator first, bit_iterator last,
                                  const bool &value) {
  return std::find_if(first, last,
                      [&](bool element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20bool_find_if_capture
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator{{.*}} {cir.bool_eq_pred}

bit_iterator bool_find_if_not_capture(bit_iterator first, bit_iterator last,
                                      const bool &value) {
  return std::find_if_not(first, last,
                          [&](bool element) { return element != value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z24bool_find_if_not_capture
// CHECK: cir.call @_ZNSt3__111find_if_notINS_14__bit_iterator{{.*}} {cir.bool_eq_pred}

bit_iterator bool_find_if_reversed(bit_iterator first, bit_iterator last,
                                   bool value) {
  return std::find_if(first, last, [value](const bool &element) {
    return value == element;
  });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21bool_find_if_reversed
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator{{.*}} {cir.bool_eq_pred}

bit_iterator bool_find_if_identity(bit_iterator first, bit_iterator last) {
  return std::find_if(first, last, [](bool element) { return element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21bool_find_if_identity
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator{{.*}} {cir.bool_eq_pred_value = true}

bit_iterator bool_find_if_not_identity(bit_iterator first, bit_iterator last) {
  return std::find_if_not(first, last,
                          [](bool element) { return element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z25bool_find_if_not_identity
// CHECK: cir.call @_ZNSt3__111find_if_notINS_14__bit_iterator{{.*}} {cir.bool_eq_pred_value = false}

bit_iterator bool_find_if_negation(bit_iterator first, bit_iterator last) {
  return std::find_if(first, last, [](bool element) { return !element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21bool_find_if_negation
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator{{.*}} {cir.bool_eq_pred_value = false}

bit_iterator bool_find_if_not_negation(bit_iterator first, bit_iterator last) {
  return std::find_if_not(first, last,
                          [](bool element) { return !element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z25bool_find_if_not_negation
// CHECK: cir.call @_ZNSt3__111find_if_notINS_14__bit_iterator{{.*}} {cir.bool_eq_pred_value = true}

bit_iterator bool_find_if_literal(bit_iterator first, bit_iterator last) {
  return std::find_if(first, last,
                      [](bool element) { return false != element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20bool_find_if_literal
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator{{.*}} {cir.bool_eq_pred_value = true}

bit_iterator bool_find_if_generic(bit_iterator first, bit_iterator last,
                                  const bool &value) {
  return std::find_if(first, last,
                      [&](auto element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20bool_find_if_generic
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator{{.*}} {cir.bool_eq_pred}

bit_iterator bool_find_if_not_wrong_polarity(bit_iterator first,
                                             bit_iterator last,
                                             const bool &value) {
  return std::find_if_not(first, last,
                          [&](bool element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z31bool_find_if_not_wrong_polarity
// CHECK: cir.call @_ZNSt3__111find_if_notINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

bit_iterator ranges_bool_find_if_capture(bit_iterator first,
                                         bit_iterator last,
                                         const bool &value) {
  return std::ranges::find_if(
      first, last, [&](bool element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z27ranges_bool_find_if_capture
// CHECK: cir.call @_ZNKSt3__16ranges9__find_if4__fnclINS_14__bit_iterator{{.*}} {cir.bool_eq_pred}

bit_iterator ranges_bool_find_if_not_identity(bit_iterator first,
                                              bit_iterator last) {
  return std::ranges::find_if_not(
      first, last, [](bool element) { return element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z32ranges_bool_find_if_not_identity
// CHECK: cir.call @_ZNKSt3__16ranges13__find_if_not4__fnclINS_14__bit_iterator{{.*}} {cir.bool_eq_pred_value = false}

bit_iterator ranges_bool_find_if_not_wrong_polarity(bit_iterator first,
                                                    bit_iterator last,
                                                    const bool &value) {
  return std::ranges::find_if_not(
      first, last, [&](bool element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z38ranges_bool_find_if_not_wrong_polarity
// CHECK: cir.call @_ZNKSt3__16ranges13__find_if_not4__fnclINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

bit_iterator bool_find_if_two_captures(bit_iterator first, bit_iterator last,
                                       bool value, bool other) {
  return std::find_if(first, last, [value, other](bool element) {
    return element == value;
  });
}
// CHECK-LABEL: cir.func{{.*}} @_Z25bool_find_if_two_captures
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

bit_iterator bool_find_if_non_bool_capture(bit_iterator first,
                                           bit_iterator last, int value) {
  return std::find_if(first, last,
                      [value](bool element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z29bool_find_if_non_bool_capture
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

bit_iterator bool_find_if_explicit_cast(bit_iterator first, bit_iterator last,
                                        bool value) {
  return std::find_if(first, last, [value](bool element) {
    return element == static_cast<bool>(value);
  });
}
// CHECK-LABEL: cir.func{{.*}} @_Z26bool_find_if_explicit_cast
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

bit_iterator bool_find_if_int_literal(bit_iterator first, bit_iterator last) {
  return std::find_if(first, last,
                      [](bool element) { return element == 1; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z24bool_find_if_int_literal
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

bit_iterator bool_find_if_ignored_capture(bit_iterator first,
                                          bit_iterator last, bool value) {
  return std::find_if(first, last,
                      [value](bool element) { return element; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z28bool_find_if_ignored_capture
// CHECK: cir.call @_ZNSt3__17find_ifINS_14__bit_iterator
// CHECK-NOT: cir.bool_eq_pred

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
// The wchar width int gets the wide marker, not the narrow one. The
// attribute dictionary prints alphabetically, so the narrow check sits
// between the call and the wide marker where the narrow name would land.
// CHECK-LABEL: cir.func{{.*}} @_Z10search_intPiS_S_S_
// CHECK: cir.call @_ZNSt3__16searchIPiS1_EET_S2_S2_T0_S3_(
// CHECK-NOT: cir.narrow_char_params
// CHECK-SAME: cir.wide_char_params

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
// CHECK: cir.call @_ZNSt3__16searchIPwS1_EET_S2_S2_T0_S3_(
// CHECK-NOT: cir.narrow_char_params
// CHECK-SAME: cir.wide_char_params

short *search_short(short *first1, short *last1, short *first2,
                    short *last2) {
  return std::search(first1, last1, first2, last2);
}
// short is not a trusted builtin kind for the wide whitelist on any
// target, so it gets neither marker.
// CHECK-LABEL: cir.func{{.*}} @_Z12search_shortPsS_S_S_
// CHECK: cir.call @_ZNSt3__16searchIPsS1_EET_S2_S2_T0_S3_
// CHECK-NOT: cir.narrow_char_params
// CHECK-NOT: cir.wide_char_params

volatile char *search_volatile(volatile char *first1, volatile char *last1,
                               volatile char *first2,
                               volatile char *last2) {
  return std::search(first1, last1, first2, last2);
}
// CHECK-LABEL: cir.func{{.*}} @_Z15search_volatilePVcS0_S0_S0_
// CHECK: cir.call @_ZNSt3__16searchIPVcS2_EET_S3_S3_T0_S4_
// CHECK-NOT: cir.narrow_char_params

bool equal_pred_generic(char *first1, char *last1, char *first2) {
  return std::equal(first1, last1, first2,
                    [](auto a, auto b) { return a == b; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z18equal_pred_generic
// CHECK: cir.call @_ZNSt3__15equalIPcS1_Z18equal_pred_genericS1_S1_S1_E3$_0EEbT_S3_T0_T1_({{.*}}) {cir.elem_eq_binary_pred}

bool equal_pred_reversed(unsigned char *first1, unsigned char *last1,
                         unsigned char *first2) {
  return std::equal(first1, last1, first2,
                    [](unsigned char a, unsigned char b) { return b == a; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z19equal_pred_reversed
// CHECK: cir.call @_ZNSt3__15equalIPhS1_Z19equal_pred_reversedS1_S1_S1_E3$_0EEbT_S3_T0_T1_({{.*}}) {cir.elem_eq_binary_pred}

bool equal_pred_capturing(char *first1, char *last1, char *first2, char x) {
  return std::equal(first1, last1, first2,
                    [x](char a, char b) { return a == b && a != x; });
}
// The capture carries state the pure equality proof cannot cover.
// CHECK-LABEL: cir.func{{.*}} @_Z20equal_pred_capturing
// CHECK: cir.call @_ZNSt3__15equalIPcS1_Z20equal_pred_capturingS1_S1_S1_cE3$_0EEbT_S3_T0_T1_
// CHECK-NOT: cir.elem_eq_binary_pred

bool equal_pred_inequality(char *first1, char *last1, char *first2) {
  return std::equal(first1, last1, first2,
                    [](char a, char b) { return a != b; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z21equal_pred_inequality
// CHECK: cir.call @_ZNSt3__15equalIPcS1_Z21equal_pred_inequalityS1_S1_S1_E3$_0EEbT_S3_T0_T1_
// CHECK-NOT: cir.elem_eq_binary_pred

bool equal_pred_cast(char *first1, char *last1, char *first2) {
  return std::equal(first1, last1, first2,
                    [](char a, char b) { return a == (unsigned char)b; });
}
// The explicit cast changes the comparison at values above 0x7f, so the
// proof must not look through it.
// CHECK-LABEL: cir.func{{.*}} @_Z15equal_pred_cast
// CHECK: cir.call @_ZNSt3__15equalIPcS1_Z15equal_pred_castS1_S1_S1_E3$_0EEbT_S3_T0_T1_
// CHECK-NOT: cir.elem_eq_binary_pred

bool equal_pred_int(int *first1, int *last1, int *first2) {
  return std::equal(first1, last1, first2,
                    [](int a, int b) { return a == b; });
}
// The lambda proof covers wide elements too.
// CHECK-LABEL: cir.func{{.*}} @_Z14equal_pred_int
// CHECK: cir.call @_ZNSt3__15equalIPiS1_Z14equal_pred_intS1_S1_S1_E3$_0EEbT_S3_T0_T1_({{.*}}) {cir.elem_eq_binary_pred, cir.wide_char_params}

struct UserEqual {
  bool operator()(int lhs, int rhs) const { return lhs == rhs; }
};

namespace wrong {
struct equal_to {
  bool operator()(int lhs, int rhs) const { return lhs == rhs; }
};
}

bool mismatch_byte_functor(char *first1, char *last1, char *first2) {
  return std::mismatch(first1, last1, first2, std::__equal_to{}).first ==
         last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_byte_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPcS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred}

bool mismatch_wchar_default(wchar_t *first1, wchar_t *last1,
                            wchar_t *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z22mismatch_wchar_default
// CHECK: cir.call @_ZNSt3__18mismatchIPwS1_EE{{.*}} {cir.wide_char_params}

bool mismatch_wchar_pred(wchar_t *first1, wchar_t *last1, wchar_t *first2) {
  return std::mismatch(first1, last1, first2, std::__equal_to{}).first ==
         last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z19mismatch_wchar_pred
// CHECK: cir.call @_ZNSt3__18mismatchIPwS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_wchar_bounded(wchar_t *first1, wchar_t *last1, wchar_t *first2,
                            wchar_t *last2) {
  return std::mismatch(first1, last1, first2, last2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z22mismatch_wchar_bounded
// CHECK: cir.call @_ZNSt3__18mismatchIPwS1_EE{{.*}} {cir.wide_char_params}

bool mismatch_wchar_bounded_pred(wchar_t *first1, wchar_t *last1,
                                 wchar_t *first2, wchar_t *last2) {
  return std::mismatch(first1, last1, first2, last2, std::__equal_to{})
             .first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z27mismatch_wchar_bounded_pred
// CHECK: cir.call @_ZNSt3__18mismatchIPwS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_int_default(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z20mismatch_int_default
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_EE{{.*}} {cir.wide_char_params}

bool mismatch_int_functor(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2, std::__equal_to{}).first ==
         last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z20mismatch_int_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_int_bounded(int *first1, int *last1, int *first2, int *last2) {
  return std::mismatch(first1, last1, first2, last2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z20mismatch_int_bounded
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_EE{{.*}} {cir.wide_char_params}

bool mismatch_int_bounded_functor(int *first1, int *last1, int *first2,
                                  int *last2) {
  return std::mismatch(first1, last1, first2, last2, std::__equal_to{})
             .first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z28mismatch_int_bounded_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_uint_default(unsigned *first1, unsigned *last1,
                           unsigned *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_uint_default
// CHECK: cir.call @_ZNSt3__18mismatchIPjS1_EE{{.*}} {cir.wide_char_params}

bool mismatch_uint_pred(unsigned *first1, unsigned *last1, unsigned *first2) {
  return std::mismatch(first1, last1, first2, std::__equal_to{}).first ==
         last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z18mismatch_uint_pred
// CHECK: cir.call @_ZNSt3__18mismatchIPjS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_uint_bounded(unsigned *first1, unsigned *last1,
                           unsigned *first2, unsigned *last2) {
  return std::mismatch(first1, last1, first2, last2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_uint_bounded
// CHECK: cir.call @_ZNSt3__18mismatchIPjS1_EE{{.*}} {cir.wide_char_params}

bool mismatch_uint_bounded_pred(unsigned *first1, unsigned *last1,
                                unsigned *first2, unsigned *last2) {
  return std::mismatch(first1, last1, first2, last2, std::__equal_to{})
             .first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z26mismatch_uint_bounded_pred
// CHECK: cir.call @_ZNSt3__18mismatchIPjS1_NS_10__equal_toEEE{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_std_equal_to(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2, std::equal_to<int>{}).first ==
         last1;
}
// Only the reserved __equal_to name is licensed, since its body is a
// pending instantiation here and cannot be proven.
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_std_equal_to
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_8equal_toIiEEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_stateful_functor(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2,
                       std::__stateful_equal_to{0})
             .first == last1;
}
// A field carries state the equality meaning cannot account for.
// CHECK-LABEL: cir.func{{.*}} @_Z25mismatch_stateful_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_19__stateful_equal_toEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_stateful_base(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2,
                       std::__base_equal_to{{0}})
             .first == last1;
}
// A base class carries the state instead.
// CHECK-LABEL: cir.func{{.*}} @_Z22mismatch_stateful_base
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_15__base_equal_toEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_unavailable_functor(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2,
                       std::__unavailable_equal_to{})
             .first == last1;
}
// Not the reserved name, so no license applies.
// CHECK-LABEL: cir.func{{.*}} @_Z28mismatch_unavailable_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_22__unavailable_equal_toEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_inequality_functor(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2, std::__not_equal{}).first ==
         last1;
}
// Not the reserved name either, its body never enters the decision.
// CHECK-LABEL: cir.func{{.*}} @_Z27mismatch_inequality_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_11__not_equalEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_converted_functor(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2,
                       std::__converted_equal_to{})
             .first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z26mismatch_converted_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_NS_20__converted_equal_toEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_user_functor(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2, UserEqual{}).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_user_functor
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_9UserEqualE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_wrong_namespace(int *first1, int *last1, int *first2) {
  return std::mismatch(first1, last1, first2, wrong::equal_to{}).first ==
         last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z24mismatch_wrong_namespace
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_N5wrong8equal_toEEE{{.*}} {cir.wide_char_params}
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_bool(bool *first1, bool *last1, bool *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z13mismatch_bool
// CHECK: cir.call @_ZNSt3__18mismatchIPbS1_EE
// CHECK-NOT: cir.wide_char_params

// The classifier whitelists the plain int types, so every _BitInt
// declines regardless of width.
using PaddedBitInt = _BitInt(31);
bool mismatch_padded_bitint(PaddedBitInt *first1, PaddedBitInt *last1,
                            PaddedBitInt *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z22mismatch_padded_bitint
// CHECK: cir.call @_ZNSt3__18mismatchIPDB31_S2_EE
// CHECK-NOT: cir.wide_char_params

enum MismatchEnum : int { mismatch_zero };
bool mismatch_enum(MismatchEnum *first1, MismatchEnum *last1,
                   MismatchEnum *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z13mismatch_enum
// CHECK: cir.call @_ZNSt3__18mismatchIP12MismatchEnumS2_EE
// CHECK-NOT: cir.wide_char_params

bool mismatch_volatile(volatile int *first1, volatile int *last1,
                       volatile int *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z17mismatch_volatile
// CHECK: cir.call @_ZNSt3__18mismatchIPViS2_EE
// CHECK-NOT: cir.wide_char_params

bool mismatch_short(short *first1, short *last1, short *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z14mismatch_short
// CHECK: cir.call @_ZNSt3__18mismatchIPsS1_EE
// CHECK-NOT: cir.wide_char_params

bool mismatch_long_long(long long *first1, long long *last1,
                        long long *first2) {
  return std::mismatch(first1, last1, first2).first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z18mismatch_long_long
// CHECK: cir.call @_ZNSt3__18mismatchIPxS1_EE
// CHECK-NOT: cir.wide_char_params

bool mismatch_char(char *first1, char *last1, char *first2) {
  auto r = std::mismatch(first1, last1, first2);
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z13mismatch_char
// CHECK: cir.call @_ZNSt3__18mismatchIPcS1_EENS_4pairIT_T0_EES3_S3_S4_({{.*}}) {cir.narrow_char_params}

bool mismatch_pred_generic(char *first1, char *last1, char *first2) {
  auto r = std::mismatch(first1, last1, first2,
                         [](auto a, auto b) { return a == b; });
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_pred_generic
// CHECK: cir.call @_ZNSt3__18mismatchIPcS1_Z21mismatch_pred_genericS1_S1_S1_E3$_0EENS_4pairIT_T0_EES4_S4_S5_T1_({{.*}}) {cir.elem_eq_binary_pred}

bool mismatch_pred_int(int *first1, int *last1, int *first2) {
  auto r = std::mismatch(first1, last1, first2,
                         [](int a, int b) { return a == b; });
  return r.first == last1;
}
// The generalized element proof marks the typed int lambda too.
// CHECK-LABEL: cir.func{{.*}} @_Z17mismatch_pred_int
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_Z17mismatch_pred_intS1_S1_S1_E3$_0EENS_4pairIT_T0_EES4_S4_S5_T1_({{.*}}) {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_bounded_char(char *first1, char *last1, char *first2,
                           char *last2) {
  auto r = std::mismatch(first1, last1, first2, last2);
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z21mismatch_bounded_char
// CHECK: cir.call @_ZNSt3__18mismatchIPcS1_EENS_4pairIT_T0_EES3_S3_S4_S4_({{.*}}) {cir.narrow_char_params}

bool mismatch_bounded_int(int *first1, int *last1, int *first2,
                          int *last2) {
  auto r = std::mismatch(first1, last1, first2, last2);
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z20mismatch_bounded_int
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_EENS_4pairIT_T0_EES3_S3_S4_S4_
// CHECK-NOT: cir.narrow_char_params

bool mismatch_bounded_pred_generic(char *first1, char *last1, char *first2,
                                   char *last2) {
  auto r = std::mismatch(first1, last1, first2, last2,
                         [](auto a, auto b) { return a == b; });
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z29mismatch_bounded_pred_generic
// CHECK: cir.call @_ZNSt3__18mismatchIPcS1_Z29mismatch_bounded_pred_genericS1_S1_S1_S1_E3$_0EENS_4pairIT_T0_EES4_S4_S5_S5_T1_({{.*}}) {cir.elem_eq_binary_pred}

bool mismatch_bounded_pred_int(int *first1, int *last1, int *first2,
                               int *last2) {
  auto r = std::mismatch(first1, last1, first2, last2,
                         [](int a, int b) { return a == b; });
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z25mismatch_bounded_pred_int
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_Z25mismatch_bounded_pred_intS1_S1_S1_S1_E3$_0EENS_4pairIT_T0_EES4_S4_S5_S5_T1_({{.*}}) {cir.elem_eq_binary_pred, cir.wide_char_params}

bool ranges_byte_wrong_not(char *first, char *last, char value) {
  auto it = std::ranges::find_if_not(
      first, last, [&](char element) { return element == value; });
  return it != last;
}
// The equality body under find_if_not has the wrong polarity for the fold.
// CHECK-LABEL: cir.func{{.*}} @_Z21ranges_byte_wrong_not
// CHECK-NOT: cir.byte_eq_pred

bit_iterator ranges_bool_wrong_if(bit_iterator first, bit_iterator last,
                                  bool x) {
  return std::ranges::find_if(
      first, last, [&](auto element) { return element != x; });
}
// The inequality body under find_if has the wrong polarity for the fold.
// CHECK-LABEL: cir.func{{.*}} @_Z20ranges_bool_wrong_if
// CHECK-NOT: cir.bool_eq_pred

int *int_eligible(int *first, int *last, const int &value) {
  return std::find(first, last, value);
}
// An int of the target wchar_t width searches like wchar_t does.
// CHECK-LABEL: cir.func{{.*}} @_Z12int_eligible
// CHECK: cir.call @_ZNSt3__14findIPiiEET_S2_S2_RKT0_({{.*}}) {cir.wide_char_params}
// UCHAR-LABEL: cir.func{{.*}} @_Z12int_eligible
// UCHAR: cir.call @_ZNSt3__14findIPiiEET_S2_S2_RKT0_({{.*}}) {cir.wide_char_params}

unsigned *uint_wrapped(std::span_iter<unsigned *> first,
                       std::span_iter<unsigned *> last,
                       const unsigned &value) {
  return std::find(first, last, value).ptr;
}
// CHECK-LABEL: cir.func{{.*}} @_Z12uint_wrapped
// CHECK: cir.call @_ZNSt3__14findINS_9span_iterIPjEEjEET_S4_S4_RKT0_({{.*}}) {cir.wide_char_params}

short *wide_short(short *first, short *last, const short &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z10wide_short
// CHECK: cir.call @_ZNSt3__14findIPssEET_S2_S2_RKT0_
// CHECK-NOT: cir.wide_char_params

long long *wide_longlong(long long *first, long long *last,
                         const long long &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z13wide_longlong
// CHECK: cir.call @_ZNSt3__14findIPxxEET_S2_S2_RKT0_
// CHECK-NOT: cir.wide_char_params

enum IntSized : int { kZero };
IntSized *wide_enum(IntSized *first, IntSized *last, const IntSized &value) {
  return std::find(first, last, value);
}
// An enum of the right width can carry a user defined operator==.
// CHECK-LABEL: cir.func{{.*}} @_Z9wide_enum
// CHECK-NOT: cir.wide_char_params

int *int_long_value(int *first, int *last, const long &value) {
  return std::find(first, last, value);
}
// The comparison converts the elements to long, so this is not a search
// for any single int bit pattern.
// CHECK-LABEL: cir.func{{.*}} @_Z14int_long_value
// CHECK: cir.call @_ZNSt3__14findIPilEET_S2_S2_RKT0_
// CHECK-NOT: cir.wide_char_params

volatile int *wide_volatile(volatile int *first, volatile int *last,
                            const int &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z13wide_volatile
// CHECK: cir.call @_ZNSt3__14findIPViiEET_S3_S3_RKT0_
// CHECK-NOT: cir.wide_char_params

int *wide_find_if_ref_capture(int *first, int *last, const int &value) {
  return std::find_if(first, last,
                      [&](int element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z24wide_find_if_ref_capture
// CHECK: cir.call @_ZNSt3__17find_ifIPiZ24wide_find_if_ref_capture{{.*}} {cir.wide_eq_pred}

int *wide_find_if_not_ne(int *first, int *last, const int &value) {
  return std::find_if_not(first, last,
                          [&](int element) { return element != value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z19wide_find_if_not_ne
// CHECK: cir.call @_ZNSt3__111find_if_notIPiZ19wide_find_if_not_ne{{.*}} {cir.wide_eq_pred}

int *wide_find_if_literal(int *first, int *last) {
  return std::find_if(first, last,
                      [](int element) { return element == 70000; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z20wide_find_if_literal
// CHECK: cir.call @_ZNSt3__17find_ifIPiZ20wide_find_if_literal{{.*}} {cir.wide_eq_pred_value = #cir.int<70000> : !s32i}

unsigned *wide_find_if_negative_literal(unsigned *first, unsigned *last) {
  return std::find_if(first, last,
                      [](unsigned element) { return element == -1; });
}
// The AST converts the int literal to unsigned before this side is
// evaluated, so the recorded bits are exactly the source comparison.
// CHECK-LABEL: cir.func{{.*}} @_Z29wide_find_if_negative_literal
// CHECK: cir.call @_ZNSt3__17find_ifIPjZ29wide_find_if_negative_literal{{.*}} {cir.wide_eq_pred_value = #cir.int<4294967295> : !u32i}

unsigned *wide_find_if_negative_long(unsigned *first, unsigned *last) {
  return std::find_if(first, last,
                      [](unsigned element) { return element == -1L; });
}
// The comparison happens at long, where the unsigned int element never
// equals the negative constant.
// CHECK-LABEL: cir.func{{.*}} @_Z26wide_find_if_negative_long
// CHECK: cir.call @_ZNSt3__17find_ifIPjZ26wide_find_if_negative_long
// CHECK-NOT: cir.wide_eq_pred

int *wide_find_if_wrong_polarity(int *first, int *last, const int &value) {
  return std::find_if(first, last,
                      [&](int element) { return element != value; });
}
// The inequality body under find_if has the wrong polarity for the fold.
// CHECK-LABEL: cir.func{{.*}} @_Z27wide_find_if_wrong_polarity
// CHECK: cir.call @_ZNSt3__17find_ifIPiZ27wide_find_if_wrong_polarity
// CHECK-NOT: cir.wide_eq_pred

int *ranges_wide_find_if_capture(int *first, int *last, const int &value) {
  return std::ranges::find_if(first, last,
                              [&](int element) { return element == value; });
}
// CHECK-LABEL: cir.func{{.*}} @_Z27ranges_wide_find_if_capture
// CHECK: cir.call @_ZNKSt3__16ranges9__find_if4__fnclIPiS4_Z27ranges_wide_find_if_capture{{.*}} {cir.wide_eq_pred}

bool equal_bounded_pred(char *first1, char *last1, char *first2,
                        char *last2) {
  return std::equal(first1, last1, first2, last2,
                    [](char x, char y) { return x == y; });
}
// The proof would pass, but no raised operation consumes the marker on
// the five parameter overload, so it stays unmarked.
// CHECK-LABEL: cir.func{{.*}} @_Z18equal_bounded_pred
// CHECK: cir.call @_ZNSt3__15equalIPcS1_Z18equal_bounded_predS1_S1_S1_S1_E3$_0EEbT_S3_T0_S4_T1_
// CHECK-NOT: cir.elem_eq_binary_pred

bool mismatch_int_lambda_pred(int *first1, int *last1, int *first2) {
  auto r = std::mismatch(first1, last1, first2,
                         [](int x, int y) { return x == y; });
  return r.first == last1;
}
// The typed int lambda takes the same generalized element proof the
// functor path uses.
// CHECK-LABEL: cir.func{{.*}} @_Z24mismatch_int_lambda_predPiS_S_
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_Z24mismatch_int_lambda_predS1_S1_S1_E3$_0{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_int_lambda_bounded_pred(int *first1, int *last1, int *first2,
                                      int *last2) {
  auto r = std::mismatch(first1, last1, first2, last2,
                         [](int x, int y) { return x == y; });
  return r.first == last1;
}
// CHECK-LABEL: cir.func{{.*}} @_Z32mismatch_int_lambda_bounded_predPiS_S_S_
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_Z32mismatch_int_lambda_bounded_predS1_S1_S1_S1_E3$_0{{.*}} {cir.elem_eq_binary_pred, cir.wide_char_params}

bool mismatch_int_lambda_wrong_polarity(int *first1, int *last1,
                                        int *first2) {
  auto r = std::mismatch(first1, last1, first2,
                         [](int x, int y) { return x != y; });
  return r.first == last1;
}
// The inequality body proves nothing for memcmp equality.
// CHECK-LABEL: cir.func{{.*}} @_Z34mismatch_int_lambda_wrong_polarityPiS_S_
// CHECK: cir.call @_ZNSt3__18mismatchIPiS1_Z34mismatch_int_lambda_wrong_polarity
// CHECK-NOT: cir.elem_eq_binary_pred
