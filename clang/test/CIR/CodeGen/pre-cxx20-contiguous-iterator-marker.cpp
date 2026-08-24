// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.good.cir
// RUN: FileCheck --input-file=%t.good.cir --check-prefix=GOOD %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -DTWO_FIELD -fclangir -emit-cir %s -o %t.two.cir
// RUN: FileCheck --input-file=%t.two.cir --check-prefix=TWO %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -DWRONG_TAG -fclangir -emit-cir %s -o %t.tag.cir
// RUN: FileCheck --input-file=%t.tag.cir --check-prefix=TAG %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -DMEMBER_STRUCT -fclangir -emit-cir %s -o %t.member.cir
// RUN: FileCheck --input-file=%t.member.cir --check-prefix=MEMBER %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -DPARTIAL -fclangir -emit-cir %s -o %t.partial.cir
// RUN: FileCheck --input-file=%t.partial.cir --check-prefix=PARTIAL %s

struct owner {};

namespace std {
inline namespace __1 {
template <class Iter, class T>
Iter find(Iter first, Iter last, const T &value);
}
}

#if defined(WRONG_TAG) || defined(MEMBER_STRUCT) || defined(PARTIAL)
// The wrapper definitions for these runs live at the end of the file.
#elif defined(TWO_FIELD)

namespace std {
inline namespace __1 {
template <class P> struct __wrap_iter {
  P ptr;
  P end;
};
}
}

namespace __gnu_cxx {
inline namespace __8 {
template <class P, class C> struct __normal_iterator {
  P ptr;
  P end;
};
}
}

char *two_field_libcxx(std::__wrap_iter<char *> first,
                       std::__wrap_iter<char *> last,
                       const char &value) {
  return std::find(first, last, value).ptr;
}
// TWO-LABEL: cir.func{{.*}} @_Z16two_field_libcxx
// TWO: cir.call @_ZNSt3__14findINS_11__wrap_iterIPcEEcEET_S4_S4_RKT0_
// TWO-NOT: cir.narrow_char_params
// TWO-NOT: cir.wide_char_params
// TWO: cir.return

int *two_field_gnu(__gnu_cxx::__normal_iterator<int *, owner> first,
                   __gnu_cxx::__normal_iterator<int *, owner> last,
                   const int &value) {
  return std::find(first, last, value).ptr;
}
// TWO-LABEL: cir.func{{.*}} @_Z13two_field_gnu
// TWO: cir.call @_ZNSt3__14findIN9__gnu_cxx3__817__normal_iteratorIPi5ownerEEiEET_S7_S7_RKT0_
// TWO-NOT: cir.narrow_char_params
// TWO-NOT: cir.wide_char_params
// TWO: cir.return

#else

namespace std {
inline namespace __1 {
template <class P> struct __wrap_iter {
  P ptr;
};

template <> struct __wrap_iter<signed char *> {
  signed char *ptr;
};

struct __derived_wrap_iter : __wrap_iter<unsigned char *> {};
}
}

namespace __gnu_cxx {
inline namespace __8 {
template <class P, class C> struct __normal_iterator {
  P ptr;
};

template <> struct __normal_iterator<unsigned char *, owner> {
  unsigned char *ptr;
};
}

namespace nested {
template <class P, class C> struct __normal_iterator {
  P ptr;
};
}
}

namespace user {
template <class P> struct __wrap_iter {
  P ptr;
};
}

char *libcxx_char(std::__wrap_iter<char *> first,
                  std::__wrap_iter<char *> last, const char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z11libcxx_char
// GOOD: cir.call @_ZNSt3__14findINS_11__wrap_iterIPcEEcEET_S4_S4_RKT0_({{.*}}) {cir.narrow_char_params}

int *libcxx_int(std::__wrap_iter<int *> first,
                std::__wrap_iter<int *> last, const int &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z10libcxx_int
// GOOD: cir.call @_ZNSt3__14findINS_11__wrap_iterIPiEEiEET_S4_S4_RKT0_({{.*}}) {cir.wide_char_params}

char *gnu_char(__gnu_cxx::__normal_iterator<char *, owner> first,
               __gnu_cxx::__normal_iterator<char *, owner> last,
               const char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z8gnu_char
// GOOD: cir.call @_ZNSt3__14findIN9__gnu_cxx3__817__normal_iteratorIPc5ownerEEcEET_S7_S7_RKT0_({{.*}}) {cir.narrow_char_params}

// The reserved wrapper trust also licenses wmemchr for a target width int.
int *gnu_int(__gnu_cxx::__normal_iterator<int *, owner> first,
             __gnu_cxx::__normal_iterator<int *, owner> last,
             const int &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z7gnu_int
// GOOD: cir.call @_ZNSt3__14findIN9__gnu_cxx3__817__normal_iteratorIPi5ownerEEiEET_S7_S7_RKT0_({{.*}}) {cir.wide_char_params}

char *outside_std(user::__wrap_iter<char *> first,
                  user::__wrap_iter<char *> last, const char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z11outside_std
// GOOD: cir.call @_ZNSt3__14findIN4user11__wrap_iterIPcEEcEET_S5_S5_RKT0_
// GOOD-NOT: cir.narrow_char_params
// GOOD-NOT: cir.wide_char_params
// GOOD: cir.return

unsigned char *derived_wrapper(std::__derived_wrap_iter first,
                               std::__derived_wrap_iter last,
                               const unsigned char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z15derived_wrapper
// GOOD: cir.call @_ZNSt3__14findINS_19__derived_wrap_iterEhEET_S2_S2_RKT0_
// GOOD-NOT: cir.narrow_char_params
// GOOD-NOT: cir.wide_char_params
// GOOD: cir.return

volatile char *volatile_pointee(
    std::__wrap_iter<volatile char *> first,
    std::__wrap_iter<volatile char *> last, const volatile char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z16volatile_pointee
// GOOD: cir.call @_ZNSt3__14findINS_11__wrap_iterIPVcEES2_EET_S5_S5_RKT0_
// GOOD-NOT: cir.narrow_char_params
// GOOD-NOT: cir.wide_char_params
// GOOD: cir.return

char *ordinary_nested_gnu(
    __gnu_cxx::nested::__normal_iterator<char *, owner> first,
    __gnu_cxx::nested::__normal_iterator<char *, owner> last,
    const char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z19ordinary_nested_gnu
// GOOD: cir.call @_ZNSt3__14findIN9__gnu_cxx6nested17__normal_iteratorIPc5ownerEEcEET_S7_S7_RKT0_
// GOOD-NOT: cir.narrow_char_params
// GOOD-NOT: cir.wide_char_params
// GOOD: cir.return

signed char *specialized_libcxx(std::__wrap_iter<signed char *> first,
                                std::__wrap_iter<signed char *> last,
                                const signed char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z18specialized_libcxx
// GOOD: cir.call @_ZNSt3__14findINS_11__wrap_iterIPaEEaEET_S4_S4_RKT0_
// GOOD-NOT: cir.narrow_char_params
// GOOD-NOT: cir.wide_char_params
// GOOD: cir.return

unsigned char *specialized_gnu(
    __gnu_cxx::__normal_iterator<unsigned char *, owner> first,
    __gnu_cxx::__normal_iterator<unsigned char *, owner> last,
    const unsigned char &value) {
  return std::find(first, last, value).ptr;
}
// GOOD-LABEL: cir.func{{.*}} @_Z15specialized_gnu
// GOOD: cir.call @_ZNSt3__14findIN9__gnu_cxx3__817__normal_iteratorIPh5ownerEEhEET_S7_S7_RKT0_
// GOOD-NOT: cir.narrow_char_params
// GOOD-NOT: cir.wide_char_params
// GOOD: cir.return

#endif

#if defined(WRONG_TAG)

namespace std {
inline namespace __1 {
struct forward_iterator_tag {};
template <class P> struct __wrap_iter {
  P ptr;
  typedef forward_iterator_tag iterator_concept;
};
}
}

char *wrong_tag(std::__wrap_iter<char *> first, std::__wrap_iter<char *> last,
                const char &value) {
  return std::find(first, last, value).ptr;
}
// A typedef that names the wrong tag vetoes the name license too.
// TAG: cir.call @_ZNSt3__14findINS_11__wrap_iterIPcEEcEET_S4_S4_RKT0_
// TAG-NOT: cir.narrow_char_params

#elif defined(MEMBER_STRUCT)

namespace std {
inline namespace __1 {
template <class P> struct __wrap_iter {
  P ptr;
  struct iterator_concept {};
};
}
}

char *member_struct(std::__wrap_iter<char *> first,
                    std::__wrap_iter<char *> last, const char &value) {
  return std::find(first, last, value).ptr;
}
// Any member spelling the name without the tag vetoes the license.
// MEMBER: cir.call @_ZNSt3__14findINS_11__wrap_iterIPcEEcEET_S4_S4_RKT0_
// MEMBER-NOT: cir.narrow_char_params

#elif defined(PARTIAL)

namespace std {
inline namespace __1 {
template <class P> struct __wrap_iter;
template <class P> struct __wrap_iter<P *> {
  P *ptr;
};
}
}

char *through_partial(std::__wrap_iter<char *> first,
                      std::__wrap_iter<char *> last, const char &value) {
  return std::find(first, last, value).ptr;
}
// An instantiation resolved through a partial specialization is not the
// primary template shape the license names.
// PARTIAL: cir.call @_ZNSt3__14findINS_11__wrap_iterIPcEEcEET_S4_S4_RKT0_
// PARTIAL-NOT: cir.narrow_char_params

#endif
