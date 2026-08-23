// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-passes -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DGNU_SHAPE -fclangir -clangir-disable-passes -emit-cir %s -o %t.gnu.cir
// RUN: FileCheck --input-file=%t.gnu.cir --check-prefix=GNU %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DPARTIAL_SHAPE -fclangir -clangir-disable-passes -emit-cir %s -o %t.partial.cir
// RUN: FileCheck --input-file=%t.partial.cir --check-prefix=PARTIAL %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DLIBCXX_BIT_SHAPE -fclangir -clangir-disable-passes -emit-cir %s -o %t.libcpp-bit.cir
// RUN: FileCheck --input-file=%t.libcpp-bit.cir --check-prefix=LIBCPP_BIT %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DGNU_BIT_SHAPE -fclangir -clangir-disable-passes -emit-cir %s -o %t.gnu-bit.cir
// RUN: FileCheck --input-file=%t.gnu-bit.cir --check-prefix=GNU_BIT %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DBIT_WRONG_NAMES -fclangir -clangir-disable-passes -emit-cir %s -o %t.bit-wrong-names.cir
// RUN: FileCheck --input-file=%t.bit-wrong-names.cir --check-prefix=BIT_NEGATIVE %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DBIT_EXTRA_MEMBER -fclangir -clangir-disable-passes -emit-cir %s -o %t.bit-extra-member.cir
// RUN: FileCheck --input-file=%t.bit-extra-member.cir --check-prefix=BIT_NEGATIVE %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -DBIT_NEGATIVE_SHAPES -fclangir -clangir-disable-passes -emit-cir %s -o %t.bit-negative.cir
// RUN: FileCheck --input-file=%t.bit-negative.cir --check-prefix=BIT_NEGATIVE %s

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
// LIBCPP_BIT-COUNT-2: std_type_info = #cir.std_type_info<"std::bit_iterator", !cir.bool, roles {bit_offset = array<i32: 1>, word_pointer = array<i32: 0>}>
// GNU_BIT-COUNT-2: std_type_info = #cir.std_type_info<"std::bit_iterator", !cir.bool, roles {bit_offset = array<i32: 0, 1>, word_pointer = array<i32: 0, 0>}>
// BIT_NEGATIVE-NOT: std_type_info

#if defined(LIBCXX_BIT_SHAPE)

namespace std {
inline namespace __1 {

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
};

}
}

using libcxx_bit_iterator = std::__bit_iterator<std::vector<bool>, false>;
using libcxx_const_bit_iterator = std::__bit_iterator<std::vector<bool>, true>;

unsigned long bit_sizes(libcxx_bit_iterator &mutable_iterator,
                        libcxx_const_bit_iterator &const_iterator) {
  return sizeof(mutable_iterator) + sizeof(const_iterator);
}

#elif defined(GNU_BIT_SHAPE)

namespace std {

struct iterator {};
struct _Bit_iterator_base : iterator {
  unsigned long *_M_p;
  unsigned int _M_offset;
};
struct _Bit_iterator : _Bit_iterator_base {};
struct _Bit_const_iterator : _Bit_iterator_base {};

}

unsigned long bit_sizes(std::_Bit_iterator &mutable_iterator,
                        std::_Bit_const_iterator &const_iterator) {
  return sizeof(mutable_iterator) + sizeof(const_iterator);
}

#elif defined(BIT_WRONG_NAMES)

namespace std {

struct _Bit_iterator_base {
  unsigned long *_M_word;
  unsigned int _M_bit;
};
struct _Bit_iterator : _Bit_iterator_base {};
struct _Bit_const_iterator : _Bit_iterator_base {};

}

unsigned long bit_sizes(std::_Bit_iterator &mutable_iterator,
                        std::_Bit_const_iterator &const_iterator) {
  return sizeof(mutable_iterator) + sizeof(const_iterator);
}

#elif defined(BIT_EXTRA_MEMBER)

namespace std {

struct _Bit_iterator_base {
  unsigned long *_M_p;
  unsigned int _M_offset;
};
struct _Bit_iterator : _Bit_iterator_base {
  unsigned int extra;
};

}

unsigned long bit_size(std::_Bit_iterator &iterator) {
  return sizeof(iterator);
}

#elif defined(BIT_NEGATIVE_SHAPES)

namespace std {
inline namespace __1 {

struct good_alloc {
  using word_type = unsigned long;
  using pointer = word_type *;
  using const_pointer = const word_type *;
};
struct signed_alloc {
  using word_type = long;
  using pointer = word_type *;
  using const_pointer = const word_type *;
};
struct volatile_alloc {
  using word_type = volatile unsigned long;
  using pointer = word_type *;
  using const_pointer = const word_type *;
};
struct fancy_pointer {
  unsigned long *raw;
};
struct fancy_alloc {
  using word_type = unsigned long;
  using pointer = fancy_pointer;
  using const_pointer = fancy_pointer;
};
struct user_alloc {
  using word_type = unsigned long;
  using pointer = word_type *;
  using const_pointer = const word_type *;
};

template <class T, class Alloc> class vector {
public:
  using __storage_type = typename Alloc::word_type;
  using __storage_pointer = typename Alloc::pointer;
  using __const_storage_pointer = typename Alloc::const_pointer;
};

template <class T> class vector<T, user_alloc> {
public:
  using __storage_type = typename user_alloc::word_type;
  using __storage_pointer = typename user_alloc::pointer;
  using __const_storage_pointer = typename user_alloc::const_pointer;
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
};

template <>
class __bit_iterator<vector<bool, good_alloc>, true> {
  unsigned long *__seg_;
  unsigned __ctz_;
};

}
}

using signed_iterator =
    std::__bit_iterator<std::vector<bool, std::signed_alloc>, false>;
using volatile_iterator =
    std::__bit_iterator<std::vector<bool, std::volatile_alloc>, false>;
using fancy_iterator =
    std::__bit_iterator<std::vector<bool, std::fancy_alloc>, false>;
using partial_iterator =
    std::__bit_iterator<std::vector<bool, std::user_alloc>, false>;
using specialized_iterator =
    std::__bit_iterator<std::vector<bool, std::good_alloc>, true>;
struct derived_iterator : volatile_iterator {};

namespace user {
struct __bit_iterator {
  unsigned long *__seg_;
  unsigned __ctz_;
};
}

namespace std {
inline namespace __1 {
// The container argument must be a vector instantiation.
template <class T> class flexholder {
public:
  using __storage_pointer = unsigned long *;
  using __const_storage_pointer = const unsigned long *;
};
}
}

using other_container_iterator =
    std::__bit_iterator<std::flexholder<bool>, false>;

unsigned long bit_sizes(signed_iterator &signed_value,
                        volatile_iterator &volatile_value,
                        fancy_iterator &fancy_value,
                        partial_iterator &partial_value,
                        specialized_iterator &specialized_value,
                        derived_iterator &derived_value,
                        user::__bit_iterator &outside_value,
                        other_container_iterator &other_container_value) {
  return sizeof(signed_value) + sizeof(volatile_value) + sizeof(fancy_value) +
         sizeof(partial_value) + sizeof(specialized_value) +
         sizeof(derived_value) + sizeof(outside_value) +
         sizeof(other_container_value);
}

#else

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

#endif
