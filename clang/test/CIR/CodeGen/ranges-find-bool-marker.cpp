// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// The ranges::find call operator over two standard bit iterators with a
// referenced bool value and a std::identity projection carries the same
// cir.bool_params marker as the plain std::find form.

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

struct identity {
  template <class T> T &&operator()(T &&t) const;
};

struct not_identity {
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

using bit_iterator = std::__bit_iterator<std::vector<bool>, false>;

bit_iterator marked(bit_iterator first, bit_iterator last, const bool &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z6marked
// CHECK: cir.call @{{.*}}__find_fn{{.*}}({{.*}}) {{{.*}}cir.bool_params

int *unmarked_element(int *first, int *last, const int &value) {
  return std::ranges::find(first, last, value);
}
// CHECK-LABEL: cir.func{{.*}} @_Z16unmarked_element
// CHECK: cir.call @{{.*}}__find_fn
// CHECK-NOT: cir.bool_params

bit_iterator unmarked_projection(bit_iterator first, bit_iterator last,
                                 const bool &value) {
  return std::ranges::find(first, last, value, std::not_identity{});
}
// CHECK-LABEL: cir.func{{.*}} @_Z19unmarked_projection
// CHECK: cir.call @{{.*}}__find_fn
// CHECK-NOT: cir.bool_params
