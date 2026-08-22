// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all -clangir-enable-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// CIR: IR Dump After IdiomRecognizer: cir-idiom-recognizer

// The implicit-check-not on the RAISED run makes any surviving std::find call
// an error in the post-pass dump, so the test only passes if it was raised to
// cir.std.find. The FINAL run checks the lowered output and its implicit-check-not
// proves no raised operation leaked past LoweringPrepare.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=RAISED '--implicit-check-not=cir.call @_ZSt4find' '--implicit-check-not=cir.call @_ZSt6search' '--implicit-check-not=cir.call @_ZNKSt6ranges'
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.cir --implicit-check-not=cir.std.

// On targets where plain char is unsigned the pointer lowers to !u8i, and
// recognition works the same.
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fno-signed-char -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.aarch64.cir 2>&1 | FileCheck %s --check-prefix=RAISED '--implicit-check-not=cir.call @_ZSt6search' '--implicit-check-not=cir.call @_ZNKSt6ranges'

// A no builtin list for another function survives the round trip on the
// rebuilt call.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fno-builtin-memcpy -fclangir -clangir-enable-idiom-recognizer -emit-cir %s -o %t.no-builtin-memcpy.cir
// RUN: FileCheck %s --check-prefix=NO-BUILTIN-MEMCPY --input-file=%t.no-builtin-memcpy.cir

// With builtins disabled the strlen call is left alone, while the tagged
// std::find is unaffected and still raises.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fno-builtin -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-BUILTINS --implicit-check-not=cir.std.strlen

namespace std {
template <class Iter, class T>
__attribute__((pure)) Iter find(Iter, Iter, const T &) noexcept;
template <class Iter1, class Iter2>
Iter1 search(Iter1, Iter1, Iter2, Iter2);
// Real std::identity is empty and call convention lowering drops an
// empty record argument, so this one carries a byte to keep the
// projection operand visible in the lowered call.
struct identity {
  unsigned char state;
  template <class T> T &&operator()(T &&t) const;
};
template <class T> struct remove_ref {
  typedef T type;
};
template <class T> struct remove_ref<T &> {
  typedef T type;
};
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
namespace ranges {
namespace __find {
struct __fn {
  template <class Iter, class Sent, class T, class Proj = identity>
  Iter operator()(Iter first, Sent last, const T &value,
                  Proj proj = {}) const noexcept;
  template <class Range, class T, class Proj = identity>
  typename remove_ref<Range>::type::iterator
  operator()(Range &&r, const T &value, Proj proj = {}) const noexcept;
};
}
inline namespace __cpo {
inline constexpr __find::__fn find{};
}
}
}
extern "C" unsigned long strlen(const char *);

char *test_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// Raised to cir.std.find, then lowered back to the exact same call with its
// operands in source order and its attributes.
// RAISED: cir.std.find(
// RAISED-SAME: @_ZSt4findIPccET_S1_S1_RKT0_
// NO-BUILTINS: cir.std.find(
// NO-BUILTINS-SAME: @_ZSt4findIPccET_S1_S1_RKT0_
// FINAL: %[[FIRST_ADDR:.*]] = cir.alloca "first"
// FINAL: %[[LAST_ADDR:.*]] = cir.alloca "last"
// FINAL: %[[VALUE_ADDR:.*]] = cir.alloca "value"
// FINAL: %[[FIRST:.*]] = cir.load{{.*}} %[[FIRST_ADDR]] :
// FINAL: %[[LAST:.*]] = cir.load{{.*}} %[[LAST_ADDR]] :
// FINAL: %[[VALUE:.*]] = cir.load{{.*}} %[[VALUE_ADDR]] :
// FINAL: cir.call @_ZSt4findIPccET_S1_S1_RKT0_(%[[FIRST]], %[[LAST]], %[[VALUE]])
// FINAL-SAME: nothrow side_effect(pure)
// FINAL-SAME: {llvm.noundef}
// FINAL-SAME: -> (!cir.ptr<!s8i> {llvm.noundef})
// FINAL-NOT: cir.call @_ZSt4find


char *test_search(char *first1, char *last1, char *first2, char *last2) {
  return std::search(first1, last1, first2, last2);
}
// RAISED-LABEL: cir.func{{.*}} @_Z11test_searchPcS_S_S_
// RAISED: %[[SFIRST1:.*]] = cir.load
// RAISED: %[[SLAST1:.*]] = cir.load
// RAISED: %[[SFIRST2:.*]] = cir.load
// RAISED: %[[SLAST2:.*]] = cir.load
// RAISED: cir.std.search(%[[SFIRST1]] : {{.*}}, %[[SLAST1]] : {{.*}}, %[[SFIRST2]] : {{.*}}, %[[SLAST2]] : {{.*}}, @_ZSt6searchIPcS0_ET_S1_S1_T0_S2_) -> {{.*}}cir.narrow_char_params
// FINAL-LABEL: cir.func{{.*}} @_Z11test_searchPcS_S_S_
// FINAL: %[[QFIRST1:.*]] = cir.load
// FINAL: %[[QLAST1:.*]] = cir.load
// FINAL: %[[QFIRST2:.*]] = cir.load
// FINAL: %[[QLAST2:.*]] = cir.load
// FINAL: cir.call @_ZSt6searchIPcS0_ET_S1_S1_T0_S2_(%[[QFIRST1]], %[[QLAST1]], %[[QFIRST2]], %[[QLAST2]])
unsigned char *test_ranges_find(unsigned char *first, unsigned char *last,
                                const unsigned char &value) {
  return std::ranges::find(first, last, value);
}
// The call through the customization point object raises with all five
// operands and lowers back to the exact same call with them in order.
// RAISED: %[[CPO:.*]] = cir.get_global @_ZNSt6ranges5__cpo4findE
// RAISED: %[[FIRST:.*]] = cir.load align(8)
// RAISED: %[[LAST:.*]] = cir.load align(8)
// RAISED: %[[VALUE:.*]] = cir.load %
// RAISED: %[[PROJ:.*]] = cir.load align(1)
// RAISED: cir.std.ranges.find(%[[CPO]] : !cir.ptr<!rec_std3A3Aranges3A3A__find3A3A__fn>, %[[FIRST]] : !cir.ptr<!u8i>, %[[LAST]] : !cir.ptr<!u8i>, %[[VALUE]] : !cir.ptr<!u8i>, %[[PROJ]] : !rec_std3A3Aidentity, @_ZNKSt6ranges6__find4__fnclIPhS3_hSt8identityEET_S5_T0_RKT1_T2_)
// RAISED-SAME: cir.narrow_char_params
// FINAL: %[[RCPO:.*]] = cir.get_global @_ZNSt6ranges5__cpo4findE
// FINAL: %[[RFIRST:.*]] = cir.load align(8)
// FINAL: %[[RLAST:.*]] = cir.load align(8)
// FINAL: %[[RVALUE:.*]] = cir.load %
// FINAL: %[[RPROJ:.*]] = cir.load %
// FINAL: cir.call @_ZNKSt6ranges6__find4__fnclIPhS3_hSt8identityEET_S5_T0_RKT1_T2_(%[[RCPO]], %[[RFIRST]], %[[RLAST]], %[[RVALUE]], %[[RPROJ]]) nothrow
// FINAL-SAME: cir.narrow_char_params

unsigned char *test_ranges_find_range(std::vector<unsigned char> &v,
                                       const unsigned char &value) {
  return std::ranges::find(v, value);
}
// The whole range call raises with the range's address and lowers back to
// the exact same call with its operands in order.
// RAISED: %[[RCPO2:.*]] = cir.get_global @_ZNSt6ranges5__cpo4findE
// RAISED: cir.std.ranges.find_range(%[[RCPO2]] : {{.*}}, %{{.*}} : !cir.ptr<!rec_std3A3Avector3Cunsigned_char3E>, %{{.*}} : !cir.ptr<!u8i>, %{{.*}} : !rec_std3A3Aidentity, @_ZNKSt6ranges6__find4__fnclIRSt6vectorIhEhSt8identityEENSt10remove_refIT_E4type8iteratorEOS8_RKT0_T1_)
// RAISED-SAME: cir.narrow_char_params
// FINAL: %[[QCPO:.*]] = cir.get_global @_ZNSt6ranges5__cpo4findE
// FINAL: cir.call @_ZNKSt6ranges6__find4__fnclIRSt6vectorIhEhSt8identityEENSt10remove_refIT_E4type8iteratorEOS8_RKT0_T1_(%[[QCPO]], %{{.*}}, %{{.*}}, %{{.*}}) nothrow
// FINAL-SAME: cir.narrow_char_params

unsigned long test_strlen(const char *s) { return strlen(s); }
// RAISED: cir.std.strlen(
// RAISED-SAME: @strlen
// RAISED-SAME: -> !u64i
// FINAL: %[[S_ADDR:.*]] = cir.alloca "s"
// FINAL: %[[S:.*]] = cir.load{{.*}} %[[S_ADDR]] :
// FINAL: cir.call @strlen(%[[S]]) nothrow
// FINAL-SAME: (!cir.ptr<!s8i> {llvm.noundef})
// FINAL-SAME: -> !u64i
// NO-BUILTIN-MEMCPY: cir.call @strlen
// NO-BUILTIN-MEMCPY-SAME: nobuiltins = ["memcpy"]
// NO-BUILTINS: cir.call @strlen

// A function merely named like the std one is not raised, and it survives the
// whole pipeline as the same plain call.
char *find(char *first, char *last, const char &value);
char *test_non_std_find(char *first, char *last, const char &value) {
  return find(first, last, value);
}
// RAISED: cir.call @_Z4findPcS_RKc
// RAISED-NOT: cir.std.find
// FINAL: cir.call @_Z4findPcS_RKc

// A member function named find in std is not std::find. The types are chosen
// so the call reaches the member exclusion itself.
namespace std {
struct string {
  string *find(string *first, string *last);
};
}

std::string *test_member_find(std::string &s, std::string *f, std::string *l) {
  return s.find(f, l);
}
// RAISED: cir.call @_ZNSt6string4findEPS_S0_
// RAISED-NOT: cir.std.find
// FINAL: cir.call @_ZNSt6string4findEPS_S0_
