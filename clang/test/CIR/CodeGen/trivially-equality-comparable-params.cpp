// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --implicit-check-not=trivially_equality_comparable_params

using Alias = const int;

namespace std {
inline namespace __1 {
template <class It, class T> It find(It first, It last, const T &value);
template <class It, class T> It find(It a, It b, const T &v, const T &w);
short *find(short *first, short *last, short value);
int *find(int *first, int *last, Alias &value);
char *find(char *first, ...);
int *find();
} // namespace __1
} // namespace std

namespace other {
template <class It, class T> It find(It first, It last, const T &value);
} // namespace other

#define AS1 __attribute__((address_space(1)))

enum E : unsigned char {};
struct S {};

char *f_char(char *a, char *b, const char &v) { return std::find(a, b, v); }
// CHECK: cir.call @_ZNSt3__14findIPccEET_S2_S2_RKT0_({{.*}}{cir.trivially_equality_comparable_params}

const char *f_const_char(const char *a, const char *b, const char &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPKccEET_S3_S3_RKT0_({{.*}}{cir.trivially_equality_comparable_params}

unsigned char *f_uchar(unsigned char *a, unsigned char *b,
                       const unsigned char &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPhhEET_S2_S2_RKT0_({{.*}}{cir.trivially_equality_comparable_params}
// CHECK-SAME: : (!cir.ptr<!u8i>{{.*}}, !cir.ptr<!u8i>{{.*}}, !cir.ptr<!u8i>{{.*}}) -> (!cir.ptr<!u8i>

char8_t *f_char8(char8_t *a, char8_t *b, const char8_t &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPDuDuEET_S2_S2_RKT0_({{.*}}{cir.trivially_equality_comparable_params}

wchar_t *f_wchar(wchar_t *a, wchar_t *b, const wchar_t &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPwwEET_S2_S2_RKT0_({{.*}}{cir.trivially_equality_comparable_params}

AS1 char *f_as1(AS1 char *a, AS1 char *b, const AS1 char &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPU3AS1cS1_EET_S3_S3_RKT0_({{.*}}{cir.trivially_equality_comparable_params}

// CIRGen does not check arity. The recognizer does.
char *f_four(char *a, char *b, const char &v, const char &w) {
  return std::find(a, b, v, w);
}
// CHECK: cir.call @_ZNSt3__14findIPccEET_S2_S2_RKT0_S5_({{.*}}{cir.trivially_equality_comparable_params}

// A const hidden inside a typedef still qualifies.
int *f_alias(int *a, int *b, Alias &v) { return std::find(a, b, v); }
// CHECK: cir.call @_ZNSt3__14findEPiS0_RKi({{.*}}{cir.trivially_equality_comparable_params}

// Same CIR type as unsigned char, different AST type.
E *f_enum(E *a, E *b, const E &v) { return std::find(a, b, v); }
// CHECK: cir.call @_ZNSt3__14findIP1ES1_EET_S3_S3_RKT0_(
// CHECK-SAME: : (!cir.ptr<!u8i>{{.*}}, !cir.ptr<!u8i>{{.*}}, !cir.ptr<!u8i>{{.*}}) -> (!cir.ptr<!u8i>

float *f_float(float *a, float *b, const float &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPffEET_S2_S2_RKT0_(

bool *f_bool(bool *a, bool *b, const bool &v) { return std::find(a, b, v); }
// CHECK: cir.call @_ZNSt3__14findIPbbEET_S2_S2_RKT0_(
// CHECK-SAME: : (!cir.ptr<!cir.bool>{{.*}}, !cir.ptr<!cir.bool>{{.*}}, !cir.ptr<!cir.bool>{{.*}}) -> (!cir.ptr<!cir.bool>

volatile char *f_volatile(volatile char *a, volatile char *b,
                          const volatile char &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPVcS1_EET_S3_S3_RKT0_(

_Atomic(unsigned char) *f_atomic(_Atomic(unsigned char) *a,
                                 _Atomic(unsigned char) *b,
                                 const _Atomic(unsigned char) &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPU7_AtomichS1_EET_S3_S3_RKT0_(
// CHECK-SAME: : (!cir.ptr<!u8i>{{.*}}, !cir.ptr<!u8i>{{.*}}, !cir.ptr<!u8i>{{.*}}) -> (!cir.ptr<!u8i>

// _BitInt(8) is an integer but not a BuiltinType, so it is not marked.
_BitInt(8) *f_bitint(_BitInt(8) *a, _BitInt(8) *b, const _BitInt(8) &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPDB8_S1_EET_S3_S3_RKT0_(

int S::*f_memptr(int S::*a, int S::*b, const int &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIM1SiiEET_S3_S3_RKT0_(

// char and signed char share !s8i here, so a byte compare would be right,
// but the marker requires one AST type.
char *f_mixed(char *a, char *b, const signed char &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPcaEET_S2_S2_RKT0_(
// CHECK-SAME: : (!cir.ptr<!s8i>{{.*}}, !cir.ptr<!s8i>{{.*}}, !cir.ptr<!s8i>{{.*}}) -> (!cir.ptr<!s8i>

AS1 char *f_as_mixed(AS1 char *a, AS1 char *b, const char &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPU3AS1ccEET_S3_S3_RKT0_(

short *f_byvalue(short *a, short *b, short v) { return std::find(a, b, v); }
// CHECK: cir.call @_ZNSt3__14findEPsS0_s(

char *f_unrelated(char *a, char *b, const char &v) {
  return other::find(a, b, v);
}
// CHECK: cir.call @_ZN5other4findIPccEET_S2_S2_RKT0_(

char *f_variadic(char *a) { return std::find(a, 0); }
// CHECK: cir.call @_ZNSt3__14findEPcz(

// A nullary std::find gets the identity but not the marker.
int *f_empty() { return std::find(); }
// CHECK: cir.call @_ZNSt3__14findEv
// CHECK: cir.func private @_ZNSt3__14findEv{{.*}}func_info<#cir.func_identity<"std::find">>
