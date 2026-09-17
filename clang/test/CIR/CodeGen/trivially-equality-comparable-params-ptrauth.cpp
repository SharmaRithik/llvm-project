// RUN: %clang_cc1 -triple arm64-apple-darwin -fptrauth-intrinsics -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --implicit-check-not=trivially_equality_comparable_params

// Both pointer auth forms reach CIR as a plain !u64i. AddrDisc signs the value
// with its storage address so two equal values can hold different bytes.

namespace std {
inline namespace __1 {
template <class It, class T> It find(It first, It last, const T &value);
} // namespace __1
} // namespace std

using AddrDisc = __UINT64_TYPE__ __ptrauth(1, 1, 1);
using NoAddrDisc = __UINT64_TYPE__ __ptrauth(1, 0, 1);

AddrDisc *addr_disc(AddrDisc *a, AddrDisc *b, const AddrDisc &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPU9__ptrauthILj1ELb1ELj1EEyS1_EET_S3_S3_RKT0_(
// CHECK-SAME: : (!cir.ptr<!u64i>{{.*}}, !cir.ptr<!u64i>{{.*}}, !cir.ptr<!u64i>{{.*}}) -> (!cir.ptr<!u64i>

NoAddrDisc *no_addr_disc(NoAddrDisc *a, NoAddrDisc *b, const NoAddrDisc &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPU9__ptrauthILj1ELb0ELj1EEyS1_EET_S3_S3_RKT0_({{.*}}{cir.trivially_equality_comparable_params}

// A __ptrauth iterator and a plain value are two types that share !u64i, yet
// the same value is stored as different bytes in each.
NoAddrDisc *mixed(NoAddrDisc *a, NoAddrDisc *b, const __UINT64_TYPE__ &v) {
  return std::find(a, b, v);
}
// CHECK: cir.call @_ZNSt3__14findIPU9__ptrauthILj1ELb0ELj1EEyyEET_S3_S3_RKT0_(
// CHECK-SAME: : (!cir.ptr<!u64i>{{.*}}, !cir.ptr<!u64i>{{.*}}, !cir.ptr<!u64i>{{.*}}) -> (!cir.ptr<!u64i>
