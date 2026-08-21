// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

// A type that is not trivial for calls travels by reference to the one
// argument object the caller constructs, the callee moves from and the
// caller destroys. The checks pin that the address given to the call is
// the same temporary the constructor filled, with no copy in between.

struct Owner {
  int *val;
  Owner(int v);
  Owner(Owner &&other) : val(other.val) { other.val = nullptr; }
  ~Owner();
};

void sink(Owner o);

void pass_temporary(int v) { sink(Owner(v)); }

// LLVM-LABEL: @_Z14pass_temporaryi
// LLVM: %[[TMP:.*]] = alloca %struct.Owner, align 8
// LLVM: call void @_ZN5OwnerC1Ei(ptr {{.*}}%[[TMP]], i32 {{.*}})
// LLVM-CIR: call void @_Z4sink5Owner(ptr byref(%struct.Owner) align 8 %[[TMP]])
// LLVM-OGCG: call void @_Z4sink5Owner(ptr {{.*}}align 8 {{.*}}%[[TMP]])
// LLVM: call void @_ZN5OwnerD1Ev(ptr {{.*}}%[[TMP]])
