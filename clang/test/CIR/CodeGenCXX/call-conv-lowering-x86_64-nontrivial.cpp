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

// The call below also carries small record arguments that the rewrite
// coerces through fresh slots it inserts right before the call. Those
// inserted reads and writes must not scare the byref path away from
// reusing the constructed temporary's address.

struct It { Owner *p; };
struct Cmp { bool (*f)(const Owner &, const Owner &); };

void sink_mixed(It first, long n, Owner o, Cmp comp);

void pass_mixed(It first, long n, Cmp comp, int v) {
  sink_mixed(first, n, Owner(v), comp);
}

// LLVM-LABEL: @_Z10pass_mixed2Itl3Cmpi
// LLVM: %[[MTMP:.*]] = alloca %struct.Owner, align 8
// LLVM: call void @_ZN5OwnerC1Ei(ptr {{.*}}%[[MTMP]], i32 {{.*}})
// LLVM-CIR: call void @_Z10sink_mixed2Itl5Owner3Cmp(ptr %{{.*}}, i64 {{.*}}, ptr byref(%struct.Owner) align 8 %[[MTMP]], ptr %{{.*}})
// LLVM-OGCG: call void @_Z10sink_mixed2Itl5Owner3Cmp(ptr %{{.*}}, i64 {{.*}}, ptr {{.*}}align 8 {{.*}}%[[MTMP]], ptr %{{.*}})
// LLVM: call void @_ZN5OwnerD1Ev(ptr {{.*}}%[[MTMP]])

// A record operand wider than one eightbyte coerces through a slot the
// rewrite reaches with a cast, and a type with a trivial copy but a
// nontrivial destructor arrives without a constructor call. Both shapes
// must still hand the callee the temporary itself.

struct Trip { int a; int b; int c; };
struct Guard {
  long a, b, c;
  ~Guard();
};

void sink_trip(Trip t, Owner o);
void sink_guard(Trip t, Guard g);

void pass_trip(Trip t, int v) { sink_trip(t, Owner(v)); }

// LLVM-LABEL: @_Z9pass_trip4Tripi
// LLVM: %[[TTMP:.*]] = alloca %struct.Owner, align 8
// LLVM: call void @_ZN5OwnerC1Ei(ptr {{.*}}%[[TTMP]], i32 {{.*}})
// LLVM-CIR: call void @_Z9sink_trip4Trip5Owner(i64 %{{.*}}, i32 %{{.*}}, ptr byref(%struct.Owner) align 8 %[[TTMP]])
// LLVM-OGCG: call void @_Z9sink_trip4Trip5Owner(i64 %{{.*}}, i32 %{{.*}}, ptr {{.*}}align 8 {{.*}}%[[TTMP]])
// LLVM: call void @_ZN5OwnerD1Ev(ptr {{.*}}%[[TTMP]])

void pass_guard(Trip t, Guard g) { sink_guard(t, static_cast<Guard &&>(g)); }

// LLVM-LABEL: @_Z10pass_guard4Trip5Guard
// LLVM: %[[GTMP:.*]] = alloca %struct.Guard, align 8
// LLVM-CIR: call void @_Z10sink_guard4Trip5Guard(i64 %{{.*}}, i32 %{{.*}}, ptr byref(%struct.Guard) align 8 %[[GTMP]])
// LLVM-OGCG: call void @_Z10sink_guard4Trip5Guard(i64 %{{.*}}, i32 %{{.*}}, ptr {{.*}}align 8 {{.*}}%[[GTMP]])
// LLVM: call void @_ZN5GuardD1Ev(ptr {{.*}}%[[GTMP]])
