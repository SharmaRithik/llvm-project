// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O1 \
// RUN:   -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir \
// RUN:   "--cir-loop-interchange=emit-analysis-remarks=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s
// RUN: cir-opt %t.cir \
// RUN:   --cir-loop-interchange \
// RUN:   | FileCheck %s --check-prefix=INTERCHANGE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O3 \
// RUN:   -floop-interchange -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=INTERCHANGE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O3 \
// RUN:   -floop-interchange -fno-loop-interchange -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOINTERCHANGE

#define N 128
#define K 8

static double A[N][N];
static double B[N][N];
static double C[N][N];
static double D[N][N];
static long L[N][N];

// CHECK-DAG: recognized loop nest in @tri_upper outer init constant condition induction lt constant inner init constant condition induction lt induction memory safe
void tri_upper(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_fill outer init constant condition induction lt constant inner init constant condition induction lt induction memory safe
void tri_fill(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[j][i] = 0.0;
}

// CHECK-DAG: recognized loop nest in @tri_ldlt outer init constant condition induction lt constant inner init constant condition induction lt induction memory safe
void tri_ldlt(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[j][i] -= A[j][i] * C[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_lower outer init constant condition induction lt constant inner init induction condition induction lt constant memory safe
void tri_lower(void) {
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j)
      B[i][j] = A[j][i] + C[j][i] + D[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_addk outer init constant condition induction lt sub(constant,constant) inner init constant condition induction lt add(induction,constant) memory safe
void tri_addk(void) {
  for (int i = 0; i < N - K; ++i)
    for (int j = 0; j < i + K; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_mismatched_offset {{.*}} memory safe profitability profitable
void tri_mismatched_offset(void) {
  for (int i = 0; i < N - K; ++i)
    for (int j = 0; j < i + K + 1; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_variant outer init constant condition induction lt div(constant,constant) inner init constant condition induction lt mul(constant,induction) memory safe
void tri_variant(void) {
  for (int i = 1; i < N / 2; ++i)
    for (int j = 0; j < 2 * i; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_mul outer init constant condition induction lt constant inner init constant condition mul(induction,induction) lt constant memory safe
void tri_mul(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j * i < N; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_arg outer init constant condition induction lt constant inner init symbol condition induction lt constant memory unsupported address
long tri_arg(long lo) {
  long sum = 0;
  for (long i = 0; i < N; ++i)
    for (long j = lo; j < N; ++j)
      sum += L[j][i];
  return sum;
}

// CHECK-DAG: recognized loop nest in @shifted_dependence {{.*}} memory potential dependence
void shifted_dependence(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[j][i] = B[j][i - 1];
}

// CHECK-DAG: recognized loop nest in @pointer_access {{.*}} memory unsupported address
void pointer_access(double *p) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      p[j * N + i] = 0.0;
}

extern void opaque(void);

// CHECK-DAG: recognized loop nest in @unknown_call {{.*}} memory unsupported operation
void unknown_call(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      opaque();
}

// CHECK-DAG: recognized loop nest in @already_contiguous {{.*}} memory safe profitability not profitable
void already_contiguous(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[i][j] = A[i][j];
}

// CHECK-DAG: recognized loop nest in @balanced_layout {{.*}} memory safe profitability not profitable
void balanced_layout(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[j][i] = A[i][j];
}

// CHECK-NOT: recognized loop nest in @rejected_volatile
void rejected_volatile(void) {
  for (volatile int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      B[j][i] = A[j][i];
}

// INTERCHANGE-LABEL: cir.func dso_local @tri_upper()
// INTERCHANGE: [[I:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[J:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[ZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[ZERO]], [[J]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[JCOND:%[0-9]+]] = cir.load{{.*}} [[J]]
// INTERCHANGE-NEXT: [[JBOUND:%[0-9]+]] = cir.const #cir.int<127>
// INTERCHANGE-NEXT: [[JCMP:%[0-9]+]] = cir.cmp lt [[JCOND]], [[JBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: [[JVALUE:%[0-9]+]] = cir.load{{.*}} [[J]]
// INTERCHANGE-NEXT: [[ISTART:%[0-9]+]] = cir.inc nsw [[JVALUE]]
// INTERCHANGE-NEXT: cir.store{{.*}} [[ISTART]], [[I]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[ICOND:%[0-9]+]] = cir.load{{.*}} [[I]]
// INTERCHANGE-NEXT: [[IBOUND:%[0-9]+]] = cir.const #cir.int<128>
// INTERCHANGE-NEXT: [[ICMP:%[0-9]+]] = cir.cmp lt [[ICOND]], [[IBOUND]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_lower()
// INTERCHANGE: [[LOWERI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[LOWERZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: [[LOWERJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: cir.store{{.*}} [[LOWERZERO]], [[LOWERJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[LOWERJCOND:%[0-9]+]] = cir.load{{.*}} [[LOWERJ]]
// INTERCHANGE-NEXT: [[LOWERJBOUND:%[0-9]+]] = cir.const #cir.int<128>
// INTERCHANGE-NEXT: [[LOWERJCMP:%[0-9]+]] = cir.cmp lt [[LOWERJCOND]], [[LOWERJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: cir.store{{.*}} [[LOWERZERO]], [[LOWERI]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[LOWERICOND:%[0-9]+]] = cir.load{{.*}} [[LOWERI]]
// INTERCHANGE-NEXT: [[LOWERJVALUE:%[0-9]+]] = cir.load{{.*}} [[LOWERJ]]
// INTERCHANGE-NEXT: [[LOWERICMP:%[0-9]+]] = cir.cmp le [[LOWERICOND]], [[LOWERJVALUE]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_addk()
// INTERCHANGE: [[ADDKI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[ADDKIZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: [[ADDKJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[ADDKJZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[ADDKJZERO]], [[ADDKJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[ADDKJCOND:%[0-9]+]] = cir.load{{.*}} [[ADDKJ]]
// INTERCHANGE-NEXT: [[ADDKJBOUND:%[0-9]+]] = cir.const #cir.int<127>
// INTERCHANGE-NEXT: [[ADDKJCMP:%[0-9]+]] = cir.cmp lt [[ADDKJCOND]], [[ADDKJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: [[ADDKJVALUE:%[0-9]+]] = cir.load{{.*}} [[ADDKJ]]
// INTERCHANGE-NEXT: [[ADDKK:%[0-9]+]] = cir.const #cir.int<8>
// INTERCHANGE-NEXT: [[ADDKBELOW:%[0-9]+]] = cir.cmp lt [[ADDKJVALUE]], [[ADDKK]]
// INTERCHANGE-NEXT: [[ADDKDIFF:%[0-9]+]] = cir.sub [[ADDKJVALUE]], [[ADDKK]]
// INTERCHANGE-NEXT: [[ADDKADJUST:%[0-9]+]] = cir.inc [[ADDKDIFF]]
// INTERCHANGE-NEXT: [[ADDKSTART:%[0-9]+]] = cir.select if [[ADDKBELOW]] then [[ADDKIZERO]] else [[ADDKADJUST]]
// INTERCHANGE-NEXT: cir.store{{.*}} [[ADDKSTART]], [[ADDKI]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_mismatched_offset()
// INTERCHANGE: [[MISMATCHI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[MISMATCHZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[MISMATCHZERO]], [[MISMATCHI]]

// INTERCHANGE-LABEL: cir.func dso_local @shifted_dependence()
// INTERCHANGE: [[SHIFTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[SHIFTONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[SHIFTONE]], [[SHIFTI]]

// INTERCHANGE-LABEL: cir.func dso_local @already_contiguous()
// INTERCHANGE: [[CONTIGI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[CONTIGONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[CONTIGONE]], [[CONTIGI]]

// INTERCHANGE-LABEL: cir.func dso_local @balanced_layout()
// INTERCHANGE: [[BALANCEDI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[BALANCEDONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[BALANCEDONE]], [[BALANCEDI]]

// NOINTERCHANGE-LABEL: cir.func dso_local @tri_upper()
// NOINTERCHANGE: [[NOI:%[0-9]+]] = cir.alloca "i"
// NOINTERCHANGE-NEXT: [[NOONE:%[0-9]+]] = cir.const #cir.int<1>
// NOINTERCHANGE-NEXT: cir.store{{.*}} [[NOONE]], [[NOI]]
// NOINTERCHANGE-NEXT: cir.for : cond {
