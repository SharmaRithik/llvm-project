// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O1 \
// RUN:   -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir \
// RUN:   "--cir-loop-interchange=emit-analysis-remarks=true" \
// RUN:   -o /dev/null 2>&1 | FileCheck %s
// RUN: cir-opt %t.cir --cir-loop-interchange -o %t.once.cir
// RUN: cir-opt %t.once.cir --cir-loop-interchange -o %t.twice.cir
// RUN: diff %t.once.cir %t.twice.cir && FileCheck %s --check-prefix=INTERCHANGE < %t.once.cir
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

// CHECK-DAG: recognized loop nest in @reused_induction_storage outer init constant condition induction lt constant inner init constant condition induction lt induction memory safe profitability profitable
void reused_induction_storage(void) {
  int i, j;
  for (i = 1; i < N; ++i)
    for (j = 0; j < i; ++j)
      B[j][i] = A[j][i];
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      D[i][j] = C[i][j];
}

// CHECK-NOT: recognized loop nest in @mutated_induction_storage
void mutated_induction_storage(void) {
  int i, j;
  for (i = 1; i < N; ++i)
    for (j = 0; j < i; ++j) {
      B[j][i] = A[j][i];
      if (B[j][i] == 0.0)
        i = N;
    }
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

void unsigned_scaled(void) {
  for (unsigned i = 1; i < N / 2; ++i)
    for (unsigned j = 0; j < 2 * i; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_mul outer init constant condition induction lt constant inner init constant condition mul(induction,induction) lt constant memory safe
void tri_mul(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j * i < N; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_mul_mismatched_extent {{.*}} memory safe profitability profitable
void tri_mul_mismatched_extent(void) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j * i < 127; ++j)
      B[j][i] = A[j][i];
}

// CHECK-DAG: recognized loop nest in @tri_arg outer init constant condition induction lt constant inner init symbol condition induction lt constant memory safe profitability profitable
long tri_arg(long lo) {
  long sum = 0;
  for (long i = 0; i < N; ++i)
    for (long j = lo; j < N; ++j)
      sum += L[j][i];
  return sum;
}

// CHECK-DAG: recognized loop nest in @floating_reduction {{.*}} memory unsupported address
double floating_reduction(int lo) {
  double sum = 0.0;
  for (int i = 0; i < N; ++i)
    for (int j = lo; j < N; ++j)
      sum += A[j][i];
  return sum;
}

// CHECK-DAG: recognized loop nest in @subtract_reduction {{.*}} memory unsupported address
long subtract_reduction(long lo) {
  long sum = 0;
  for (long i = 0; i < N; ++i)
    for (long j = lo; j < N; ++j)
      sum -= L[j][i];
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

// CHECK-DAG: recognized loop nest in @restricted_pointer_access {{.*}} memory safe profitability profitable
void restricted_pointer_access(double (*restrict a)[N],
                               double (*restrict b)[N]) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      b[j][i] = a[j][i];
}

// CHECK-DAG: recognized loop nest in @unrestricted_pointer_access {{.*}} memory unsupported address
void unrestricted_pointer_access(double (*a)[N], double (*b)[N]) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      b[j][i] = a[j][i];
}

// CHECK-DAG: recognized loop nest in @restricted_shifted_access {{.*}} memory potential dependence
void restricted_shifted_access(double (*restrict a)[N]) {
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      a[j][i] = a[j][i - 1];
}

// CHECK-DAG: recognized loop nest in @reassigned_restricted_pointer {{.*}} memory unsupported address
void reassigned_restricted_pointer(double (*restrict a)[N],
                                   double (*restrict b)[N]) {
  a = b;
  for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
      a[j][i] = 0.0;
}

// CHECK-DAG: recognized anchored loop band in @anchored_single_phase outer init add(induction,constant) inner candidates 1
void anchored_single_phase(void) {
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      B[i][j] = 0.0;
      for (int k = 0; k < N; ++k)
        B[i][j] += A[k][i] * A[k][j];
      C[j][i] = B[i][j];
    }
}

// CHECK-DAG: recognized anchored loop band in @anchored_multiple_phases outer init add(induction,constant) inner candidates 2
void anchored_multiple_phases(void) {
  for (int k = 0; k < N; ++k)
    for (int j = k + 1; j < N; ++j) {
      B[k][j] = 0.0;
      for (int i = 0; i < N; ++i)
        B[k][j] += A[i][k] * A[i][j];
      for (int i = 0; i < N; ++i)
        C[i][j] = A[i][j] - B[k][j];
    }
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

// INTERCHANGE-LABEL: cir.func dso_local @tri_fill()
// INTERCHANGE: [[FILLI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[FILLJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[FILLZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[FILLZERO]], [[FILLJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[FILLJCOND:%[0-9]+]] = cir.load{{.*}} [[FILLJ]]
// INTERCHANGE-NEXT: [[FILLJBOUND:%[0-9]+]] = cir.const #cir.int<127>
// INTERCHANGE-NEXT: [[FILLJCMP:%[0-9]+]] = cir.cmp lt [[FILLJCOND]], [[FILLJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: [[FILLJVALUE:%[0-9]+]] = cir.load{{.*}} [[FILLJ]]
// INTERCHANGE-NEXT: [[FILLISTART:%[0-9]+]] = cir.inc nsw [[FILLJVALUE]]
// INTERCHANGE-NEXT: cir.store{{.*}} [[FILLISTART]], [[FILLI]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_ldlt()
// INTERCHANGE: [[LDLTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[LDLTJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[LDLTZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[LDLTZERO]], [[LDLTJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[LDLTJCOND:%[0-9]+]] = cir.load{{.*}} [[LDLTJ]]
// INTERCHANGE-NEXT: [[LDLTJBOUND:%[0-9]+]] = cir.const #cir.int<127>
// INTERCHANGE-NEXT: [[LDLTJCMP:%[0-9]+]] = cir.cmp lt [[LDLTJCOND]], [[LDLTJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: [[LDLTJVALUE:%[0-9]+]] = cir.load{{.*}} [[LDLTJ]]
// INTERCHANGE-NEXT: [[LDLTISTART:%[0-9]+]] = cir.inc nsw [[LDLTJVALUE]]
// INTERCHANGE-NEXT: cir.store{{.*}} [[LDLTISTART]], [[LDLTI]]

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

// INTERCHANGE-LABEL: cir.func dso_local @tri_variant()
// INTERCHANGE: [[VARIANTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[VARIANTJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[VARIANTZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[VARIANTZERO]], [[VARIANTJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[VARIANTJCOND:%[0-9]+]] = cir.load{{.*}} [[VARIANTJ]]
// INTERCHANGE-NEXT: [[VARIANTJBOUND:%[0-9]+]] = cir.const #cir.int<126>
// INTERCHANGE-NEXT: [[VARIANTJCMP:%[0-9]+]] = cir.cmp lt [[VARIANTJCOND]], [[VARIANTJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: [[VARIANTJVALUE:%[0-9]+]] = cir.load{{.*}} [[VARIANTJ]]
// INTERCHANGE-NEXT: [[VARIANTCOEFF:%[0-9]+]] = cir.const #cir.int<2>
// INTERCHANGE-NEXT: [[VARIANTQUOT:%[0-9]+]] = cir.div [[VARIANTJVALUE]], [[VARIANTCOEFF]]
// INTERCHANGE-NEXT: [[VARIANTSTART:%[0-9]+]] = cir.inc [[VARIANTQUOT]]
// INTERCHANGE-NEXT: cir.store{{.*}} [[VARIANTSTART]], [[VARIANTI]]

// INTERCHANGE-LABEL: cir.func dso_local @unsigned_scaled()
// INTERCHANGE: [[UNSIGNEDI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[UNSIGNEDONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[UNSIGNEDONE]], [[UNSIGNEDI]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_mul()
// INTERCHANGE: [[MULI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[MULIONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: [[MULJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[MULJZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[MULJZERO]], [[MULJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[MULJCOND:%[0-9]+]] = cir.load{{.*}} [[MULJ]]
// INTERCHANGE-NEXT: [[MULJBOUND:%[0-9]+]] = cir.const #cir.int<128>
// INTERCHANGE-NEXT: [[MULJCMP:%[0-9]+]] = cir.cmp lt [[MULJCOND]], [[MULJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: cir.store{{.*}} [[MULIONE]], [[MULI]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[MULICOND:%[0-9]+]] = cir.load{{.*}} [[MULI]]
// INTERCHANGE-NEXT: [[MULJVALUE:%[0-9]+]] = cir.load{{.*}} [[MULJ]]
// INTERCHANGE-NEXT: [[MULZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: [[MULISZERO:%[0-9]+]] = cir.cmp eq [[MULJVALUE]], [[MULZERO]]
// INTERCHANGE-NEXT: [[MULONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: [[MULDIVISOR:%[0-9]+]] = cir.select if [[MULISZERO]] then [[MULONE]] else [[MULJVALUE]]
// INTERCHANGE-NEXT: [[MULREDUCED:%[0-9]+]] = cir.const #cir.int<127>
// INTERCHANGE-NEXT: [[MULQUOT:%[0-9]+]] = cir.div [[MULREDUCED]], [[MULDIVISOR]]
// INTERCHANGE-NEXT: [[MULBOUND:%[0-9]+]] = cir.inc [[MULQUOT]]
// INTERCHANGE-NEXT: [[MULICMP:%[0-9]+]] = cir.cmp lt [[MULICOND]], [[MULBOUND]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_mul_mismatched_extent()
// INTERCHANGE: [[MULMISMATCHI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[MULMISMATCHONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[MULMISMATCHONE]], [[MULMISMATCHI]]

// INTERCHANGE-LABEL: cir.func dso_local @tri_arg(
// INTERCHANGE: [[ARGLO:%[0-9]+]] = cir.alloca "lo"
// INTERCHANGE: [[ARGSUM:%[0-9]+]] = cir.alloca "sum"
// INTERCHANGE: [[ARGI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[ARGIZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: [[ARGJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[ARGJSTART:%[0-9]+]] = cir.load{{.*}} [[ARGLO]]
// INTERCHANGE-NEXT: cir.store{{.*}} [[ARGJSTART]], [[ARGJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE-NEXT: [[ARGJCOND:%[0-9]+]] = cir.load{{.*}} [[ARGJ]]
// INTERCHANGE-NEXT: [[ARGJBOUND:%[0-9]+]] = cir.const #cir.int<128>
// INTERCHANGE-NEXT: [[ARGJCMP:%[0-9]+]] = cir.cmp lt [[ARGJCOND]], [[ARGJBOUND]]
// INTERCHANGE: } body {
// INTERCHANGE-NEXT: cir.store{{.*}} [[ARGIZERO]], [[ARGI]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE: [[ARGREDUCE:%[0-9]+]] = cir.add {{%[0-9]+}}, {{%[0-9]+}} : !s64i

// INTERCHANGE-LABEL: cir.func dso_local @floating_reduction(
// INTERCHANGE: [[FLOATI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[FLOATZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[FLOATZERO]], [[FLOATI]]

// INTERCHANGE-LABEL: cir.func dso_local @subtract_reduction(
// INTERCHANGE: [[SUBI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[SUBZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[SUBZERO]], [[SUBI]]

// INTERCHANGE-LABEL: cir.func dso_local @shifted_dependence()
// INTERCHANGE: [[SHIFTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[SHIFTONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[SHIFTONE]], [[SHIFTI]]

// INTERCHANGE-LABEL: cir.func dso_local @restricted_pointer_access(
// INTERCHANGE: [[RESTRICTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[RESTRICTJ:%[0-9]+]] = cir.alloca "j"
// INTERCHANGE-NEXT: [[RESTRICTZERO:%[0-9]+]] = cir.const #cir.int<0>
// INTERCHANGE-NEXT: cir.store{{.*}} [[RESTRICTZERO]], [[RESTRICTJ]]
// INTERCHANGE-NEXT: cir.for : cond {
// INTERCHANGE: } body {
// INTERCHANGE: cir.store{{.*}}, [[RESTRICTI]]
// INTERCHANGE-NEXT: cir.for : cond {

// INTERCHANGE-LABEL: cir.func dso_local @unrestricted_pointer_access(
// INTERCHANGE: [[UNRESTRICTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[UNRESTRICTONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[UNRESTRICTONE]], [[UNRESTRICTI]]

// INTERCHANGE-LABEL: cir.func dso_local @restricted_shifted_access(
// INTERCHANGE: [[RESTRICTSHIFTI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[RESTRICTSHIFTONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[RESTRICTSHIFTONE]], [[RESTRICTSHIFTI]]

// INTERCHANGE-LABEL: cir.func dso_local @reassigned_restricted_pointer(
// INTERCHANGE: [[REASSIGNI:%[0-9]+]] = cir.alloca "i"
// INTERCHANGE-NEXT: [[REASSIGNONE:%[0-9]+]] = cir.const #cir.int<1>
// INTERCHANGE-NEXT: cir.store{{.*}} [[REASSIGNONE]], [[REASSIGNI]]

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
