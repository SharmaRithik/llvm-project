// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O1 \
// RUN:   -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir --cir-loop-interchange -o - | FileCheck %s

#define N 64
#define M 10

static double A[N][M];
static double B[N][M];
static double C[N][N];
static double D[N][N];

// CHECK-LABEL: cir.func dso_local @triangular_tiled(
// CHECK-DAG: cir.alloca "i.tile"{{.*}}!cir.ptr<!s64i>
// CHECK-DAG: cir.alloca "j.tile"
// CHECK-DAG: cir.alloca "k.tile"
// CHECK: #cir.int<32>
// CHECK: #cir.int<4>
void triangular_tiled(double (*restrict c)[N], double (*restrict a)[M],
                      double (*restrict b)[M]) {
  int i, j, k;
  for (i = 0; i < N; ++i) {
    for (j = 0; j <= i; ++j)
      c[i][j] *= 2.0;
    for (k = 0; k < M; ++k)
      for (j = 0; j <= i; ++j)
        c[i][j] += a[j][k] * b[i][k] + b[j][k] * a[i][k];
  }
}

// CHECK-LABEL: cir.func dso_local @triangular_may_alias(
// CHECK-NOT: cir.alloca "i.tile"
// CHECK: cir.return
void triangular_may_alias(double (*c)[N], double (*a)[M], double (*b)[M]) {
  int i, j, k;
  for (i = 0; i < N; ++i) {
    for (j = 0; j <= i; ++j)
      c[i][j] *= 2.0;
    for (k = 0; k < M; ++k)
      for (j = 0; j <= i; ++j)
        c[i][j] += a[j][k] * b[i][k] + b[j][k] * a[i][k];
  }
}

// CHECK-LABEL: cir.func dso_local @triangular_no_scale()
// CHECK-NOT: cir.alloca "i.tile"
// CHECK: cir.return
void triangular_no_scale(void) {
  int i, j, k;
  for (i = 0; i < N; ++i)
    for (k = 0; k < M; ++k)
      for (j = 0; j <= i; ++j)
        C[i][j] += A[j][k] * B[i][k];
}

// CHECK-LABEL: cir.func dso_local @triangular_different_scale()
// CHECK-NOT: cir.alloca "i.tile"
// CHECK: cir.return
void triangular_different_scale(void) {
  int i, j, k;
  for (i = 0; i < N; ++i) {
    for (j = 0; j <= i; ++j)
      D[i][j] *= 2.0;
    for (k = 0; k < M; ++k)
      for (j = 0; j <= i; ++j)
        C[i][j] += A[j][k] * B[i][k];
  }
}
