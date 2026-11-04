// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

void use(int);

// CHECK-LABEL: define {{.*}} @_Z7forLoopi
void forLoop(int n) {
  // CHECK: br label %{{.*}}, !llvm.loop ![[FOR:[0-9]+]]
#pragma unroll
  for (int i = 0; i < n; ++i)
    use(i);
}

// CHECK-LABEL: define {{.*}} @_Z9whileLoopi
void whileLoop(int n) {
  // CHECK: br label %{{.*}}, !llvm.loop ![[WHILE:[0-9]+]]
#pragma unroll 4
  while (n--)
    use(n);
}

// CHECK-LABEL: define {{.*}} @_Z6doLoopi
void doLoop(int n) {
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[DO:[0-9]+]]
#pragma clang loop unroll(disable)
  do
    use(n);
  while (n--);
}

// CHECK-LABEL: define {{.*}} @_Z8fullLoopv
void fullLoop() {
  // CHECK: br label %{{.*}}, !llvm.loop ![[FULL:[0-9]+]]
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i)
    use(i);
}

// CHECK-LABEL: define {{.*}} @_Z8noPragmai
void noPragma(int n) {
  // CHECK: br label %{{.*}}, !llvm.loop ![[PLAIN:[0-9]+]]
  for (int i = 0; i < n; ++i)
    use(i);
}

// CHECK-DAG: ![[FOR]] = distinct !{![[FOR]], ![[MP:[0-9]+]], ![[ENABLE:[0-9]+]]}
// CHECK-DAG: ![[WHILE]] = distinct !{![[WHILE]], ![[MP]], ![[COUNT:[0-9]+]]}
// CHECK-DAG: ![[DO]] = distinct !{![[DO]], ![[MP]], ![[DISABLE:[0-9]+]]}
// CHECK-DAG: ![[FULL]] = distinct !{![[FULL]], ![[MP]], ![[UFULL:[0-9]+]]}
// CHECK-DAG: ![[PLAIN]] = distinct !{![[PLAIN]], ![[MP]]}
// CHECK-DAG: ![[MP]] = !{!"llvm.loop.mustprogress"}
// CHECK-DAG: ![[ENABLE]] = !{!"llvm.loop.unroll.enable"}
// CHECK-DAG: ![[COUNT]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK-DAG: ![[DISABLE]] = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: ![[UFULL]] = !{!"llvm.loop.unroll.full"}
