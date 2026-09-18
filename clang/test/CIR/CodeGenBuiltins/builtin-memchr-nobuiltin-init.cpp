// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin -O2 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin -O2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// -fno-builtin must also reach the call emitted for a dynamic initializer.

char buf[4];
int dummy = (__builtin_memchr(buf, 1, 4), 0);

void f(const void *p, int c, __SIZE_TYPE__ n) { __builtin_memchr(p, c, n); }

// LLVM-LABEL: define{{.*}} void @_Z1fPKvim(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef %{{.*}})
// LLVM-LABEL: define internal void @_GLOBAL__sub_I_{{.*}}(
// LLVM: call ptr @memchr(ptr noundef nonnull @buf, i32 noundef 1, i64 noundef 4)
