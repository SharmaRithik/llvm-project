// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct S {
  ~S();
};

void *f(const char *p, int c) {
  S s;
  return __builtin_memchr(p, c, 32);
}

// LLVM-LABEL: define{{.*}} ptr @_Z1fPKci(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef 32) #[[NOUNWIND:[0-9]+]]
// LLVM-NOT: landingpad
// LLVM: attributes #[[NOUNWIND]] = { nounwind }
