// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-incompatible-library-redeclaration -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-incompatible-library-redeclaration -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-incompatible-library-redeclaration -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// The user's memchr has another prototype, so the builtin's call goes
// through its address cast to the builtin's type.

char *memchr(char *, char, unsigned);

char *ordinary(char *p, char c) { return memchr(p, c, 3); }

void *builtin(const char *p, int c) { return __builtin_memchr(p, c, 32); }

// CIR-LABEL: cir.func {{.*}}@builtin(
// CIR: %[[ADDR:.*]] = cir.get_global @memchr : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s8i, !u32i) -> !cir.ptr<!s8i>>>
// CIR: %[[FN:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s8i, !u32i) -> !cir.ptr<!s8i>>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i, !u64i) -> !cir.ptr<!void>>>
// CIR: cir.call %[[FN]]({{%.*}}, {{%.*}}, {{%.*}}) nothrow : (!cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i, !u64i) -> !cir.ptr<!void>>>, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>

// LLVM-LABEL: define{{.*}} ptr @ordinary(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i8 noundef signext %{{.*}}, i32 noundef 3)
// LLVM: declare ptr @memchr(ptr noundef, i8 noundef signext, i32 noundef)
// LLVM-LABEL: define{{.*}} ptr @builtin(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef 32)
