// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-libcall-lowering %s -o %t.cir 2>&1 | FileCheck --check-prefix=CIR-BEFORE %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin -O2 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin -O2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memchr -fclangir -emit-cir %s -o %t.named.cir
// RUN: FileCheck --check-prefix=CIR-NAMED --input-file=%t.named.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memchr -fclangir -emit-llvm %s -o %t.named-cir.ll
// RUN: FileCheck --check-prefix=LLVM-NAMED --input-file=%t.named-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-memchr -emit-llvm %s -o %t.named.ll
// RUN: FileCheck --check-prefix=LLVM-NAMED --input-file=%t.named.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-strlen -fclangir -emit-cir %s -o %t.other.cir
// RUN: FileCheck --check-prefix=CIR-OTHER --input-file=%t.other.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-strlen -fclangir -emit-llvm %s -o %t.other-cir.ll
// RUN: FileCheck --check-prefix=LLVM-OTHER --input-file=%t.other-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-builtin-strlen -emit-llvm %s -o %t.other.ll
// RUN: FileCheck --check-prefix=LLVM-OTHER --input-file=%t.other.ll %s

// The call carries the same no builtin list as a call the user writes, so
// LLVM keeps it under -fno-builtin even when the result is unused.

void f(const void *p, int c, __SIZE_TYPE__ n) { __builtin_memchr(p, c, n); }

// CIR-BEFORE-LABEL: cir.func {{.*}}@f(
// CIR-BEFORE: cir.libc.memchr({{%.*}}, {{%.*}}, {{%.*}}) {nobuiltins = []} : !cir.ptr<!void>, !s32i, !u64i

// CIR-LABEL: cir.func {{.*}}@f(
// CIR: cir.call @memchr({{%.*}}, {{%.*}}, {{%.*}}) nothrow {nobuiltin, nobuiltins = []} : (!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>
// CIR: cir.func private @memchr(!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void> attributes {nobuiltins = [], nothrow}

// LLVM-LABEL: define{{.*}} void @f(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef %{{.*}}) #[[NOBUILTIN:[0-9]+]]
// LLVM: attributes #[[NOBUILTIN]] = { nobuiltin nounwind "no-builtins" }

// CIR-NAMED-LABEL: cir.func {{.*}}@f(
// CIR-NAMED: cir.call @memchr({{%.*}}, {{%.*}}, {{%.*}}) nothrow {nobuiltin, nobuiltins = ["memchr"]} : (
// CIR-NAMED: cir.func private @memchr({{.*}}) -> !cir.ptr<!void> attributes {nobuiltins = ["memchr"], nothrow}

// LLVM-NAMED-LABEL: define{{.*}} void @f(
// LLVM-NAMED: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef %{{.*}}) #[[NAMED:[0-9]+]]
// LLVM-NAMED: attributes #[[NAMED]] = { nobuiltin nounwind "no-builtin-memchr" }

// CIR-OTHER-LABEL: cir.func {{.*}}@f(
// CIR-OTHER: cir.call @memchr({{%.*}}, {{%.*}}, {{%.*}}) nothrow {nobuiltins = ["strlen"]} : (
// CIR-OTHER: cir.func private @memchr({{.*}}) -> !cir.ptr<!void> attributes {nobuiltins = ["strlen"], nothrow}

// LLVM-OTHER-LABEL: define{{.*}} void @f(
// LLVM-OTHER: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef %{{.*}}) #[[OTHER:[0-9]+]]
// LLVM-OTHER: attributes #[[OTHER]] = { nounwind "no-builtin-strlen" }
