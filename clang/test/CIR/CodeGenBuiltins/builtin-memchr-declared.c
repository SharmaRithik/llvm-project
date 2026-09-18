// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void *memchr(const void *, int, __SIZE_TYPE__);

void *ordinary(const char *p, int c) { return memchr(p, c, 32); }

// CIR-LABEL: cir.func {{.*}}@ordinary(
// CIR: %[[LEN:.*]] = cir.const #cir.int<32> : !u64i
// CIR: cir.call @memchr({{%.*}}, {{%.*}}, %[[LEN]]) nothrow : (!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>
// CIR: cir.func private @memchr(!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>

// LLVM-LABEL: define{{.*}} ptr @ordinary(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef 32) #[[NOUNWIND:[0-9]+]]
// LLVM: declare ptr @memchr(ptr noundef, i32 noundef, i64 noundef)

void *builtin(const char *p, int c) { return __builtin_memchr(p, c, 32); }

// CIR-LABEL: cir.func {{.*}}@builtin(
// CIR: %[[LEN:.*]] = cir.const #cir.int<32> : !u64i
// CIR: cir.call @memchr({{%.*}}, {{%.*}}, %[[LEN]]) nothrow : (!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>

// LLVM-LABEL: define{{.*}} ptr @builtin(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef 32) #[[NOUNWIND]]
// LLVM: attributes #[[NOUNWIND]] = { nounwind }
