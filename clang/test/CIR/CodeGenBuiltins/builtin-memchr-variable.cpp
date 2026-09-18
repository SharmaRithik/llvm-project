// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// A variable takes the memchr name. The builtin's call takes the same cast
// path as for a declaration of another type.

int memchr;

void *builtin(const char *p, int c) { return __builtin_memchr(p, c, 32); }

// CIR: cir.global external @memchr = #cir.int<0> : !s32i
// CIR-LABEL: cir.func {{.*}}@_Z7builtinPKci(
// CIR: %[[ADDR:.*]] = cir.get_global @memchr : !cir.ptr<!s32i>
// CIR: %[[FN:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!s32i> -> !cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i, !u64i) -> !cir.ptr<!void>>>
// CIR: cir.call %[[FN]]({{%.*}}, {{%.*}}, {{%.*}}) nothrow : (!cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i, !u64i) -> !cir.ptr<!void>>>, !cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>

// LLVM: @memchr = global i32 0
// LLVM-LABEL: define{{.*}} ptr @_Z7builtinPKci(
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef 32)
