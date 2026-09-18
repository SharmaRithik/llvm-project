// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-libcall-lowering %s -o %t.cir 2>&1 | FileCheck --check-prefix=CIR-BEFORE %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-passes -emit-llvm %s -o %t-nopasses.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-nopasses.ll %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-libcall-lowering %s -o /dev/null 2>&1 | FileCheck --check-prefix=ILP32 %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.i686.ll
// RUN: FileCheck --check-prefix=LLVM32 --input-file=%t.i686.ll %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-llvm %s -o %t.i686-classic.ll
// RUN: FileCheck --check-prefix=LLVM32 --input-file=%t.i686-classic.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnux32 -fclangir -emit-llvm %s -o %t.x32.ll
// RUN: FileCheck --check-prefix=LLVM32 --input-file=%t.x32.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnux32 -emit-llvm %s -o %t.x32-classic.ll
// RUN: FileCheck --check-prefix=LLVM32 --input-file=%t.x32-classic.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.a64.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.a64.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.a64-classic.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.a64-classic.ll %s
// RUN: %clang_cc1 -triple msp430-unknown-unknown -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-libcall-lowering %s -o /dev/null 2>&1 | FileCheck --check-prefix=I16 %s
// RUN: %clang_cc1 -triple msp430-unknown-unknown -fclangir -emit-llvm %s -o %t.msp430.ll
// RUN: FileCheck --check-prefix=LLVM16 --input-file=%t.msp430.ll %s
// RUN: %clang_cc1 -triple msp430-unknown-unknown -emit-llvm %s -o %t.msp430-classic.ll
// RUN: FileCheck --check-prefix=LLVM16 --input-file=%t.msp430-classic.ll %s

// The cir-libcall-lowering pass replaces cir.libc.memchr with a call to
// memchr. The CIR-BEFORE, ILP32 and I16 checks read the module printed
// before that pass.

void *test_char_memchr(const char arg[32]) {
  return __builtin_char_memchr(arg, 123, 32);
}

// CIR-BEFORE: module{{.*}}cir.int_type_width = 32 : i32{{.*}}cir.size_type_width = 64 : i32
// CIR-BEFORE-LABEL: @test_char_memchr
// CIR-BEFORE: %[[PATTERN:.*]] = cir.const #cir.int<123> : !s32i
// CIR-BEFORE: %[[LEN:.*]] = cir.const #cir.int<32> : !u64i
// CIR-BEFORE: {{%.*}} = cir.libc.memchr({{%.*}}, %[[PATTERN]], %[[LEN]])
// ILP32: module{{.*}}cir.int_type_width = 32 : i32{{.*}}cir.size_type_width = 32 : i32
// ILP32-LABEL: @test_char_memchr
// ILP32: %[[PATTERN32:.*]] = cir.const #cir.int<123> : !s32i
// ILP32: %[[LEN32:.*]] = cir.const #cir.int<32> : !u32i
// ILP32: cir.libc.memchr({{.*}}, %[[PATTERN32]], %[[LEN32]]) : !cir.ptr<!void>, !s32i, !u32i
// I16: module{{.*}}cir.int_type_width = 16 : i32{{.*}}cir.size_type_width = 16 : i32
// I16-LABEL: @test_char_memchr
// I16: %[[PATTERN16:.*]] = cir.const #cir.int<123> : !s16i
// I16: %[[LEN16:.*]] = cir.const #cir.int<32> : !u16i
// I16: cir.libc.memchr({{.*}}, %[[PATTERN16]], %[[LEN16]]) : !cir.ptr<!void>, !s16i, !u16i
// LLVM32-DAG: declare ptr @memchr(ptr noundef, i32 noundef, i32 noundef)
// LLVM32-DAG: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef 123, i32 noundef 32)
// LLVM16-DAG: declare ptr @memchr(ptr noundef, i16 noundef, i16 noundef)
// LLVM16-DAG: call ptr @memchr(ptr noundef %{{.*}}, i16 noundef 123, i16 noundef 32)

// CIR-LABEL: @test_char_memchr
// CIR: %[[PATTERN:.*]] = cir.const #cir.int<123> : !s32i
// CIR: %[[LEN:.*]] = cir.const #cir.int<32> : !u64i
// CIR: cir.call @memchr({{%.*}}, %[[PATTERN]], %[[LEN]]) nothrow : (!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>

// LLVM-LABEL: @test_char_memchr
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef 123, i64 noundef 32) #[[NOUNWIND:[0-9]+]]
// LLVM: ret ptr

void *test_memchr(const void *ptr, int val, __SIZE_TYPE__ size) {
  return __builtin_memchr(ptr, val, size);
}

// CIR-BEFORE-LABEL: @test_memchr
// CIR-BEFORE: {{%.*}} = cir.libc.memchr({{%.*}}, {{%.*}}, {{%.*}})

// CIR-LABEL: @test_memchr
// CIR: cir.call @memchr({{%.*}}, {{%.*}}, {{%.*}}) nothrow : (!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void>
// CIR: cir.func private @memchr(!cir.ptr<!void> {llvm.noundef}, !s32i {llvm.noundef}, !u64i {llvm.noundef}) -> !cir.ptr<!void> attributes {nothrow}

// LLVM-LABEL: @test_memchr
// LLVM: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i64 noundef %{{.*}}) #[[NOUNWIND]]
// LLVM: ret ptr
// LLVM: attributes #[[NOUNWIND]] = { nounwind }
// ILP32-LABEL: @test_memchr
// ILP32: cir.libc.memchr({{.*}}) : !cir.ptr<!void>, !s32i, !u32i
// I16-LABEL: @test_memchr
// I16: cir.libc.memchr({{.*}}) : !cir.ptr<!void>, !s16i, !u16i
// LLVM32-DAG: call ptr @memchr(ptr noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}})
// LLVM16-DAG: call ptr @memchr(ptr noundef %{{.*}}, i16 noundef %{{.*}}, i16 noundef %{{.*}})
