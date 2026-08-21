// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir --canonicalize | FileCheck %s --check-prefix=CIR

// A volatile asm whose result feeds nothing does things its constraints do
// not describe, so optimization must not remove it.  cir.asm used to report
// no memory effects, which let canonicalization erase every benchmark style
// barrier whose result is unused and time empty loops.  The asm below is
// exactly that shape: its updated operand is never read again.

void dead_volatile_asm(int x) {
  __asm__ volatile("" : "+r"(x));
}

// CHECK-LABEL: @dead_volatile_asm
// CHECK: call i32 asm sideeffect "", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
// CIR-LABEL: @dead_volatile_asm
// CIR: cir.asm

int keeps_live_volatile_asm(int x) {
  __asm__ volatile("" : "+r"(x));
  return x;
}

// CHECK-LABEL: @keeps_live_volatile_asm
// CHECK: call i32 asm sideeffect "", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
