# RUN: llvm-exegesis -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 --benchmark-phase=assemble-measured-code -mode=latency -opcode-name=ADD8 | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD8
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-DAG: - '[[REG1:[A-Z0-9]+]]=0x0'
CHECK-DAG: ...
