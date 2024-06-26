// RUN: rm -rf %t
// RUN: split-file %s %t

//--- missing_header.c
#include "missing.h"

// RUN: not clang-scan-deps -- %clang -c %t/missing_tu.c 2>%t/missing_tu.errs
// RUN: echo EOF >> %t/missing_tu.errs
// RUN: cat %t/missing_tu.errs | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-MISSING-TU -DPREFIX=%/t
// CHECK-MISSING-TU: Error while scanning dependencies for [[PREFIX]]/missing_tu.c
// CHECK-MISSING-TU-NEXT: error: no such file or directory: '[[PREFIX]]/missing_tu.c'
// CHECK-MISSING-TU-NEXT: error: no input files
// CHECK-MISSING-TU-NEXT: error:
// CHECK-MISSING-TU-NEXT: EOF

// RUN: not clang-scan-deps -- %clang -c %t/missing_header.c 2>%t/missing_header.errs
// RUN: echo EOF >> %t/missing_header.errs
// RUN: cat %t/missing_header.errs | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-MISSING-HEADER -DPREFIX=%/t
// CHECK-MISSING-HEADER: Error while scanning dependencies for [[PREFIX]]/missing_header.c
// CHECK-MISSING-HEADER-NEXT: fatal error: 'missing.h' file not found
// CHECK-MISSING-HEADER-NEXT: EOF
