// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s
// CHECK-NOT: cir.std.

// Each call satisfies every recognizer check except the one guard it pins.

namespace std {
// Variadic, only viable for the all-pointer call in test_variadic.
char *find(char *first, ...);
// Result type differs from the iterator type.
int find(char *first, char *last, const char &value);
// Searched value type differs from the element type.
char *find(char *first, char *last, const int &value);
}

char *test_variadic(char *first, char *last, char *value) {
  return std::find(first, last, value);
}

int test_result_type(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}

char *test_pattern_type(char *first, char *last, const int &value) {
  return std::find(first, last, value);
}
