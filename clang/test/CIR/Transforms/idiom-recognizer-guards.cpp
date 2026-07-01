// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s
// CHECK-NOT: cir.std.

// None of the calls below may be raised. Each one satisfies every recognizer
// check except the single guard it pins, so this test also fails if the
// recognizer starts creating verifier-invalid operations.

namespace std {
// Variadic, only viable for the all-pointer call in test_variadic.
char *find(char *first, ...);
// Result type differs from the iterator type.
int find(char *first, char *last, const char &value);
// Wrong arity.
char *find(char *first, char *last, const char &value, int n);
}
// Non-fundamental result type.
extern "C" __int128 strlen(const char *);

char *test_variadic(char *first, char *last, char *value) {
  return std::find(first, last, value);
}

int test_result_type(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}

char *test_arity(char *first, char *last, const char &value) {
  return std::find(first, last, value, 1);
}

__int128 test_strlen_result(const char *s) { return strlen(s); }
