// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all -clangir-enable-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// CIR: IR Dump After IdiomRecognizer: cir-idiom-recognizer

// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=RAISED
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.cir

namespace std {
template <class Iter, class T> Iter find(Iter, Iter, const T &);
}
extern "C" unsigned long strlen(const char *);

char *test_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// std::find is raised, and lowered back to the original call in operand order
// when nothing consumed it.
// RAISED: cir.std.find(
// RAISED-SAME: @_ZSt4find
// FINAL: %[[FIRST:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s8i>>
// FINAL: %[[LAST:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s8i>>
// FINAL: %[[VALUE:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s8i>>
// FINAL: cir.call @_ZSt4find{{.*}}(%[[FIRST]], %[[LAST]], %[[VALUE]])
// FINAL-SAME: {llvm.noundef}

unsigned long test_strlen(const char *s) { return strlen(s); }
// The call attributes survive the raise and lower back round trip.
// RAISED: cir.std.strlen(
// RAISED-SAME: @strlen
// FINAL: cir.call @strlen(%{{.*}}) nothrow

// A function merely named like the std one is not raised.
char *find(char *first, char *last, const char &value);
char *test_non_std_find(char *first, char *last, const char &value) {
  return find(first, last, value);
}
// RAISED: cir.call @_Z4find

// A member function named find in the std namespace is not std::find. The
// signature is chosen so the call would satisfy the recognizer's type checks,
// pinning the member exclusion itself.
namespace std {
struct string {
  string *find(string *first, string *last);
};
}

std::string *test_member_find(std::string &s, std::string *f, std::string *l) {
  return s.find(f, l);
}
// RAISED: cir.call @_ZNSt6string4find
// RAISED-NOT: cir.std.find
