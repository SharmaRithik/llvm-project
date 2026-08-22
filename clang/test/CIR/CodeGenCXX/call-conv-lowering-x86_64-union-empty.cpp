// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// A union's empty record members hold no data bytes, so classic CodeGen
// sizes the coercion from the remaining members alone. The same source
// runs through both pipelines and one set of checks pins the signatures,
// so a divergence in either fails. The shapes mirror libc++'s
// std::variant storage union and std::format argument value union.

struct Empty {};
struct Alt {
  Empty e;
};
union VariantShaped {
  signed char dummy;
  Alt alt;
};
union InnerEmpty {};
union VariantTail {
  signed char dummy;
  Alt alt;
  InnerEmpty tail;
};
struct Monostate {};
union FormatShaped {
  Monostate m;
  bool b;
  int i;
  const void *p;
  long long ll;
};
union EmptyOnly {
  Monostate m;
};
union ArrayEmpty {
  Monostate m[3];
  int i;
};

VariantShaped pass_variant(VariantShaped u);
VariantTail pass_tail(VariantTail u);
FormatShaped pass_format(FormatShaped u);
EmptyOnly pass_empty_only(EmptyOnly u);
ArrayEmpty pass_array_empty(ArrayEmpty u);

VariantShaped call_variant(VariantShaped u) { return pass_variant(u); }
// LLVM: define{{.*}} i8 @_Z12call_variant13VariantShaped(i8 %{{.+}})
// LLVM:   call{{.*}} i8 @_Z12pass_variant13VariantShaped(i8 %{{.+}})

VariantTail call_tail(VariantTail u) { return pass_tail(u); }
// LLVM: define{{.*}} i8 @_Z9call_tail11VariantTail(i8 %{{.+}})
// LLVM:   call{{.*}} i8 @_Z9pass_tail11VariantTail(i8 %{{.+}})

FormatShaped call_format(FormatShaped u) { return pass_format(u); }
// LLVM: define{{.*}} ptr @_Z11call_format12FormatShaped(ptr %{{.+}})
// LLVM:   call{{.*}} ptr @_Z11pass_format12FormatShaped(ptr %{{.+}})

EmptyOnly call_empty_only(EmptyOnly u) { return pass_empty_only(u); }
// LLVM: define{{.*}} void @_Z15call_empty_only9EmptyOnly()
// LLVM:   call{{.*}} void @_Z15pass_empty_only9EmptyOnly()

ArrayEmpty call_array_empty(ArrayEmpty u) { return pass_array_empty(u); }
// LLVM: define{{.*}} i32 @_Z16call_array_empty10ArrayEmpty(i32 %{{.+}})
// LLVM:   call{{.*}} i32 @_Z16pass_array_empty10ArrayEmpty(i32 %{{.+}})
