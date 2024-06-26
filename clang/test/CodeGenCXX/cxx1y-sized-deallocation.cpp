// Check that delete exprs call the sized deallocation function if
// -fsized-deallocation is passed in C++11 or std >= C++14.
// RUN: %clang_cc1 -std=c++11 -fsized-deallocation %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s

// Check that we don't used sized deallocation with -fno-sized-deallocation or without C++14.
// RUN: %clang_cc1 -std=c++11 %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefix=CHECK-UNSIZED
// RUN: %clang_cc1 -std=c++14 %s -emit-llvm -triple x86_64-linux-gnu -fno-sized-deallocation -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-UNSIZED

// CHECK-UNSIZED-NOT: _ZdlPvm
// CHECK-UNSIZED-NOT: _ZdaPvm

typedef decltype(sizeof(0)) size_t;

typedef int A;
struct B { int n; };
struct C { ~C() {} };
struct D { D(); virtual ~D() {} };
struct E {
  void *operator new(size_t);
  void *operator new[](size_t);
  void operator delete(void *) noexcept;
  void operator delete[](void *) noexcept;
};
struct F {
  void *operator new(size_t);
  void *operator new[](size_t);
  void operator delete(void *, size_t) noexcept;
  void operator delete[](void *, size_t) noexcept;
};

template<typename T> T get();

template<typename T>
void del() {
  ::delete get<T*>();
  ::delete[] get<T*>();
  delete get<T*>();
  delete[] get<T*>();
}

template void del<A>();
template void del<B>();
template void del<C>();
template void del<D>();
template void del<E>();
template void del<F>();

D::D() {}

// CHECK-LABEL: define weak_odr void @_Z3delIiEvv()
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 4)
// CHECK: call void @_ZdaPv(ptr noundef %{{[^ ]*}})
//
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 4)
// CHECK: call void @_ZdaPv(ptr noundef %{{[^ ]*}})

// CHECK-LABEL: declare void @_ZdlPvm(ptr

// CHECK-LABEL: define weak_odr void @_Z3delI1BEvv()
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 4)
// CHECK: call void @_ZdaPv(ptr noundef %{{[^ ]*}})
//
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 4)
// CHECK: call void @_ZdaPv(ptr noundef %{{[^ ]*}})

// CHECK-LABEL: define weak_odr void @_Z3delI1CEvv()
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 1)
// CHECK: mul i64 1, %{{[^ ]*}}
// CHECK: add i64 %{{[^ ]*}}, 8
// CHECK: call void @_ZdaPvm(ptr noundef %{{[^ ]*}}, i64 noundef %{{[^ ]*}})
//
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 1)
// CHECK: mul i64 1, %{{[^ ]*}}
// CHECK: add i64 %{{[^ ]*}}, 8
// CHECK: call void @_ZdaPvm(ptr noundef %{{[^ ]*}}, i64 noundef %{{[^ ]*}})

// CHECK-LABEL: declare void @_ZdaPvm(ptr

// CHECK-LABEL: define weak_odr void @_Z3delI1DEvv()
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 8)
// CHECK: mul i64 8, %{{[^ ]*}}
// CHECK: add i64 %{{[^ ]*}}, 8
// CHECK: call void @_ZdaPvm(ptr noundef %{{[^ ]*}}, i64 noundef %{{[^ ]*}})
//
// CHECK-NOT: Zdl
// CHECK: call void %{{.*}}
// CHECK-NOT: Zdl
// CHECK: mul i64 8, %{{[^ ]*}}
// CHECK: add i64 %{{[^ ]*}}, 8
// CHECK: call void @_ZdaPvm(ptr noundef %{{[^ ]*}}, i64 noundef %{{[^ ]*}})

// CHECK-LABEL: define weak_odr void @_Z3delI1EEvv()
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 1)
// CHECK: call void @_ZdaPv(ptr noundef %{{[^ ]*}})
//
// CHECK: call void @_ZN1EdlEPv(ptr noundef %{{[^ ]*}})
// CHECK: call void @_ZN1EdaEPv(ptr noundef %{{[^ ]*}})

// CHECK-LABEL: define weak_odr void @_Z3delI1FEvv()
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 1)
// CHECK: mul i64 1, %{{[^ ]*}}
// CHECK: add i64 %{{[^ ]*}}, 8
// CHECK: call void @_ZdaPvm(ptr noundef %{{[^ ]*}}, i64 noundef %{{[^ ]*}})
//
// CHECK: call void @_ZN1FdlEPvm(ptr noundef %{{[^ ]*}}, i64 noundef 1)
// CHECK: mul i64 1, %{{[^ ]*}}
// CHECK: add i64 %{{[^ ]*}}, 8
// CHECK: call void @_ZN1FdaEPvm(ptr noundef %{{[^ ]*}}, i64 noundef %{{[^ ]*}})


// CHECK-LABEL: define linkonce_odr void @_ZN1DD0Ev(ptr {{[^,]*}} %this)
// CHECK: call void @_ZdlPvm(ptr noundef %{{[^ ]*}}, i64 noundef 8)
