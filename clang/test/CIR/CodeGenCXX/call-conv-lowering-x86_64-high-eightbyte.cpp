// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --input-file=%t.cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --input-file=%t.ogcg.ll %s --check-prefix=OGCG

// The empty array members classify as NoClass, so the only data sits in the
// high eightbyte and the struct passes and returns as one i8 taken FROM BYTE
// EIGHT of its memory. The coercion slot access must honor that direct
// offset on both sides of both boundaries, or the wire byte is padding and
// the real member is never written.

struct Empty {};

struct HiOnly {
  Empty pad[8];
  bool ok;
};

bool take(HiOnly v) { return v.ok; }
// LLVM-LABEL: define{{.*}} i1 @_Z4take6HiOnly(i8
// LLVM: %[[SLOT_AT_8:.*]] = getelementptr i8, ptr %{{.*}}, i64 8
// LLVM: store i8 %{{.*}}, ptr %[[SLOT_AT_8]]
// OGCG-LABEL: define{{.*}} i1 @_Z4take6HiOnly(i8
// OGCG: %[[SLOT_AT_8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// OGCG: store i8 %{{.*}}, ptr %[[SLOT_AT_8]]

HiOnly make(bool b) {
  HiOnly v;
  v.ok = b;
  return v;
}
// LLVM-LABEL: define{{.*}} i8 @_Z4makeb(
// LLVM: %[[RET_AT_8:.*]] = getelementptr i8, ptr %{{.*}}, i64 8
// LLVM: %[[RET:.*]] = load i8, ptr %[[RET_AT_8]]
// LLVM: ret i8 %[[RET]]
// OGCG-LABEL: define{{.*}} i8 @_Z4makeb(
// OGCG: %[[RET_AT_8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// OGCG: %[[RET:.*]] = load i8, ptr %[[RET_AT_8]]
// OGCG: ret i8 %[[RET]]

bool roundtrip(bool b) { return take(make(b)); }
// LLVM-LABEL: define{{.*}} i1 @_Z9roundtripb(
// LLVM: %[[CALLRES:.*]] = call i8 @_Z4makeb(
// LLVM: %[[BACK_AT_8:.*]] = getelementptr i8, ptr %{{.*}}, i64 8
// LLVM: store i8 %[[CALLRES]], ptr %[[BACK_AT_8]]
// LLVM: %[[ARG_AT_8:.*]] = getelementptr i8, ptr %{{.*}}, i64 8
// LLVM: %[[ARG:.*]] = load i8, ptr %[[ARG_AT_8]]
// LLVM: call{{.*}} i1 @_Z4take6HiOnly(i8 %[[ARG]])
// OGCG-LABEL: define{{.*}} i1 @_Z9roundtripb(
// OGCG: %[[CALLRES:.*]] = call i8 @_Z4makeb(
// OGCG: %[[BACK_AT_8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// OGCG: store i8 %[[CALLRES]], ptr %[[BACK_AT_8]]
// OGCG: %[[ARG_AT_8:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// OGCG: %[[ARG:.*]] = load i8, ptr %[[ARG_AT_8]]
// OGCG: call{{.*}} i1 @_Z4take6HiOnly(i8 %[[ARG]])
