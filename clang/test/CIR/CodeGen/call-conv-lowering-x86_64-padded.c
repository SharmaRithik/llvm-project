// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

// Padded and packed records classify from each field's layout offset, so the
// signatures below match classic CodeGen.  The only difference is the
// llvm.noalias CIR adds on byval, matching -fpass-by-value-is-noalias.

struct Packed {
  char c;
  int i;
} __attribute__((packed));

struct OverAligned {
  int i;
} __attribute__((aligned(16)));

struct AlignedByte {
  unsigned char c;
} __attribute__((aligned(4)));

struct Inner {
  short a, b, c;
};

struct TailPadded {
  struct Inner s;
  char t;
};

struct BigPadded {
  int i;
} __attribute__((aligned(32)));

// The packed int sits at offset 1, which the eightbyte rules send to memory.
int use_packed(struct Packed p) { return p.i; }
// LLVM-CIR: define{{.*}} i32 @use_packed(ptr noalias noundef byval(%struct.Packed) align 8 %{{.*}})
// LLVM-OGCG: define{{.*}} i32 @use_packed(ptr noundef byval(%struct.Packed) align 8 %{{.*}})

// Tail padding from the alignment does not widen the coercion.
int use_over_aligned(struct OverAligned o) { return o.i; }
// LLVM: define{{.*}} i32 @use_over_aligned(i32 %{{.*}})

unsigned use_aligned_byte(struct AlignedByte b) { return b.c; }
// LLVM: define{{.*}} i32 @use_aligned_byte(i8 %{{.*}})

// A padded record nested inside another record coerces from the data bytes.
int use_tail_padded(struct TailPadded t) { return t.s.b; }
// LLVM: define{{.*}} i32 @use_tail_padded(i64 %{{.*}})

// Over two eightbytes of declared size is memory whatever the content, and
// the byval slot carries the declared alignment.
int use_big_padded(struct BigPadded b) { return b.i; }
// LLVM-CIR: define{{.*}} i32 @use_big_padded(ptr noalias noundef byval(%struct.BigPadded) align 32 %{{.*}})
// LLVM-OGCG: define{{.*}} i32 @use_big_padded(ptr noundef byval(%struct.BigPadded) align 32 %{{.*}})

struct OverAligned ret_over_aligned(int v) {
  struct OverAligned o = {v};
  return o;
}
// LLVM: define{{.*}} i32 @ret_over_aligned(i32 noundef %{{.*}})

// A memory class return uses sret with the record's own alignment.
struct Packed ret_packed(int v) {
  struct Packed p = {1, v};
  return p;
}
// LLVM: define{{.*}} void @ret_packed(ptr dead_on_unwind noalias writable sret(%struct.Packed) align 1 %{{.*}}, i32 noundef %{{.*}})

int (*indirect_target)(struct Packed) = use_packed;

int call_all(struct Packed p, struct OverAligned o, struct AlignedByte b,
             struct TailPadded t, struct BigPadded g) {
  int r = use_packed(p) + use_over_aligned(o) + use_aligned_byte(b) +
          use_tail_padded(t) + use_big_padded(g);
  r += ret_over_aligned(r).i + ret_packed(r).i;
  r += indirect_target(p);
  return r;
}
// LLVM-CIR: define{{.*}} i32 @call_all(ptr noalias noundef byval(%struct.Packed) align 8 %{{.*}}, i32 %{{.*}}, i8 %{{.*}}, i64 %{{.*}}, ptr noalias noundef byval(%struct.BigPadded) align 32 %{{.*}})
// LLVM-OGCG: define{{.*}} i32 @call_all(ptr noundef byval(%struct.Packed) align 8 %{{.*}}, i32 %{{.*}}, i8 %{{.*}}, i64 %{{.*}}, ptr noundef byval(%struct.BigPadded) align 32 %{{.*}})
// LLVM-CIR: call i32 @use_packed(ptr noalias noundef byval(%struct.Packed) align 8 %{{.*}})
// LLVM-OGCG: call i32 @use_packed(ptr noundef byval(%struct.Packed) align 8 %{{.*}})
// LLVM: call i32 @use_over_aligned(i32 %{{.*}})
// LLVM: call i32 @use_aligned_byte(i8 %{{.*}})
// LLVM: call i32 @use_tail_padded(i64 %{{.*}})
// LLVM-CIR: call i32 @use_big_padded(ptr noalias noundef byval(%struct.BigPadded) align 32 %{{.*}})
// LLVM-OGCG: call i32 @use_big_padded(ptr noundef byval(%struct.BigPadded) align 32 %{{.*}})
// LLVM: call i32 @ret_over_aligned(i32 noundef %{{.*}})
// LLVM: call void @ret_packed(ptr dead_on_unwind writable sret(%struct.Packed) align 1 %{{.*}}, i32 noundef %{{.*}})
// LLVM-CIR: call i32 %{{.*}}(ptr noalias noundef byval(%struct.Packed) align 8 %{{.*}})
// LLVM-OGCG: call i32 %{{.*}}(ptr noundef byval(%struct.Packed) align 8 %{{.*}})

// A padded record whose data is all floating point classifies SSE, so the
// tail padding must not push it to memory or widen the coercion.
struct FloatPair {
  float a, b;
} __attribute__((aligned(16)));

float use_float_pair(struct FloatPair p) { return p.a + p.b; }
// LLVM: define{{.*}} float @use_float_pair(<2 x float> %{{.*}})

// Bitfield access units at their natural offsets classify without the
// declared widths, so only padded or packed records reject them.
struct NaturalBits {
  int tag : 8;
  int rest : 24;
};

int use_natural_bits(struct NaturalBits b) { return b.tag; }
// LLVM: define{{.*}} i32 @use_natural_bits(i32 %{{.*}})
