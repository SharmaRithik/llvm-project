// RUN: mlir-translate --mlir-to-wgsl %s | FileCheck %s

// Loop-carried values become variables. Swapped values are snapshotted
// before the yield assigns them.
// CHECK-LABEL: fn loops(@builtin(local_invocation_id) local_id: vec3<u32>)
// CHECK:         var [[A:a[0-9]+]]: f32 = 0.0f;
// CHECK-NEXT:    var [[B:a[0-9]+]]: f32 = 1.0f;
// CHECK-NEXT:    for (var [[I:iv[0-9]+]]: u32 = 0u; [[I]] < 8u; [[I]] += 1u) {
// CHECK-NEXT:      let [[S:v[0-9]+]]: f32 = [[A]] + [[B]];
// CHECK-NEXT:      let [[T:t[0-9]+]] = [[B]];
// CHECK-NEXT:      [[A]] = [[T]];
// CHECK-NEXT:      [[B]] = [[S]];
// CHECK-NEXT:    }

// Integers are u32; signed operations reinterpret their operands as i32.
// CHECK-NEXT:    let [[Q:v[0-9]+]]: u32 = bitcast<u32>(bitcast<i32>(local_id.x) / bitcast<i32>(4294967293u));
// CHECK-NEXT:    let [[LT:v[0-9]+]]: bool = bitcast<i32>([[Q]]) < bitcast<i32>(4294967293u);

// scf.if results become variables assigned in each branch.
// CHECK-NEXT:    var [[R:r[0-9]+]]: f32;
// CHECK-NEXT:    if ([[LT]]) {
// CHECK-NEXT:      [[R]] = [[A]];
// CHECK-NEXT:    } else {
// CHECK-NEXT:      [[R]] = [[B]];
// CHECK-NEXT:    }

// N-D workgroup memrefs are flattened in row-major order.
// CHECK-NEXT:    let [[H:v[0-9]+]]: f16 = f16([[R]]);
// CHECK-NEXT:    shmem0[local_id.x + 8u] = [[H]];
// CHECK-NEXT:    workgroupBarrier();
// CHECK-NEXT:    let [[BASE:b[0-9]+]]: u32 = 8u;
// CHECK-NEXT:    let [[V:v[0-9]+]]: vec4<f16> = vec4<f16>(shmem0[[[BASE]] + 0u], shmem0[[[BASE]] + 1u], shmem0[[[BASE]] + 2u], shmem0[[[BASE]] + 3u]);
// CHECK-NEXT:    let [[VF:v[0-9]+]]: vec4<f32> = vec4<f32>([[V]]);
// CHECK-NEXT:    let [[SUM:v[0-9]+]]: f32 = ((([[VF]].x + [[VF]].y) + [[VF]].z) + [[VF]].w);
// CHECK-NEXT:    buf[local_id.x] = [[SUM]];
// CHECK-NEXT:    return;
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @loops(%buf: memref<?xf32> {wgsl.name = "buf"})
        workgroup(%sh: memref<4x8xf16, #gpu.address_space<workgroup>>)
        kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %f0 = arith.constant 0.0 : f32
      %f1 = arith.constant 1.0 : f32
      %tid = gpu.thread_id x
      %r:2 = scf.for unsigned %i = %c0 to %c8 step %c1 iter_args(%a = %f0, %b = %f1) -> (f32, f32) {
        %s = arith.addf %a, %b : f32
        scf.yield %b, %s : f32, f32
      }
      %neg = arith.constant -3 : i32
      %ti = arith.index_cast %tid : index to i32
      %q = arith.divsi %ti, %neg : i32
      %lt = arith.cmpi slt, %q, %neg : i32
      %sel = scf.if %lt -> f32 {
        scf.yield %r#0 : f32
      } else {
        scf.yield %r#1 : f32
      }
      %h = arith.truncf %sel : f32 to f16
      memref.store %h, %sh[%c1, %tid] : memref<4x8xf16, #gpu.address_space<workgroup>>
      gpu.barrier memfence [#gpu.address_space<workgroup>]
      %v = vector.load %sh[%c1, %c0] : memref<4x8xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      %vf = arith.extf %v : vector<4xf16> to vector<4xf32>
      %sum = vector.reduction <add>, %vf : vector<4xf32> into f32
      memref.store %sum, %buf[%tid] : memref<?xf32>
      gpu.return
    }
  }
}
