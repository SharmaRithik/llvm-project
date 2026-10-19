// RUN: mlir-translate --mlir-to-wgsl %s -o %t.wgsl
// RUN: FileCheck %s --input-file=%t.wgsl

// Runtime views use offsets/strides in elements of the underlying storage
// buffer. A cast may erase this information from the type, but not the view.
// CHECK: let [[OFFSET:v[0-9]+]]: u32 = arg2[0u].x;
// CHECK: let [[STRIDE:v[0-9]+]]: u32 = arg2[0u].y;
// CHECK: let [[ROW:v[0-9]+]]: u32 = arg2[0u].z;
// CHECK: let {{v[0-9]+}}: f32 = arg0[(([[OFFSET]] + (2u * [[STRIDE]])) + 3u)];
// CHECK: let {{v[0-9]+}}: f32 = arg0[(((([[OFFSET]] + ([[ROW]] * [[STRIDE]])) + 1u) + ([[STRIDE]] * 2u)) + 2u)];
// CHECK: let {{v[0-9]+}}: f32 = arg0[(([[OFFSET]] + [[STRIDE]]) + 2u)];
// CHECK: let [[BASE:b[0-9]+]]: u32 = ([[OFFSET]] + [[STRIDE]]);
// CHECK: vec4<f32>(arg0[[[BASE]] + 0u], arg0[[[BASE]] + 1u], arg0[[[BASE]] + 2u], arg0[[[BASE]] + 3u])
// Reinterpret casts replace metadata relative to the underlying allocation;
// neither of these accesses may retain the first view's offset or stride.
// CHECK: let {{v[0-9]+}}: f32 = arg0[7u];
// CHECK: let {{v[0-9]+}}: f32 = arg0[([[ROW]] + 2u)];
// A dynamic-to-static cast asserts the inner stride. Retain the runtime
// offset while using that unit-stride assertion for a contiguous vector load.
// CHECK: let [[REFINED:b[0-9]+]]: u32 = [[ROW]];
// CHECK: vec4<f32>(arg0[[[REFINED]] + 0u], arg0[[[REFINED]] + 1u], arg0[[[REFINED]] + 2u], arg0[[[REFINED]] + 3u])

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @main(%src: memref<?xf32>, %dst: memref<?xf32>,
                   %params: memref<4xi32> {wgsl.address_space = "uniform"})
        kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %offset32 = memref.load %params[%c0] : memref<4xi32>
      %stride32 = memref.load %params[%c1] : memref<4xi32>
      %row32 = memref.load %params[%c2] : memref<4xi32>
      %offset = arith.index_castui %offset32 : i32 to index
      %stride = arith.index_castui %stride32 : i32 to index
      %row = arith.index_castui %row32 : i32 to index
      %view = memref.reinterpret_cast %src to offset: [%offset],
          sizes: [8, 8], strides: [%stride, 1]
          : memref<?xf32> to memref<8x8xf32, strided<[?, 1], offset: ?>>
      %v0 = memref.load %view[%c2, %c3]
          : memref<8x8xf32, strided<[?, 1], offset: ?>>
      memref.store %v0, %dst[%c0] : memref<?xf32>

      %cast = memref.cast %view
          : memref<8x8xf32, strided<[?, 1], offset: ?>>
          to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %slice = memref.subview %cast[%row, 1][2, 4][2, 1]
          : memref<?x?xf32, strided<[?, ?], offset: ?>>
          to memref<2x4xf32, strided<[?, ?], offset: ?>>
      %v1 = memref.load %slice[%c1, %c2]
          : memref<2x4xf32, strided<[?, ?], offset: ?>>
      memref.store %v1, %dst[%c1] : memref<?xf32>

      %reduced = memref.subview %view[1, 0][1, 4][1, 1]
          : memref<8x8xf32, strided<[?, 1], offset: ?>>
          to memref<4xf32, strided<[1], offset: ?>>
      %v2 = memref.load %reduced[%c2]
          : memref<4xf32, strided<[1], offset: ?>>
      memref.store %v2, %dst[%c2] : memref<?xf32>
      %v3 = vector.load %view[%c1, %c0]
          : memref<8x8xf32, strided<[?, 1], offset: ?>>, vector<4xf32>
      vector.store %v3, %dst[%c8] : memref<?xf32>, vector<4xf32>

      %reset = memref.reinterpret_cast %view to offset: [5],
          sizes: [4], strides: [1]
          : memref<8x8xf32, strided<[?, 1], offset: ?>>
          to memref<4xf32, strided<[1], offset: 5>>
      %v4 = memref.load %reset[%c2] : memref<4xf32, strided<[1], offset: 5>>
      memref.store %v4, %dst[%c3] : memref<?xf32>
      %dynamic_reset = memref.reinterpret_cast %view to offset: [%row],
          sizes: [4], strides: [1]
          : memref<8x8xf32, strided<[?, 1], offset: ?>>
          to memref<4xf32, strided<[1], offset: ?>>
      %v5 = memref.load %dynamic_reset[%c2]
          : memref<4xf32, strided<[1], offset: ?>>
      memref.store %v5, %dst[%c4] : memref<?xf32>
      %unknown_stride = memref.reinterpret_cast %src to offset: [%row],
          sizes: [4], strides: [%stride]
          : memref<?xf32> to memref<4xf32, strided<[?], offset: ?>>
      %assert_unit_stride = memref.cast %unknown_stride
          : memref<4xf32, strided<[?], offset: ?>>
          to memref<4xf32, strided<[1], offset: ?>>
      %v6 = vector.load %assert_unit_stride[%c0]
          : memref<4xf32, strided<[1], offset: ?>>, vector<4xf32>
      vector.store %v6, %dst[%c8] : memref<?xf32>, vector<4xf32>
      gpu.return
    }
  }
}
