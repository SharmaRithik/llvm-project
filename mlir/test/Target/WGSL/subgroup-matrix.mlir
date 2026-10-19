// RUN: mlir-translate --mlir-to-wgsl %s | FileCheck %s

// Subgroup MMA operations map onto the WGSL subgroup matrix builtins. The
// matrix types take <element, columns, rows>.
// CHECK:      diagnostic(off, chromium.subgroup_matrix_uniformity);
// CHECK-NEXT: enable f16;
// CHECK-NEXT: enable subgroups;
// CHECK-NEXT: enable chromium_experimental_subgroup_matrix;
// CHECK:      var<workgroup> tiles: array<f16, 1024>;
// CHECK:      @compute @workgroup_size(32, 1, 1)
// CHECK-NEXT: fn main(@builtin(subgroup_id) sg_id: u32) {
// CHECK-NEXT:   let [[OFF:v[0-9]+]]: u32 = sg_id * 16u;
// CHECK-NEXT:   let [[A:v[0-9]+]]: subgroup_matrix_left<f16, 16, 16> = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&tiles, [[OFF]] * 32u, false, 32u);
// CHECK-NEXT:   let [[B:v[0-9]+]]: subgroup_matrix_right<f16, 8, 16> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 16>>(&tiles, 512u, true, 32u);
// CHECK-NEXT:   let [[D:v[0-9]+]]: subgroup_matrix_result<f32, 8, 16> = subgroupMatrixMultiplyAccumulate([[A]], [[B]], subgroup_matrix_result<f32, 8, 16>());
// CHECK-NEXT:   subgroupMatrixStore(&c, 0u, [[D]], true, 16u);
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @main(%c: memref<?xf32> {wgsl.name = "c"})
        workgroup(%sh: memref<32x32xf16, #gpu.address_space<workgroup>> {wgsl.name = "tiles"})
        kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %sg = gpu.subgroup_id : index
      %row = arith.muli %sg, %c16 : index
      %f0 = arith.constant 0.0 : f32
      %acc = gpu.subgroup_mma_constant_matrix %f0 : !gpu.mma_matrix<16x8xf32, "COp">
      %a = gpu.subgroup_mma_load_matrix %sh[%row, %c0] leadDimension 32 : memref<32x32xf16, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf16, "AOp">
      %b = gpu.subgroup_mma_load_matrix %sh[%c16, %c0] leadDimension 32 transpose : memref<32x32xf16, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf16, "BOp">
      %d = gpu.subgroup_mma_compute %a, %b, %acc : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x8xf16, "BOp"> -> !gpu.mma_matrix<16x8xf32, "COp">
      gpu.subgroup_mma_store_matrix %d, %c[%c0] leadDimension 16 transpose : !gpu.mma_matrix<16x8xf32, "COp">, memref<?xf32>
      gpu.return
    }
  }
}
