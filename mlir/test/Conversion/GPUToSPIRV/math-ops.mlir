// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

// Math ops inside a gpu.func are lowered by convert-gpu-to-spirv directly,
// without a separate convert-math-to-spirv run beforehand.

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // CHECK-LABEL: spirv.func @math_ops
    // CHECK-SAME:    %[[X:arg[0-9]+]]: f32
    // CHECK-SAME:    %[[Y:arg[0-9]+]]: f32
    // CHECK-SAME:    %[[I:arg[0-9]+]]: i32
    gpu.func @math_ops(%x : f32, %y : f32, %i : i32) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // CHECK: %[[S:.+]] = spirv.GL.Sqrt %[[X]] : f32
      %s = math.sqrt %x : f32
      // CHECK: %[[E:.+]] = spirv.GL.Exp %[[Y]] : f32
      %e = math.exp %y : f32
      // CHECK: spirv.GL.Fma %[[S]], %[[E]], %[[X]] : f32
      %f = math.fma %s, %e, %x : f32
      // CHECK: spirv.BitCount %[[I]] : i32
      %p = math.ctpop %i : i32
      gpu.return
    }
  }
}
