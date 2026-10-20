//===- MatmulCompiler.h - WGSL matmul compiler C API --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_WGSL_MATMULCOMPILER_H
#define MLIR_TARGET_WGSL_MATMULCOMPILER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A static f16 matmul with f16 or f32 accumulation. Output is column-major.
// N <= 4 with f32 accumulation selects the subgroup GEMV schedule.
typedef struct {
  uint32_t abi_version;
  uint32_t use_mma;
  uint32_t subgroup_size;
  uint32_t max_workgroup_size;
  uint32_t max_shared_memory_bytes;
  uint32_t optimization_level; // 0 to 2
  uint32_t storage_alignment; // In elements.
  uint32_t nvidia_native;
  uint32_t mma_shapes; // 16x16 f16/f32 = 1/2, 16x8 f16/f32 = 4/8.
  uint32_t batch_count;
  uint32_t merged_inputs; // Both inputs share one binding.
} MlirWgslMatmulOptions;

typedef struct {
  char *wgsl;
  char *scheduled_mlir;
  char *lowered_mlir;
  char *error;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t workgroup_size;
  uint32_t shared_memory_bytes;
} MlirWgslMatmulKernel;

// Returns 0 on success. Free the result in either case.
int mlirWgslCompileMatmul(const char *source, const MlirWgslMatmulOptions *options,
                         MlirWgslMatmulKernel *result);
void mlirWgslFreeMatmulKernel(MlirWgslMatmulKernel *result);

#ifdef __cplusplus
}
#endif

#endif // MLIR_TARGET_WGSL_MATMULCOMPILER_H
