# WGSL code generation

The prototype has two entry points:

- `mlir-translate --mlir-to-wgsl` translates supported structured GPU kernels to WGSL.
- `mlir-wgsl-compile` compiles a supported high-level Linalg matmul to such a kernel and then emits WGSL. The same compilation is available through the native `MLIRWGSLMatmulCompiler` library.

The WGSL emitter maps GPU IDs, barriers, subgroup matrix operations, structured
control flow, supported arithmetic and memory accesses directly to WGSL. It
preserves structured control flow instead of reconstructing it from an LLVM
control-flow graph. Unsupported operations and types produce diagnostics.
Subgroup MMA currently uses Dawn's experimental subgroup matrix extension.

## Matmul pipeline

The input is a single function taking static rank-2 tensors and returning a
zero-initialized transpose-B matmul with FP16 or FP32 accumulation. The LHS is FP16;
an FP32 RHS has an explicit pointwise truncation to FP16 for the tiled matmul
paths. The subgroup GEMV path described below preserves FP32 activations.
`linalg.matmul` carries the transpose-B indexing map in this revision.
See `test/Target/WGSL/Matmul/f16.mlir` and `f32-activation.mlir` for complete
inputs. FP16 accumulation is expressed by an FP16 matmul result followed by an
explicit pointwise extension to FP32; the schedule never changes the input's
accumulation precision. No GPU operations or scheduling loops appear in the input.

| Stage | Implementation |
| --- | --- |
| Parse, verify and check supported graph | Upstream parser/verifier and generic-contraction specialization; strict target checks for the complete graph and precision |
| Choose workgroup/K tiles and physical shared layout | Target-specific scheduling in `MatmulCompiler.cpp` |
| Tile the complete result graph into workgroups | Upstream `scf::tileConsumerAndFuseProducersUsingSCF` and Linalg `TilingInterface`, producing the live outer `scf.forall`, input slices and output coordinates |
| Pad workgroup edges, tile K and fuse the RHS conversion | Upstream Linalg padding, SCF tile-and-fuse, tensor slice/padding folding and `scf::tileAndFuseProducerOfSlice`; accumulator initialization remains outside K |
| Place padded inputs in shared memory | Upstream `linalg::bufferizeToAllocation` produces the fills and copies; the target supplies padded or K-major workgroup views and preserves the pointwise update as a tensor SSA chain |
| Convert logical tensors to buffers | Upstream empty-tensor elimination and `runOneShotModuleBufferize`; the caller supplies the output destination |
| Bind logical views and map workgroups | Target runtime offsets, strides and batch decoding; upstream `mapForallToBlocksImpl` consumes the actual workgroup loop, with `affine::delinearizeIndex` recovering M/N coordinates |
| Partition the contraction and copies among workers | Upstream SCF/Linalg tiling, `mapOneForallToThreadsImpl`, Linalg vectorization or scalar loop lowering |
| Keep the accumulator in registers | Upstream vectorization derives the contraction; target adaptation carries the resulting vector through the existing K loop and emits coalesced output stores |
| Construct the kernel function | `gpu.launch` followed by the upstream GPU kernel-outlining pass; the target preserves binding and workgroup attributes |
| Lower runtime views and packed bindings | Upstream MemRef metadata expansion and alias folding; target scalar-to-vec4 storage adaptation when alignment permits |
| Pipeline next-tile loads with current-tile computation | Upstream LICM and `scf::pipelineForLoop`; target policy supplies the operation/stage ordering |
| MMA lowering | Shared pattern population used by `transform.apply_patterns.gpu.unroll_vectors_subgroup_mma`; VectorToGPU with fixes for mixed loop-carried values and unsigned comparisons |
| Register lowering | Upstream contraction/transpose/transfer lowering; added `vector-unroll-loop-carried` pass |
| Final simplification | Upstream affine lowering, unsigned-equivalence proof, integer-range optimization for MMA, canonicalization, CSE and LICM |
| Emit source | Added WGSL translation |

`StructuredSchedule.cpp` transforms the original tensor function in place.
`SharedMemorySchedule.cpp` distributes its actual bufferized fills, copies and
pointwise conversion. `MatmulCompiler.cpp`, `RuntimeBindings.cpp` and
`KernelOutlining.cpp` connect that computation to the GPU kernel and runtime ABI.
The tiled loops, views and casts remain live through code generation; the compiler
does not discard a planning graph and substitute a separate matmul implementation.

The compiler owns schedule selection and invokes upstream transformations to
implement it. Its optimization levels expose a reference schedule and two
cumulative groups of options:

| Level | Behavior |
| --- | --- |
| 0 | Reference: 16x16 MMA tile, or 32x32 register tile with 4x4 outputs per thread; K tile 16 |
| 1 | Shape-dependent MMA tiles and 16x8 fragments when supported; vec4 storage when alignment permits; K-major shared layout for register tiling |
| 2 | Native NVIDIA policy: shared-memory padding, FP16 MMA output-buffer reuse, eligible MMA/register prefetch with the final iteration peeled, and bounded projection/register tile selection |

The scheduling transformations are invoked through C++ APIs; the compiler does
not currently execute a Transform dialect schedule script. Upstream SCF now
constructs the prefetch prologue, carried values, steady loop and epilogue. The
counted K loop stays rolled; fragment decomposition and cooperative-copy
unrolling are separate operations.

Workgroup tiling starts from the function's returned Linalg operation, including
the output conversion when present, and fuses its contraction and zero fill.
The RHS truncation is deferred until K tiling so it operates on the required
input slice. The counted K loop retains the full original reduction bound,
including when K is smaller than a tile or not a tile multiple. Tensor padding
and upstream ValueBounds models describe the partial tiles; orthogonal row/K
padding is combined before storage placement.

The padded input values become workgroup views before One-Shot Bufferize. A
single restricted tensor alias represents each disjoint shared view. A computed
input updates that tensor through `tensor.insert_slice`, and upstream empty-tensor
elimination directs the original pointwise operation into the shared destination.
It runs before full-tile slice folding can remove that destination anchor, and
again after remaining padding is decomposed. There is no full-matrix temporary
for RHS conversion or output accumulation. The tile accumulator is subsequently
lowered to vector SSA. Input bindings are not declared disjoint: merged inputs
may refer to the same storage allocation.

Equivalent `linalg.generic` contractions are normalized through upstream
`specializeGenericOp`, which uses `inferContractionDims`, including permuted
iteration order. The target first restricts the scalar body to the supported
casts, multiply and add. Contraction classification alone does not prove
arbitrary unary arithmetic, initialization or extra graph computation can be
discarded. Existing whole-graph and precision checks remain in force.

The upstream GPU mapper normally derives warp IDs from linear thread IDs. WGSL
does not guarantee that relationship. An optional ID-builder argument lets this
target provide `gpu.subgroup_id` while retaining the mapper's resource checks,
forall distribution and index substitution. Thread mapping uses the upstream
thread-ID builder. Workgroup mapping uses a separate ID builder for llama.cpp's
flattened grid and batch layout, while preserving the M/N coordinates produced
by upstream tiling. It bounds the matrix-local linear ID with an unsigned
remainder and passes both M/N grid extents to upstream
`affine::delinearizeIndex`. This exposes unit grid dimensions as zero coordinates
and proves the remaining coordinates nonnegative, avoiding signed floor/mod
correction branches. Batch offsets continue to use the original flattened ID.
GPU ValueBounds external models are registered alongside the other dialect
models used by affine and tiling analyses. Integer-range optimization is enabled
on the MMA path.

Target code still selects schedules, physical shared-memory layouts, buffer
reuse, barrier placement and prefetch stages. It also handles the llama.cpp
binding/batch ABI, vec4 storage packing, the SSA accumulator adapter, coalesced
output stores and WGSL emission. Shared-memory placement uses upstream padding
bufferization rather than `linalg::promoteSubViews`; cooperative copies use the
SCF tiling and GPU mapping C++ APIs rather than the tensor-only
`transform.structured.gpu.map_copy_to_threads` operation. The compiler does not
use `flatten-memref`, `gpu-decompose-memrefs` or a Transform-dialect schedule
selection script. GEMV scheduling is unchanged by this pipeline.

Cooperative copies use upstream tiling and vectorization. Statically bounded
copy loops may be unrolled; dynamic boundary-copy loops remain rolled. Partial
input tiles are zero-filled and valid elements are copied with bounded or masked
accesses. Complete tiles allow the corresponding padding and masks to fold away.
The outer guard for excess workgroups from grid rounding remains.

The upstream Vector masked-load/store emulation has an added uniform-mask path:
a scalar broadcast, a constant splat, or a scalar select between uniform Boolean
vectors becomes one `scf.if` around a vector load or store. Masked loads preserve
their pass-through value when the condition is false. This keeps row-boundary
checks from decomposing a contiguous vector access into independent scalar
accesses. Lane-varying masks retain the existing emulation, and scalable-mask
restrictions are unchanged. The rewrite is a general Vector transformation;
it does not depend on WGSL, matmul dimensions or the storage ABI.

Prefetch applies when the lowered copies expose separate loads and stores;
dynamic copy loops or mixed load/store roots disable it. Invariant views are
hoisted before software pipelining so stage boundaries carry values rather than
memref descriptors. Prefetch holds the next tile in SSA values, loads it before
the current matrix computation, and skips unused loads after the last tile.
Output stores are coalesced and use vec4 when alignment permits. A workgroup-memory
barrier separates the last input read from reuse of input storage for the output
tile. Schedule barriers fence only workgroup memory: storage inputs are read-only,
and result storage writes follow the final barrier. The general WGSL emitter
continues to preserve both storage and workgroup fences for an unqualified
`gpu.barrier`.

The level-2 target policy is a bounded set of schedules measured on native
NVIDIA WebGPU, not an auto-tuner or a claim of optimality on every GPU. Resource
limits can select a smaller schedule. Small aligned register projections with
M/K <= 512 and 128 < N <= 512 use the 32x32 tile to expose more workgroups.
Register tiling requires FP32 accumulation.

For FP32 MMA, output staging can make the initial tile exceed shared-memory
limits. The compiler halves the output tile, keeping a 2x2 subgroup arrangement,
until it fits or reaches 32x32. This retains the K tile, padding and prefetch
where possible. On a 48-KiB target, a padded 128x128 FP32 tile needs 88 KiB,
while the 64x64 tile needs 28 KiB. The existing minimal fallback handles tighter
limits. These resource choices preserve the accumulation precision.

The compiler rejects unsupported graphs, nonzero accumulator initialization,
dynamic/incompatible shapes and requests for which no supported schedule fits
the supplied resource limits. The caller supplies actual matrix capabilities,
subgroup size, workgroup/shared-memory limits and storage alignment. MMA requires
32-lane subgroups and 16x16x16 FP16-input support for the input's accumulator type;
16x8x16 support enables the narrow-N schedule.

## Subgroup GEMV

For N <= 4, FP32 accumulation, and no explicit RHS truncation, the compiler
selects a subgroup reduction schedule. It accepts the same high-level
transpose-B Linalg matmul, with F16 weights and either F16 or F32 activations.
F32 activations retain their precision. This path uses ordinary subgroup
arithmetic rather than subgroup matrix MMA instructions.

Each subgroup owns an output row. A workgroup has up to four subgroups, bounded
by the target workgroup limit. The scheduler distributes K across subgroup
lanes, emits vec4 loads and FP32 vector FMA when alignment permits, and reduces
the lane-local vector sum with `gpu.subgroup_reduce add`. Scalar loads handle
unaligned or partial-width inputs. The WGSL emitter maps the reduction to
`subgroupAdd`; the kernel needs neither shared memory nor workgroup barriers.
For N=2..4 the weight load is reused across the activation vectors. Partial rows
and K tails are guarded, and a chunk-counted K loop avoids overflowing its final
32-bit induction increment.

This schedule is built with upstream Arith, Vector, SCF, MemRef and GPU ops;
canonicalization, CSE and LICM simplify it before the existing WGSL emitter runs.
Unlike the tiled MMA path, it does not use shared-memory Linalg tile vectorization
or VectorToGPU MMA lowering. No WGSL template is embedded in the scheduler.

The caller must provide a fixed power-of-two subgroup size between 4 and 128.
The adapter checks that the device's minimum and maximum subgroup sizes agree.
Batching, broadcasts, row strides, scalar output addressing and merged F16 inputs
use the same uniform ABI as the tiled matmul paths. Optimization level zero uses
scalar storage; levels one and two enable aligned vec4 storage. The schedule is
currently measured on native NVIDIA WebGPU, not tuned for every GPU.

## Replay an input

```sh
build/bin/mlir-wgsl-compile mlir/test/Target/WGSL/Matmul/f32-activation.mlir \
  --mma --opt-level=2 --storage-alignment=4 --nvidia-native \
  --dump-prefix=/tmp/matmul -o /tmp/matmul.wgsl
```

Omit `--mma` for register tiling. The dump contains scheduled and lowered MLIR;
`--max-workgroup-size` and `--max-shared-memory` specify resource limits.
Only assert `--storage-alignment=4` when offsets and relevant strides are aligned
to four elements. `--mma-shapes` supplies a capability bitmask: 16x16 FP16/FP32
results are bits 0/1 and 16x8 FP16/FP32 results are bits 2/3, all with K=16.
`--batch-count` describes independent applications of the rank-2 computation;
`--merged-inputs` places both FP16 inputs in one storage binding.
The version-2 C interface in `mlir/Target/WGSL/MatmulCompiler.h` returns the same artifacts
and launch metadata, and has an explicit release function for either success
or failure. It creates a separate MLIR context for each compilation.

## Buffer ABI

The emitted entry point is `main`. Group 0 has storage bindings 0/1/2 for the
LHS, RHS and result, followed by a 16-word i32 uniform at binding 3. Storage
bindings use read-write declarations to match llama.cpp's shared allocation
usage. The shader only reads its inputs and only writes its result.

| Uniform word | Meaning |
| --- | --- |
| 0 / 1 / 2 | LHS / RHS / output element offsets |
| 3 / 4 / 5 | M / N / K in the ggml ABI; this compiler specializes these dimensions |
| 6 / 7 | LHS / RHS row strides in elements |
| 8 / 9 | LHS / RHS batch-dimension-2 strides in elements |
| 10 / 11 | LHS / RHS batch-dimension-3 strides in elements |
| 12 / 13 | LHS extents in dimensions 2 / 3 |
| 14 / 15 | RHS-to-LHS broadcast factors in dimensions 2 / 3 |

The output is column-major: logical `(m,n)` is stored at `offset + n*M + m`.
The caller dispatches `ceil(M/tile_m)*ceil(N/tile_n)*batch_count` workgroups, optionally
flattened over a 2D dispatch grid. Extra workgroups introduced by rounding the
grid are guarded. The returned workgroup size and shared-memory requirement
must agree with the device limits.

With merged FP16 inputs, group 0 instead has input/output/uniform bindings 0/1/2.
The two input offsets still address their respective ranges. Inner dimensions
must be contiguous; row and batch strides may vary. The output is contiguous
across batches. The uniform layout, bounds and declared alignment are caller
contracts, checked by the llama.cpp adapter.

This path does not implement arbitrary Linalg graphs, dynamic shapes, quantized
matmuls or browser hosting of the compiler. Batching applies the rank-2 program
independently using the ABI above; it is not general batched-Linalg lowering.
