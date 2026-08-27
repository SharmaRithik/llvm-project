# CIR to CUDA Tile matmul demo

This demo makes the frontend boundary concrete. Clang parses ordinary C++20
and emits CIR. `cir-opt` produces SSA form. `cir-tile-translate` converts the
annotated CIR into CUDA Tile IR bytecode. A normal C++ compiler builds the host
launcher, which gives that bytecode directly to the CUDA Driver API.

NVCC and NVHPC are not used. The CUDA driver owns JIT compilation and device
launch.

## Pipeline

```text
matmul.cpp
  -> Clang CIR
  -> SSA CIR
  -> CUDA Tile IR bytecode
  -> CUDA Driver JIT
  -> matmul on the GPU
```

The kernel multiplies fixed 64 by 64 row-major matrices. It uses 32 by 32
output tiles and four 16-element reduction tiles. The host launches a 2 by 2
tile grid with block dimensions fixed to 1 by 1 by 1 as required by CUDA Tile
kernels.

## Requirements

- The LLVM fork containing CIR
- The CUDA Tile IR repository
- CMake, Ninja, Python, and a C++17 host compiler
- CUDA Toolkit 13.1 or newer
- A compatible NVIDIA driver and Tile-capable GPU for execution

The build-only path does not require a GPU.

## Build LLVM and the bridge

Start in a directory containing `llvm-project` and `cuda-tile`.

```sh
cmake -G Ninja -S llvm-project/llvm -B build-cir-tile \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DCLANG_ENABLE_CIR=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_PROJECTS="cuda_tile;cir_tile_integration" \
  -DLLVM_EXTERNAL_CUDA_TILE_SOURCE_DIR="$PWD/cuda-tile" \
  -DCUDA_TILE_BUILD_IN_LLVM=ON \
  -DCUDA_TILE_ENABLE_TOOLS=ON \
  -DCUDA_TILE_ENABLE_TESTING=OFF \
  -DCUDA_TILE_ENABLE_CAPI=OFF \
  -DLLVM_EXTERNAL_CIR_TILE_INTEGRATION_SOURCE_DIR="$PWD/llvm-project/clang/utils/CIRTile" \
  -DCIR_TILE_ENABLE_TESTING=ON

cmake --build build-cir-tile \
  --target clang cir-opt cir-tile-translate check-cir-tile -j 8
```

## Build and run the demo

Set the build and CUDA Toolkit paths, then run the script.

```sh
export CIR_TILE_BUILD_DIR="$PWD/build-cir-tile"
export CUDA_TOOLKIT_ROOT=/usr/local/cuda

llvm-project/clang/examples/CIRTileMatmul/run.sh
```

Set `CXX` to choose the ordinary host C++ compiler. Set
`CUDA_DRIVER_LIB_DIR` when `libcuda.so` is outside the usual Toolkit
directories.

Successful execution prints `matmul passed`.

Use build-only mode on a system without a compatible GPU.

```sh
llvm-project/clang/examples/CIRTileMatmul/run.sh --build-only
```

## Outputs

The default output directory is `cir-tile-matmul-output` under the current
directory.

| File | Meaning |
|---|---|
| `matmul.cir` | CIR emitted by the Clang C++ frontend |
| `matmul.ssa.cir` | CIR after CFG flattening and SSA promotion |
| `matmul.tilebc` | CUDA Tile IR 13.1 compatibility bytecode |
| `matmul-host` | Driver API host executable built by a normal C++ compiler |

Pass an output directory as the final script argument when a different
location is preferred.
