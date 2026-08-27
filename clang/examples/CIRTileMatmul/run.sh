#!/usr/bin/env bash

set -euo pipefail

build_only=0
if [[ ${1:-} == "--build-only" ]]; then
  build_only=1
  shift
fi

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--build-only] [output-directory]" >&2
  exit 1
fi

: "${CIR_TILE_BUILD_DIR:?set CIR_TILE_BUILD_DIR to the LLVM build directory}"
: "${CUDA_TOOLKIT_ROOT:?set CUDA_TOOLKIT_ROOT to the CUDA toolkit directory}"

demo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd -- "$demo_dir/../../.." && pwd)
output_dir=${1:-"$PWD/cir-tile-matmul-output"}

cir_clang=${CIR_CLANG:-"$CIR_TILE_BUILD_DIR/bin/clang"}
cir_opt=${CIR_OPT:-"$CIR_TILE_BUILD_DIR/bin/cir-opt"}
cir_tile_translate=${CIR_TILE_TRANSLATE:-"$CIR_TILE_BUILD_DIR/bin/cir-tile-translate"}
host_cxx=${CXX:-c++}

require_executable() {
  if [[ ! -x $1 ]]; then
    echo "missing executable $1" >&2
    exit 1
  fi
}

require_executable "$cir_clang"
require_executable "$cir_opt"
require_executable "$cir_tile_translate"

if ! command -v "$host_cxx" >/dev/null; then
  echo "missing host C++ compiler $host_cxx" >&2
  exit 1
fi

cuda_include_dir="$CUDA_TOOLKIT_ROOT/include"
if [[ ! -f $cuda_include_dir/cuda.h ]]; then
  echo "missing CUDA Driver API header $cuda_include_dir/cuda.h" >&2
  exit 1
fi

cuda_driver_lib_dir=${CUDA_DRIVER_LIB_DIR:-}
if [[ -z $cuda_driver_lib_dir ]]; then
  host_arch=$(uname -m)
  candidates=(
    "$CUDA_TOOLKIT_ROOT/lib64/stubs"
    "$CUDA_TOOLKIT_ROOT/lib/stubs"
    "$CUDA_TOOLKIT_ROOT/targets/$host_arch-linux/lib/stubs"
    "$CUDA_TOOLKIT_ROOT/lib64"
    "$CUDA_TOOLKIT_ROOT/lib"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f $candidate/libcuda.so ]]; then
      cuda_driver_lib_dir=$candidate
      break
    fi
  done
fi

if [[ -z $cuda_driver_lib_dir || ! -f $cuda_driver_lib_dir/libcuda.so ]]; then
  echo "set CUDA_DRIVER_LIB_DIR to a directory containing libcuda.so" >&2
  exit 1
fi

mkdir -p "$output_dir"

"$cir_clang" -cc1 -triple nvptx64-nvidia-cuda -std=c++20 -fclangir \
  -fno-clangir-call-conv-lowering -I "$source_root/clang/lib/Headers" \
  -emit-cir "$demo_dir/matmul.cpp" -o "$output_dir/matmul.cir"

"$cir_opt" "$output_dir/matmul.cir" -cir-flatten-cfg -mem2reg \
  -o "$output_dir/matmul.ssa.cir"

"$cir_tile_translate" --emit-bytecode "$output_dir/matmul.ssa.cir" \
  -o "$output_dir/matmul.tilebc"

"$host_cxx" -std=c++17 -Wall -Wextra -Werror \
  -isystem "$cuda_include_dir" "$demo_dir/host.cpp" \
  -L"$cuda_driver_lib_dir" -lcuda -o "$output_dir/matmul-host"

printf '%s\n' \
  "$output_dir/matmul.cir" \
  "$output_dir/matmul.ssa.cir" \
  "$output_dir/matmul.tilebc" \
  "$output_dir/matmul-host"

if [[ $build_only -eq 0 ]]; then
  "$output_dir/matmul-host" "$output_dir/matmul.tilebc"
fi
