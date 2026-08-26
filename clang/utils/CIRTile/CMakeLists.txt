cmake_minimum_required(VERSION 3.20.0)

project(CIR_TILE_INTEGRATION)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET MLIRCIR)
  message(FATAL_ERROR "CIR Tile integration requires MLIRCIR")
endif()

if(NOT TARGET CudaTileDialect)
  message(FATAL_ERROR "CIR Tile integration requires CudaTileDialect")
endif()

include_directories(${LLVM_MAIN_SRC_DIR}/../clang/include)
include_directories(${CMAKE_BINARY_DIR}/tools/clang/include)
include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(${CUDA_TILE_SOURCE_DIR}/include)
include_directories(${CUDA_TILE_BINARY_DIR}/include)

add_subdirectory(tools)
