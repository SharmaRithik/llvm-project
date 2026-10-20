//===- mlir-wgsl-compile.cpp - Compile a matmul to WGSL -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/WGSL/MatmulCompiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
static cl::opt<std::string> input(cl::Positional, cl::Required);
static cl::opt<std::string> output("o", cl::init("-"));
static cl::opt<bool> mma("mma", cl::init(false));
static cl::opt<std::string> dump("dump-prefix", cl::init(""));
static cl::opt<unsigned> threads("max-workgroup-size", cl::init(256));
static cl::opt<unsigned> memory("max-shared-memory", cl::init(32768));
static cl::opt<unsigned> optLevel("opt-level", cl::init(2));
static cl::opt<unsigned> alignment("storage-alignment", cl::init(1));
static cl::opt<bool> nvidia("nvidia-native", cl::init(false));
static cl::opt<unsigned> mmaShapes("mma-shapes", cl::init(15));
static cl::opt<unsigned> batches("batch-count", cl::init(1));
static cl::opt<bool> merged("merged-inputs", cl::init(false));

static bool writeFile(StringRef path, StringRef text) {
  std::error_code ec;
  ToolOutputFile file(path, ec, sys::fs::OF_Text);
  if (ec) {
    errs() << ec.message() << "\n";
    return false;
  }
  file.os() << text;
  file.keep();
  return true;
}
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "MLIR matmul to WGSL\n");
  auto source = MemoryBuffer::getFileOrSTDIN(input);
  if (!source) {
    errs() << source.getError().message() << "\n";
    return 1;
  }
  MlirWgslMatmulOptions options{2, mma, 32, threads, memory,
                               optLevel, alignment, nvidia, mmaShapes, batches, merged};
  MlirWgslMatmulKernel kernel{};
  int status = mlirWgslCompileMatmul((*source)->getBufferStart(), &options, &kernel);
  if (status)
    errs() << (kernel.error ? kernel.error : "compilation failed\n");
  else {
    if (!writeFile(output, kernel.wgsl))
      status = 1;
    errs() << "tile=" << kernel.tile_m << "x" << kernel.tile_n
           << " threads=" << kernel.workgroup_size
           << " shared_bytes=" << kernel.shared_memory_bytes << "\n";
  }
  if (!dump.empty()) {
    if (kernel.scheduled_mlir && !writeFile(dump + ".scheduled.mlir", kernel.scheduled_mlir))
      status = 1;
    if (kernel.lowered_mlir && !writeFile(dump + ".lowered.mlir", kernel.lowered_mlir))
      status = 1;
  }
  mlirWgslFreeMatmulKernel(&kernel);
  return status;
}
