//===- cir-tile-translate.cpp - CIR to CUDA Tile tool --------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR to CUDA Tile translator\n");

  mlir::DialectRegistry registry;
  registry.insert<cir::CIRDialect, mlir::cuda_tile::CudaTileDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  return 0;
}
