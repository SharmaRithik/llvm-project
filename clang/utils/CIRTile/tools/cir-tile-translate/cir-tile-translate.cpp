//===- cir-tile-translate.cpp - CIR to CUDA Tile tool --------------------===//

#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "clang/CIR/InitAllDialects.h"
#include "clang/CIRTile/Annotation.h"
#include "clang/CIRTile/Conversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input CIR file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"), llvm::cl::init("-"),
                   llvm::cl::value_desc("filename"));

static llvm::cl::opt<bool>
    printAnnotations("print-annotations",
                     llvm::cl::desc("Print decoded CIR Tile annotations"));

static llvm::cl::opt<bool>
    emitBytecode("emit-bytecode", llvm::cl::desc("Emit CUDA Tile IR bytecode"));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR to CUDA Tile translator\n");

  if (printAnnotations && emitBytecode) {
    llvm::errs() << "error: --print-annotations and --emit-bytecode cannot be "
                    "used together\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  cir::registerAllDialects(registry);
  registry.insert<mlir::LLVM::LLVMDialect, mlir::cuda_tile::CudaTileDialect>();

  mlir::MLIRContext context(registry);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code error = input.getError()) {
    llvm::errs() << "error: cannot open '" << inputFilename
                 << "': " << error.message() << '\n';
    return 1;
  }

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*input), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module)
    return 1;

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << '\n';
    return 1;
  }

  if (printAnnotations) {
    if (mlir::failed(
            clang::CIRTile::validateAnnotations(*module, &output->os())))
      return 1;
  } else {
    auto converted = clang::CIRTile::convertToCudaTile(*module);
    if (!converted)
      return 1;
    if (emitBytecode) {
      if (mlir::failed(
              mlir::cuda_tile::writeBytecode(output->os(), *converted)))
        return 1;
    } else {
      converted->print(output->os());
      output->os() << '\n';
    }
  }

  output->keep();
  return 0;
}
