add_mlir_tool(mlir-wgsl-compile mlir-wgsl-compile.cpp)
target_link_libraries(mlir-wgsl-compile PRIVATE MLIRWGSLMatmulCompiler LLVMSupport)
