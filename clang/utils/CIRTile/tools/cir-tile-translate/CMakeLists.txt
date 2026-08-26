set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(cir-tile-translate
  cir-tile-translate.cpp
)

target_link_libraries(cir-tile-translate
  PRIVATE
  CIRRegisterAllDialects
  CIRTileAnnotation
  CudaTileDialect
  MLIRCIR
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
)

llvm_update_compile_flags(cir-tile-translate)
