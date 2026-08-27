set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(cir-tile-translate
  cir-tile-translate.cpp
)

target_link_libraries(cir-tile-translate
  PRIVATE
  CIRRegisterAllDialects
  CIRTileSupport
  CudaTileDialect
  MLIRCIR
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
)

llvm_update_compile_flags(cir-tile-translate)
