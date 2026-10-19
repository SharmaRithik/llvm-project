add_mlir_translation_library(MLIRTargetWGSL
  TranslateToWGSL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/WGSL

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTranslateLib
  MLIRUBDialect
  MLIRVectorDialect
  )

add_subdirectory(Compiler)
