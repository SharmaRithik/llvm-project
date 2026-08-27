add_mlir_library(CIRTileSupport
  Annotation.cpp
  Conversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/clang/CIRTile

  LINK_LIBS PUBLIC
  CudaTileDialect
  MLIRCIR
  MLIRIR
)
