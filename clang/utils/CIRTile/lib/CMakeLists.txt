add_mlir_library(CIRTileAnnotation
  Annotation.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/clang/CIRTile

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
)
