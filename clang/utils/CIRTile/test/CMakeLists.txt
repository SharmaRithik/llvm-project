configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CIR_TILE_TEST_DEPENDS
  FileCheck
  count
  cir-tile-translate
  cuda-tile-opt
  cuda-tile-translate
  not
  split-file
)

add_lit_testsuite(check-cir-tile "Running the CIR Tile regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIR_TILE_TEST_DEPENDS}
)
set_target_properties(check-cir-tile PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRTile ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIR_TILE_TEST_DEPENDS}
)
