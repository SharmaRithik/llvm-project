// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -I %S/../../../lib/Headers \
// RUN:   -DBAD_TILE %s 2>&1 | FileCheck %s --check-prefix=BAD-TILE
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -I %S/../../../lib/Headers \
// RUN:   -DBAD_AXIS %s 2>&1 | FileCheck %s --check-prefix=BAD-AXIS
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -I %S/../../../lib/Headers \
// RUN:   -DBAD_VIEW %s 2>&1 | FileCheck %s --check-prefix=BAD-VIEW

#include <cir_tile.hpp>

namespace ct = cuda::tiles;

#if defined(BAD_TILE)
// BAD-TILE: error: static assertion failed due to requirement
// BAD-TILE-SAME: CIR Tile dimensions must be positive powers of two
ct::tile<float, 3, 4> bad_tile;
#elif defined(BAD_AXIS)
// BAD-AXIS: error: static assertion failed due to requirement
// BAD-AXIS-SAME: CIR Tile block axis must be 0, 1, or 2
int bad_axis = ct::bid<3>();
#elif defined(BAD_VIEW)
// BAD-VIEW: error: static assertion failed due to requirement
// BAD-VIEW-SAME: CIR Tile tensor dimensions must be divisible by tile dimensions
auto bad_view = ct::load<16, 64, 64, 32>(nullptr, 0, 0);
#endif
