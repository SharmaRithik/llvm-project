#ifndef __CIR_TILE_HPP
#define __CIR_TILE_HPP

#define __tile_global__                                                       \
  __attribute__((annotate("cir_tile.v1.kernel"), convergent))

namespace cuda::tiles {
namespace detail {

template <typename T> inline constexpr bool supported_element = false;
template <> inline constexpr bool supported_element<_Float16> = true;
template <> inline constexpr bool supported_element<float> = true;

template <int Extent>
inline constexpr bool valid_extent =
    Extent > 0 && (Extent & (Extent - 1)) == 0;

template <typename T, int Rows, int Cols> struct tile_storage {
  static_assert(supported_element<T>, "unsupported CIR Tile element type");
  static_assert(valid_extent<Rows> && valid_extent<Cols>,
                "CIR Tile dimensions must be positive powers of two");
  using type = T __attribute__((ext_vector_type(Rows * Cols)));
};

template <int Axis> struct block_axis {
  static_assert(Axis >= 0 && Axis < 3,
                "CIR Tile block axis must be 0, 1, or 2");
  using type = int;
};

template <typename T, int Rows, int Cols, int FullRows, int FullCols>
struct view_tile {
  static_assert(valid_extent<FullRows> && valid_extent<FullCols>,
                "CIR Tile tensor dimensions must be positive powers of two");
  static_assert(FullRows % Rows == 0 && FullCols % Cols == 0,
                "CIR Tile tensor dimensions must be divisible by tile dimensions");
  using type = typename tile_storage<T, Rows, Cols>::type;
};

} // namespace detail

template <typename T, int Rows, int Cols>
using tile = typename detail::tile_storage<T, Rows, Cols>::type;

template <int Axis>
__attribute__((annotate("cir_tile.v1.bid", Axis), convergent))
typename detail::block_axis<Axis>::type bid();

template <int Rows, int Cols>
__attribute__((annotate("cir_tile.v1.zero", Rows, Cols), convergent))
tile<float, Rows, Cols> zeros();

template <int Rows, int Cols, int FullRows, int FullCols>
__attribute__((annotate("cir_tile.v1.load", Rows, Cols, FullRows, FullCols),
               convergent))
typename detail::view_tile<_Float16, Rows, Cols, FullRows, FullCols>::type
load(const _Float16 *ptr, int row, int col);

template <int M, int N, int K>
__attribute__((annotate("cir_tile.v1.mma", M, N, K), convergent))
tile<float, M, N> mma(tile<_Float16, M, K> lhs,
                      tile<_Float16, K, N> rhs, tile<float, M, N> acc);

template <int Rows, int Cols, int FullRows, int FullCols>
__attribute__((annotate("cir_tile.v1.store", Rows, Cols, FullRows, FullCols),
               convergent))
void store(float *ptr,
           typename detail::view_tile<float, Rows, Cols, FullRows,
                                      FullCols>::type value,
           int row, int col);

} // namespace cuda::tiles

#endif
