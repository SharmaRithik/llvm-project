#include <cir_tile.hpp>

namespace ct = cuda::tiles;

extern "C" __tile_global__ void matmul(const _Float16 *__restrict a,
                                       const _Float16 *__restrict b,
                                       float *__restrict c) {
  int block_m = ct::bid<0>();
  int block_n = ct::bid<1>();
  auto acc = ct::zeros<32, 32>();

  acc = ct::mma<32, 32, 16>(ct::load<32, 16, 64, 64>(a, block_m, 0),
                            ct::load<16, 32, 64, 64>(b, 0, block_n), acc);
  acc = ct::mma<32, 32, 16>(ct::load<32, 16, 64, 64>(a, block_m, 1),
                            ct::load<16, 32, 64, 64>(b, 1, block_n), acc);
  acc = ct::mma<32, 32, 16>(ct::load<32, 16, 64, 64>(a, block_m, 2),
                            ct::load<16, 32, 64, 64>(b, 2, block_n), acc);
  acc = ct::mma<32, 32, 16>(ct::load<32, 16, 64, 64>(a, block_m, 3),
                            ct::load<16, 32, 64, 64>(b, 3, block_n), acc);

  ct::store<32, 32, 64, 64>(c, acc, block_m, block_n);
}
