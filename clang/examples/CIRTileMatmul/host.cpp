#include <cuda.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr unsigned matrixSize = 64;
constexpr unsigned tileSize = 32;
constexpr std::uint16_t halfOne = 0x3c00;
constexpr std::uint16_t halfTwo = 0x4000;
constexpr std::uint16_t halfOneHalf = 0x3800;

void checkCuda(CUresult result, const char *call) {
  if (result == CUDA_SUCCESS)
    return;

  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::cerr << call << " failed with " << (name ? name : "CUDA error");
  if (message)
    std::cerr << ": " << message;
  std::cerr << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call) checkCuda((call), #call)

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <matmul.tilebc>\n";
    return EXIT_FAILURE;
  }

  const std::size_t elementCount = matrixSize * matrixSize;
  std::vector<std::uint16_t> a(elementCount);
  std::vector<std::uint16_t> b(elementCount);
  std::vector<float> c(elementCount, 0.0f);

  for (unsigned row = 0; row < matrixSize; ++row) {
    for (unsigned column = 0; column < matrixSize; ++column) {
      a[row * matrixSize + column] = row % 2 == 0 ? halfOne : halfOneHalf;
      b[row * matrixSize + column] = column % 2 == 0 ? halfOne : halfTwo;
    }
  }

  CUDA_CHECK(cuInit(0));

  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));

  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, nullptr, 0, device));

  CUmodule module;
  CUDA_CHECK(cuModuleLoad(&module, argv[1]));

  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, module, "matmul"));

  CUdeviceptr deviceA;
  CUdeviceptr deviceB;
  CUdeviceptr deviceC;
  CUDA_CHECK(cuMemAlloc(&deviceA, a.size() * sizeof(a[0])));
  CUDA_CHECK(cuMemAlloc(&deviceB, b.size() * sizeof(b[0])));
  CUDA_CHECK(cuMemAlloc(&deviceC, c.size() * sizeof(c[0])));
  CUDA_CHECK(cuMemcpyHtoD(deviceA, a.data(), a.size() * sizeof(a[0])));
  CUDA_CHECK(cuMemcpyHtoD(deviceB, b.data(), b.size() * sizeof(b[0])));

  void *arguments[] = {&deviceA, &deviceB, &deviceC};
  CUDA_CHECK(cuLaunchKernel(kernel, matrixSize / tileSize,
                            matrixSize / tileSize, 1, 1, 1, 1, 0, nullptr,
                            arguments, nullptr));
  CUDA_CHECK(cuCtxSynchronize());
  CUDA_CHECK(cuMemcpyDtoH(c.data(), deviceC, c.size() * sizeof(c[0])));

  CUDA_CHECK(cuMemFree(deviceC));
  CUDA_CHECK(cuMemFree(deviceB));
  CUDA_CHECK(cuMemFree(deviceA));
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuCtxDestroy(context));

  for (unsigned row = 0; row < matrixSize; ++row) {
    for (unsigned column = 0; column < matrixSize; ++column) {
      const float rowScale = row % 2 == 0 ? 1.0f : 0.5f;
      const float columnScale = column % 2 == 0 ? 1.0f : 2.0f;
      const float expected = matrixSize * rowScale * columnScale;
      const float actual = c[row * matrixSize + column];
      if (std::abs(actual - expected) > 0.001f) {
        std::cerr << "mismatch at " << row << ", " << column << ": expected "
                  << expected << ", got " << actual << '\n';
        return EXIT_FAILURE;
      }
    }
  }

  std::cout << "matmul passed\n";
  return EXIT_SUCCESS;
}
