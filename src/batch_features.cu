// CUDA path for rolling_features_batch. Compiled only with -DTICKFORGE_CUDA=ON.
//
// Mental model, briefly:
//  * One thread per symbol, looping over time. Neighbouring threads in a warp
//    (32 threads) handle neighbouring symbols, and the data is time-major, so
//    at each step the warp reads 32 consecutive doubles = 256 contiguous bytes:
//    a coalesced load, the single most important rule for GPU bandwidth.
//  * Running sums and the EWMA variance live in registers (fastest tier); the
//    value leaving the window is re-read from global memory, where the L2
//    cache usually still has it. No shared memory is needed for this access
//    pattern because nothing is reused across threads.
//  * The kernel is bandwidth-bound (a few flops per 8-byte load), so the win
//    over a CPU comes from HBM/GDDR bandwidth, and only once there are enough
//    symbols (thousands) to fill the device. Host<->device copies over PCIe
//    usually dominate for small batches; a real system keeps data resident.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "tickforge/batch_features.hpp"

namespace tickforge {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__global__ void rolling_kernel(const double* __restrict__ px, std::size_t S, std::size_t T, std::size_t window,
                               double lam, double* __restrict__ mean, double* __restrict__ vol) {
  const std::size_t s = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= S) return;
  double sum = 0.0;
  double var = 0.0;
  double prev = px[s];
  for (std::size_t t = 0; t < T; ++t) {
    const double p = px[t * S + s];
    sum += p - (t >= window ? px[(t - window) * S + s] : 0.0);
    const double count = static_cast<double>(t + 1 < window ? t + 1 : window);
    mean[t * S + s] = sum / count;
    if (t > 0) {
      const double r = log(p / prev);
      var = lam * var + (1.0 - lam) * r * r;
    }
    vol[t * S + s] = sqrt(var);
    prev = p;
  }
}

// RAII device buffer.
struct DeviceBuffer {
  double* ptr = nullptr;
  explicit DeviceBuffer(std::size_t n) { check(cudaMalloc(&ptr, n * sizeof(double)), "cudaMalloc"); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}  // namespace

bool cuda_device_present() noexcept {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void rolling_features_batch_cuda(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                                 std::span<double> out_mean, std::span<double> out_vol) {
  const std::size_t n = prices.size();
  const std::size_t T = n / n_symbols;
  DeviceBuffer d_px(n), d_mean(n), d_vol(n);
  check(cudaMemcpy(d_px.ptr, prices.data(), n * sizeof(double), cudaMemcpyHostToDevice), "H2D");
  constexpr unsigned kBlock = 128;  // 4 warps per block
  const auto blocks = static_cast<unsigned>((n_symbols + kBlock - 1) / kBlock);
  rolling_kernel<<<blocks, kBlock>>>(d_px.ptr, n_symbols, T, p.window, p.ewma_lambda, d_mean.ptr, d_vol.ptr);
  check(cudaGetLastError(), "launch");
  check(cudaMemcpy(out_mean.data(), d_mean.ptr, n * sizeof(double), cudaMemcpyDeviceToHost), "D2H mean");
  check(cudaMemcpy(out_vol.data(), d_vol.ptr, n * sizeof(double), cudaMemcpyDeviceToHost), "D2H vol");
}

}  // namespace tickforge
