// Rolling features for many symbols at once: a rolling mean of price and an
// EWMA volatility of log returns, for every (tick, symbol) cell.
//
// Layout is time-major: prices[t * n_symbols + s]. On the GPU one thread owns
// one symbol and walks forward in time; at each step the threads of a warp
// read n_symbols-adjacent doubles, so the loads coalesce into a few wide
// memory transactions. The CPU fallback uses the same layout with symbols in
// the inner loop, which is contiguous and auto-vectorises.
//
// Build with -DTICKFORGE_CUDA=ON to compile the CUDA path; otherwise (and at
// runtime when no device is present) the CPU path runs.
#pragma once

#include <cstddef>
#include <span>

namespace tickforge {

enum class Backend { Auto, Cpu, Cuda };

struct BatchFeatureParams {
  std::size_t window = 100;
  double ewma_lambda = 0.995;
};

// True when built with CUDA and a device is visible.
bool cuda_available() noexcept;

// prices, out_mean, out_vol: n_ticks * n_symbols each, time-major.
// Returns the backend that actually ran. Backend::Cuda throws if unavailable.
Backend rolling_features_batch(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                               std::span<double> out_mean, std::span<double> out_vol, Backend want = Backend::Auto);

// Reference implementation, always on the CPU.
void rolling_features_batch_cpu(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                                std::span<double> out_mean, std::span<double> out_vol);

#ifdef TICKFORGE_HAS_CUDA
void rolling_features_batch_cuda(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                                 std::span<double> out_mean, std::span<double> out_vol);
bool cuda_device_present() noexcept;
#endif

}  // namespace tickforge
