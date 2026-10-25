#include "tickforge/batch_features.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace tickforge {

namespace {
void check_shapes(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                  std::span<double> out_mean, std::span<double> out_vol) {
  if (n_symbols == 0 || prices.size() % n_symbols != 0)
    throw std::invalid_argument("prices.size() must be a multiple of n_symbols");
  if (out_mean.size() != prices.size() || out_vol.size() != prices.size())
    throw std::invalid_argument("output spans must match prices");
  if (p.window == 0) throw std::invalid_argument("window must be > 0");
}
}  // namespace

bool cuda_available() noexcept {
#ifdef TICKFORGE_HAS_CUDA
  return cuda_device_present();
#else
  return false;
#endif
}

void rolling_features_batch_cpu(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                                std::span<double> out_mean, std::span<double> out_vol) {
  check_shapes(prices, n_symbols, p, out_mean, out_vol);
  const std::size_t S = n_symbols;
  const std::size_t T = prices.size() / S;
  const double lam = p.ewma_lambda;
  // Per-symbol running state, the CPU analogue of the GPU's per-thread registers.
  std::vector<double> sum(S, 0.0), var(S, 0.0);
  for (std::size_t t = 0; t < T; ++t) {
    const double* row = prices.data() + t * S;
    const double* prev = t > 0 ? row - S : nullptr;
    const double* old = t >= p.window ? row - p.window * S : nullptr;
    const double count = static_cast<double>(t + 1 < p.window ? t + 1 : p.window);
    double* mean = out_mean.data() + t * S;
    double* vol = out_vol.data() + t * S;
    for (std::size_t s = 0; s < S; ++s) {
      sum[s] += row[s] - (old != nullptr ? old[s] : 0.0);
      mean[s] = sum[s] / count;
      if (prev != nullptr) {
        const double r = std::log(row[s] / prev[s]);
        var[s] = lam * var[s] + (1.0 - lam) * r * r;
      }
      vol[s] = std::sqrt(var[s]);
    }
  }
}

Backend rolling_features_batch(std::span<const double> prices, std::size_t n_symbols, const BatchFeatureParams& p,
                               std::span<double> out_mean, std::span<double> out_vol, Backend want) {
  if (want == Backend::Cuda && !cuda_available()) throw std::runtime_error("CUDA backend requested but unavailable");
#ifdef TICKFORGE_HAS_CUDA
  if (want != Backend::Cpu && cuda_available()) {
    check_shapes(prices, n_symbols, p, out_mean, out_vol);
    rolling_features_batch_cuda(prices, n_symbols, p, out_mean, out_vol);
    return Backend::Cuda;
  }
#endif
  rolling_features_batch_cpu(prices, n_symbols, p, out_mean, out_vol);
  return Backend::Cpu;
}

}  // namespace tickforge
