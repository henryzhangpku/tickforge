#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory_resource>
#include <vector>

#include "fixtures.hpp"
#include "tickforge/batch_features.hpp"
#include "tickforge/features.hpp"
#include "tickforge/synth.hpp"

using namespace tickforge;

// Each symbol column of the batch must equal running the streaming EwmaVol on
// that symbol alone, and the mean must equal a brute-force rolling mean.
TEST(BatchFeatures, MatchesPerSymbolReference) {
  constexpr std::size_t S = 5, T = 3000;
  std::vector<double> px(S * T);
  for (std::size_t s = 0; s < S; ++s) {
    const auto t = test::small_ticks(T, 100 + s);
    for (std::size_t i = 0; i < T; ++i) px[i * S + s] = t.price[i];
  }
  const BatchFeatureParams p{25, 0.97};
  std::vector<double> mean(S * T), vol(S * T);
  EXPECT_EQ(rolling_features_batch(px, S, p, mean, vol, Backend::Cpu), Backend::Cpu);

  for (std::size_t s = 0; s < S; ++s) {
    EwmaVol ev(p.ewma_lambda);
    for (std::size_t i = 0; i < T; ++i) {
      ev.update(Tick{0, px[i * S + s], 1, 1});
      ASSERT_NEAR(vol[i * S + s], ev.value(), 1e-15);
      double sum = 0.0;
      const std::size_t lo = i + 1 >= p.window ? i + 1 - p.window : 0;
      for (std::size_t j = lo; j <= i; ++j) sum += px[j * S + s];
      ASSERT_NEAR(mean[i * S + s], sum / static_cast<double>(i - lo + 1), 1e-9);
    }
  }
}

TEST(BatchFeatures, AutoFallsBackToCpuWithoutCuda) {
  std::vector<double> px(10, 1.0), mean(10), vol(10);
  const Backend b = rolling_features_batch(px, 2, {}, mean, vol);
  EXPECT_EQ(b, cuda_available() ? Backend::Cuda : Backend::Cpu);
}

TEST(BatchFeatures, ValidatesShapes) {
  std::vector<double> px(10, 1.0), mean(10), vol(9);
  EXPECT_THROW(rolling_features_batch(px, 3, {}, mean, mean, Backend::Cpu), std::invalid_argument);
  EXPECT_THROW(rolling_features_batch(px, 2, {}, mean, vol, Backend::Cpu), std::invalid_argument);
}

// ---- GPU path: runs only when built with CUDA *and* a device is present ----

TEST(BatchFeaturesCuda, MatchesCpu) {
  if (!cuda_available()) GTEST_SKIP() << "no CUDA device (or built without -DTICKFORGE_CUDA=ON)";
  constexpr std::size_t S = 1000, T = 2000;  // enough symbols for several blocks
  std::vector<double> px(S * T);
  Xoshiro256 rng(9);
  for (std::size_t s = 0; s < S; ++s) {
    double p = 50.0 + 100.0 * rng.uniform();
    for (std::size_t t = 0; t < T; ++t) {
      p *= std::exp(1e-3 * rng.normal());
      px[t * S + s] = p;
    }
  }
  const BatchFeatureParams p{50, 0.99};
  std::vector<double> m_cpu(S * T), v_cpu(S * T), m_gpu(S * T), v_gpu(S * T);
  rolling_features_batch(px, S, p, m_cpu, v_cpu, Backend::Cpu);
  EXPECT_EQ(rolling_features_batch(px, S, p, m_gpu, v_gpu, Backend::Cuda), Backend::Cuda);
  for (std::size_t i = 0; i < S * T; ++i) {
    // Same summation order on both sides; only log()/sqrt() rounding may differ.
    ASSERT_NEAR(m_gpu[i], m_cpu[i], 1e-9 * m_cpu[i]) << i;
    ASSERT_NEAR(v_gpu[i], v_cpu[i], 1e-9 * v_cpu[i] + 1e-15) << i;
  }
}

TEST(BatchFeaturesCuda, AutoPrefersGpu) {
  if (!cuda_available()) GTEST_SKIP() << "no CUDA device";
  std::vector<double> px(64, 1.0), mean(64), vol(64);
  EXPECT_EQ(rolling_features_batch(px, 8, {}, mean, vol), Backend::Cuda);
}

TEST(BatchFeaturesCuda, ReportsWhetherCompiledIn) {
#ifdef TICKFORGE_HAS_CUDA
  RecordProperty("cuda_compiled", "yes");
  std::cout << "[ info ] CUDA path compiled in; device present: " << (cuda_available() ? "yes" : "no") << "\n";
#else
  RecordProperty("cuda_compiled", "no");
  EXPECT_FALSE(cuda_available());
#endif
}
