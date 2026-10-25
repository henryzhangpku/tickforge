#include <gtest/gtest.h>

#include <cmath>
#include <memory_resource>
#include <vector>

#include "fixtures.hpp"
#include "tickforge/batch_features.hpp"
#include "tickforge/features.hpp"

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
