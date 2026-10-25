#include <gtest/gtest.h>

#include <cmath>
#include <memory_resource>
#include <vector>

#include "fixtures.hpp"
#include "tickforge/features.hpp"

using namespace tickforge;

namespace {
Tick tick(double price, std::uint32_t size, std::int8_t side = 1) { return {0, price, size, side}; }
}  // namespace

TEST(RollingVwap, MatchesBruteForce) {
  const auto t = test::small_ticks(10'000);
  constexpr std::size_t w = 37;
  RollingVwap f(w, std::pmr::new_delete_resource());
  for (std::size_t i = 0; i < t.price.size(); ++i) {
    const double got = f.update(t.view()[i]);
    double pv = 0.0, v = 0.0;
    for (std::size_t j = (i + 1 >= w ? i + 1 - w : 0); j <= i; ++j) {
      pv += t.price[j] * t.size[j];
      v += t.size[j];
    }
    ASSERT_NEAR(got, pv / v, 1e-9) << "at tick " << i;
  }
  EXPECT_TRUE(f.warm());
}

TEST(EwmaVol, MatchesRecurrence) {
  EwmaVol f(0.9);
  f.update(tick(100.0, 1));
  f.update(tick(101.0, 1));
  const double r1 = std::log(101.0 / 100.0);
  EXPECT_NEAR(f.value(), std::sqrt(0.1 * r1 * r1), 1e-15);
  f.update(tick(100.5, 1));
  const double r2 = std::log(100.5 / 101.0);
  EXPECT_NEAR(f.value(), std::sqrt(0.9 * 0.1 * r1 * r1 + 0.1 * r2 * r2), 1e-15);
}

TEST(OrderFlowImbalance, WindowedSignedVolume) {
  OrderFlowImbalance f(3, std::pmr::new_delete_resource());
  EXPECT_DOUBLE_EQ(f.update(tick(1, 100, 1)), 1.0);
  EXPECT_DOUBLE_EQ(f.update(tick(1, 100, -1)), 0.0);
  EXPECT_DOUBLE_EQ(f.update(tick(1, 200, -1)), -0.5);  // (100-100-200)/400
  EXPECT_DOUBLE_EQ(f.update(tick(1, 400, 1)), 1.0 / 7.0);  // first tick dropped: (-100-200+400)/700
}

TEST(Dispatch, CrtpAndVirtualAgree) {
  const auto t = test::small_ticks(5000);
  RollingVwap direct(50, std::pmr::new_delete_resource());
  VirtualFeature<RollingVwap> boxed(50, std::pmr::new_delete_resource());
  IFeature& iface = boxed;
  for (std::size_t i = 0; i < t.price.size(); ++i) ASSERT_EQ(direct.update(t.view()[i]), iface.update(t.view()[i]));
}
