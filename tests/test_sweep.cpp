#include <gtest/gtest.h>

#include <stdexcept>

#include "fixtures.hpp"
#include "tickforge/sweep.hpp"
#include "tickforge/thread_pool.hpp"

using namespace tickforge;

TEST(ThreadPool, RunsTasksAndReturnsValues) {
  ThreadPool pool(4);
  std::vector<std::future<int>> fs;
  for (int i = 0; i < 100; ++i) fs.push_back(pool.submit([i] { return i * i; }));
  for (int i = 0; i < 100; ++i) EXPECT_EQ(fs[static_cast<std::size_t>(i)].get(), i * i);
}

TEST(ThreadPool, PropagatesExceptions) {
  ThreadPool pool(2);
  auto f = pool.submit([]() -> int { throw std::runtime_error("boom"); });
  EXPECT_THROW(f.get(), std::runtime_error);
}

TEST(Sweep, ExpandIsCartesianProduct) {
  SweepGrid g;
  g.windows = {100, 200};
  g.entry_z = {1.0, 2.0, 3.0};
  g.exit_z = {0.5};
  g.ewma_lambda = {0.99, 0.995};
  const auto ps = expand(g);
  ASSERT_EQ(ps.size(), 12U);
  EXPECT_EQ(ps.front().strategy.window, 100);
  EXPECT_EQ(ps.back().strategy.window, 200);
  EXPECT_DOUBLE_EQ(ps[1].strategy.ewma_lambda, 0.995);  // fastest-varying (ofi has one value)
}

TEST(Sweep, MatchesSequentialBacktests) {
  const auto t = test::small_ticks(30'000);
  const auto ps = expand(SweepGrid{});
  const auto rs = run_sweep(t.view(), ps, 3);
  ASSERT_EQ(rs.size(), ps.size());
  for (std::size_t i = 0; i < ps.size(); ++i) EXPECT_EQ(rs[i], run_backtest(t.view(), ps[i])) << i;
}

TEST(Sweep, DeterministicAcrossThreadCounts) {
  const auto t = test::small_ticks(30'000);
  SweepGrid g;
  g.windows = {100, 250, 400};
  g.entry_z = {1.0, 1.5, 2.0, 2.5};
  const auto ps = expand(g);
  const auto baseline = run_sweep(t.view(), ps, 1);
  for (std::size_t threads : {2U, 3U, 4U, 8U}) {
    EXPECT_EQ(run_sweep(t.view(), ps, threads), baseline) << threads << " threads";
  }
}
