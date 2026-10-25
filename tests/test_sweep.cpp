#include <gtest/gtest.h>

#include <atomic>
#include <latch>
#include <map>
#include <stdexcept>
#include <string>

#include "fixtures.hpp"
#include "tickforge/sweep.hpp"
#include "tickforge/thread_pool.hpp"
#include "tickforge/work_stealing_pool.hpp"

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
    EXPECT_EQ(run_sweep(t.view(), ps, threads, PoolKind::Central), baseline) << threads << " threads, central";
    EXPECT_EQ(run_sweep(t.view(), ps, threads, PoolKind::WorkStealing), baseline) << threads << " threads, stealing";
  }
}

TEST(WorkStealingPool, RunsTasksAndPropagatesExceptions) {
  WorkStealingPool pool(4);
  std::vector<std::future<int>> fs;
  for (int i = 0; i < 1000; ++i) fs.push_back(pool.submit([i] { return 2 * i; }));
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(fs[static_cast<std::size_t>(i)].get(), 2 * i);
  auto bad = pool.submit([]() -> int { throw std::runtime_error("boom"); });
  EXPECT_THROW(bad.get(), std::runtime_error);
}

// Tasks that spawn children land on the spawning worker's own deque; idle
// workers must steal them or the tree would run on one thread.
TEST(WorkStealingPool, NestedSpawnsAreStolen) {
  constexpr int kDepth = 12;  // 4096 leaves
  WorkStealingPool pool(4);
  std::atomic<int> leaves{0};
  std::latch done(1 << kDepth);
  std::function<void(int)> node = [&](int depth) {
    if (depth == kDepth) {
      volatile double x = 0;
      for (int i = 0; i < 2000; ++i) x = x + i;  // a little work per leaf
      leaves.fetch_add(1);
      done.count_down();
      return;
    }
    pool.post([&, depth] { node(depth + 1); });
    pool.post([&, depth] { node(depth + 1); });
  };
  pool.post([&] { node(0); });
  done.wait();
  EXPECT_EQ(leaves.load(), 1 << kDepth);
  EXPECT_GT(pool.steals(), 0U);
}

TEST(WorkStealingPool, DestructorDrainsQueuedWork) {
  std::atomic<int> ran{0};
  {
    WorkStealingPool pool(2);
    for (int i = 0; i < 500; ++i) pool.post([&] { ran.fetch_add(1); });
  }
  EXPECT_EQ(ran.load(), 500);
}

TEST(Sweep, UniverseRowsAreSymbolMajor) {
  const auto a = test::small_ticks(20'000, 1);
  const auto b = test::small_ticks(20'000, 2);
  const std::vector<TickColumns> universe = {a.view(), b.view()};
  SweepGrid g;
  g.windows = {100, 300};
  g.entry_z = {1.0};
  g.exit_z = {0.25};
  const auto ps = expand(g);
  const auto rows = run_sweep(universe, ps);
  ASSERT_EQ(rows.size(), 4U);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i].index, i);
    EXPECT_EQ(rows[i].symbol, i / 2);
    EXPECT_EQ(rows[i].param, i % 2);
    EXPECT_EQ(rows[i].result, run_backtest(universe[rows[i].symbol], ps[rows[i].param])) << i;
  }
}

TEST(Sweep, ShardsPartitionTheWork) {
  const auto a = test::small_ticks(20'000, 1);
  const auto b = test::small_ticks(20'000, 2);
  const std::vector<TickColumns> universe = {a.view(), b.view()};
  const auto ps = expand(SweepGrid{});
  const auto full = run_sweep(universe, ps);
  std::map<std::size_t, SweepRow> merged;
  for (std::size_t k = 0; k < 3; ++k) {
    SweepOptions o;
    o.shard = {k, 3};
    for (const auto& r : run_sweep(universe, ps, o)) {
      EXPECT_EQ(r.index % 3, k);
      EXPECT_TRUE(merged.emplace(r.index, r).second) << "task " << r.index << " ran twice";
    }
  }
  ASSERT_EQ(merged.size(), full.size());
  for (const auto& r : full) EXPECT_EQ(merged.at(r.index), r);
}

TEST(Sweep, ShardFromEnvironment) {
  const auto env = [](std::map<std::string, std::string> m) {
    return [m = std::move(m)](const char* k) -> const char* {
      const auto it = m.find(k);
      return it == m.end() ? nullptr : it->second.c_str();
    };
  };
  EXPECT_EQ(shard_from_env(env({})).count, 1U);
  auto s = shard_from_env(env({{"JOB_COMPLETION_INDEX", "2"}, {"TICKFORGE_SHARDS", "5"}}));
  EXPECT_EQ(s.index, 2U);
  EXPECT_EQ(s.count, 5U);
  s = shard_from_env(env({{"CLOUD_RUN_TASK_INDEX", "0"}, {"CLOUD_RUN_TASK_COUNT", "4"}}));
  EXPECT_EQ(s.count, 4U);
  s = shard_from_env(env({{"TICKFORGE_SHARD", "1"}, {"TICKFORGE_SHARDS", "2"}, {"JOB_COMPLETION_INDEX", "0"}}));
  EXPECT_EQ(s.index, 1U);  // explicit wins
  EXPECT_THROW(shard_from_env(env({{"AWS_BATCH_JOB_ARRAY_INDEX", "3"}})), std::invalid_argument);
  EXPECT_THROW(shard_from_env(env({{"TICKFORGE_SHARD", "4"}, {"TICKFORGE_SHARDS", "4"}})), std::invalid_argument);
  EXPECT_THROW(shard_from_env(env({{"TICKFORGE_SHARD", "x"}, {"TICKFORGE_SHARDS", "4"}})), std::invalid_argument);
}
