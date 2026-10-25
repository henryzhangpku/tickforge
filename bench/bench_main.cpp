// Google Benchmark microbenchmarks. Results on the Pages site come from
// running this binary (Release) and converting its JSON with
// scripts/bench_to_site.py.
#include <benchmark/benchmark.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <thread>
#include <vector>

#include <unistd.h>

#include "tickforge/backtest.hpp"
#include "tickforge/batch_features.hpp"
#include "tickforge/features.hpp"
#include "tickforge/memory.hpp"
#include "tickforge/spsc_ring.hpp"
#include "tickforge/sweep.hpp"
#include "tickforge/synth.hpp"
#include "tickforge/tick_store.hpp"

namespace tf = tickforge;

namespace {

constexpr std::uint64_t kTicks = 1'000'000;

// One synthetic file, generated once, read through mmap like a real run.
const tf::MappedTickFile& dataset() {
  static const tf::MappedTickFile file = [] {
    const auto path = std::filesystem::temp_directory_path() / ("tickforge_bench_" + std::to_string(::getpid()) + ".tf");
    tf::SynthParams p;
    p.n_ticks = kTicks;
    const auto t = tf::generate_ticks(p);
    tf::write_tick_file(path, t.view(), p.seed);
    tf::MappedTickFile f(path);
    std::filesystem::remove(path);  // the mapping stays valid after unlink
    return f;
  }();
  return file;
}

tf::BacktestParams default_params() {
  tf::BacktestParams p;
  p.strategy.window = 500;
  return p;
}

// ---- storage ---------------------------------------------------------------

void BM_MmapScanPrices(benchmark::State& state) {
  const auto px = dataset().columns().price;
  for (auto _ : state) {
    double s = std::accumulate(px.begin(), px.end(), 0.0);
    benchmark::DoNotOptimize(s);
  }
  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * px.size_bytes()));
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * px.size()));
}
BENCHMARK(BM_MmapScanPrices)->Unit(benchmark::kMicrosecond);

// ---- memory ----------------------------------------------------------------

void BM_ScratchNewDelete(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<double> a(512), b(512), c(512);
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    benchmark::DoNotOptimize(c.data());
  }
}
BENCHMARK(BM_ScratchNewDelete);

void BM_ScratchArena(benchmark::State& state) {
  tf::AlignedBuffer<std::byte> block(64 * 1024);
  for (auto _ : state) {
    std::pmr::monotonic_buffer_resource arena(block.data(), block.size(), std::pmr::null_memory_resource());
    std::pmr::vector<double> a(512, &arena), b(512, &arena), c(512, &arena);
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    benchmark::DoNotOptimize(c.data());
  }
}
BENCHMARK(BM_ScratchArena);

// ---- pipeline --------------------------------------------------------------

void BM_SpscRingTransfer(benchmark::State& state) {
  constexpr std::size_t kN = 1 << 20;
  for (auto _ : state) {
    auto ring = std::make_unique<tf::SpscRing<tf::Tick, 4096>>();
    std::jthread producer([&] {
      for (std::size_t i = 0; i < kN; ++i)
        while (!ring->try_push(tf::Tick{static_cast<std::int64_t>(i), 1.0, 1, 1})) std::this_thread::yield();
    });
    std::size_t got = 0;
    tf::Tick t{};
    while (got < kN) {
      if (ring->try_pop(t))
        ++got;
      else
        std::this_thread::yield();
    }
    benchmark::DoNotOptimize(t);
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * kN));
}
BENCHMARK(BM_SpscRingTransfer)->Unit(benchmark::kMillisecond)->UseRealTime();

// ---- CRTP vs virtual -------------------------------------------------------

void BM_FeaturesCrtp(benchmark::State& state) {
  const auto& cols = dataset().columns();
  for (auto _ : state) {
    tf::FeatureSet fs(500, 0.995, std::pmr::new_delete_resource());
    double acc = 0.0;
    for (std::size_t i = 0; i < cols.size_ticks(); ++i) {
      const auto f = fs.update(cols[i]);
      acc += f.vwap + f.vol + f.ofi;
    }
    benchmark::DoNotOptimize(acc);
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * cols.size_ticks()));
}
BENCHMARK(BM_FeaturesCrtp)->Unit(benchmark::kMillisecond);

void BM_FeaturesVirtual(benchmark::State& state) {
  const auto& cols = dataset().columns();
  auto* mr = std::pmr::new_delete_resource();
  for (auto _ : state) {
    std::vector<std::unique_ptr<tf::IFeature>> fs;
    fs.push_back(std::make_unique<tf::VirtualFeature<tf::RollingVwap>>(500, mr));
    fs.push_back(std::make_unique<tf::VirtualFeature<tf::EwmaVol>>(0.995));
    fs.push_back(std::make_unique<tf::VirtualFeature<tf::OrderFlowImbalance>>(500, mr));
    double acc = 0.0;
    for (std::size_t i = 0; i < cols.size_ticks(); ++i) {
      const tf::Tick t = cols[i];
      for (auto& f : fs) acc += f->update(t);
    }
    benchmark::DoNotOptimize(acc);
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * cols.size_ticks()));
}
BENCHMARK(BM_FeaturesVirtual)->Unit(benchmark::kMillisecond);

void BM_BacktestCrtp(benchmark::State& state) {
  const auto& cols = dataset().columns();
  const auto p = default_params();
  for (auto _ : state) benchmark::DoNotOptimize(tf::run_backtest(cols, p));
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * cols.size_ticks()));
}
BENCHMARK(BM_BacktestCrtp)->Unit(benchmark::kMillisecond);

void BM_BacktestVirtual(benchmark::State& state) {
  const auto& cols = dataset().columns();
  const auto p = default_params();
  for (auto _ : state) {
    tf::Arena arena(tf::backtest_scratch_bytes(p));
    tf::VirtualMeanReversion plugin(p.strategy);
    tf::BasicBacktester<tf::VirtualStrategyRef> bt(p, tf::VirtualStrategyRef(plugin), arena.resource());
    for (std::size_t i = 0; i < cols.size_ticks(); ++i) bt.on_tick(cols[i]);
    benchmark::DoNotOptimize(bt.finish());
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * cols.size_ticks()));
}
BENCHMARK(BM_BacktestVirtual)->Unit(benchmark::kMillisecond);

void BM_BacktestPipelined(benchmark::State& state) {
  const auto& cols = dataset().columns();
  const auto p = default_params();
  for (auto _ : state) benchmark::DoNotOptimize(tf::run_backtest_pipelined(cols, p));
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * cols.size_ticks()));
}
BENCHMARK(BM_BacktestPipelined)->Unit(benchmark::kMillisecond)->UseRealTime();

// ---- parallel sweep scaling ------------------------------------------------

void BM_Sweep(benchmark::State& state) {
  const auto& cols = dataset().columns();
  tf::SweepGrid g;
  g.windows = {200, 500, 1000, 2000};
  g.entry_z = {0.5, 0.75, 1.0, 1.5};
  g.exit_z = {0.0, 0.25};
  const auto params = tf::expand(g);  // 32 backtests x 1M ticks
  const auto threads = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) benchmark::DoNotOptimize(tf::run_sweep(cols, params, threads));
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * params.size() * cols.size_ticks()));
  state.counters["backtests"] = static_cast<double>(params.size());
}
BENCHMARK(BM_Sweep)
    ->ArgName("threads")
    ->DenseRange(1, static_cast<int>(std::max(1U, std::thread::hardware_concurrency())))
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// ---- batch features --------------------------------------------------------

void BM_BatchFeaturesCpu(benchmark::State& state) {
  const auto S = static_cast<std::size_t>(state.range(0));
  constexpr std::size_t T = 4096;
  std::vector<double> px(S * T), mean(S * T), vol(S * T);
  tf::Xoshiro256 rng(1);
  for (std::size_t i = 0; i < px.size(); ++i) px[i] = 100.0 + rng.uniform();
  for (auto _ : state) {
    tf::rolling_features_batch_cpu(px, S, {100, 0.995}, mean, vol);
    benchmark::DoNotOptimize(mean.data());
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * S * T));
}
BENCHMARK(BM_BatchFeaturesCpu)->ArgName("symbols")->Arg(64)->Arg(512)->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
