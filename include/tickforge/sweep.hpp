// Parameter sweeps: the cartesian product of a grid, one backtest per point,
// optionally across a universe of symbols, spread over a thread pool and
// optionally sharded across machines.
//
// Determinism: every backtest is single-threaded and reads only immutable
// tick data plus its own parameters, and each result is written to the slot
// matching its grid index. So the output is bit-identical for any thread
// count, pool type, scheduling order or shard split (tests/test_sweep.cpp).
#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "tickforge/backtest.hpp"

namespace tickforge {

struct SweepGrid {
  std::vector<int> windows{200, 500, 1000};
  std::vector<double> entry_z{0.5, 1.0, 1.5};
  std::vector<double> exit_z{0.0, 0.25};
  std::vector<double> ewma_lambda{0.995};
  std::vector<double> ofi_max{1.0};
  ExecParams exec{};
};

// Row-major expansion: windows vary slowest, ofi_max fastest.
std::vector<BacktestParams> expand(const SweepGrid& g);

enum class PoolKind { Central, WorkStealing };

// Which slice of the global task list this process runs: task i belongs to
// shard (i % count). One shard = everything.
struct Shard {
  std::size_t index = 0;
  std::size_t count = 1;
  bool owns(std::size_t task) const noexcept { return task % count == index; }
};

struct SweepOptions {
  std::size_t threads = 0;  // 0 = hardware_concurrency()
  PoolKind pool = PoolKind::WorkStealing;
  Shard shard{};
};

struct SweepRow {
  std::size_t index = 0;   // global task index = symbol * n_params + param
  std::size_t symbol = 0;  // position in the universe
  std::size_t param = 0;   // position in the parameter list
  BacktestResult result{};
  bool operator==(const SweepRow&) const = default;
};

// Every (symbol, params) pair this shard owns, in global-index order.
std::vector<SweepRow> run_sweep(std::span<const TickColumns> universe, const std::vector<BacktestParams>& params,
                                const SweepOptions& opt = {});

// Single-symbol convenience: one result per parameter set, in order.
std::vector<BacktestResult> run_sweep(const TickColumns& cols, const std::vector<BacktestParams>& params,
                                      std::size_t threads = 0, PoolKind pool = PoolKind::WorkStealing);

// Reads the shard from the environment, so one container image runs unchanged
// on any batch scheduler. First match wins:
//   TICKFORGE_SHARD / TICKFORGE_SHARDS            (explicit)
//   JOB_COMPLETION_INDEX / TICKFORGE_SHARDS       (Kubernetes Indexed Job)
//   CLOUD_RUN_TASK_INDEX / CLOUD_RUN_TASK_COUNT   (Google Cloud Run Jobs)
//   AWS_BATCH_JOB_ARRAY_INDEX / TICKFORGE_SHARDS  (AWS Batch array jobs)
// Throws std::invalid_argument on a malformed or out-of-range value.
Shard shard_from_env(const std::function<const char*(const char*)>& getenv);
Shard shard_from_env();

}  // namespace tickforge
