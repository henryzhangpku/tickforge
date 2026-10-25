// Parameter sweeps: the cartesian product of a grid, one backtest per point,
// spread over a thread pool.
//
// Determinism: every backtest is single-threaded and reads only immutable
// tick data plus its own parameters, and each result is written to the slot
// matching its grid index. So the output is bit-identical for any thread
// count or scheduling order (tests/test_sweep.cpp checks 1..8 threads).
#pragma once

#include <cstddef>
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

// threads == 0 uses hardware_concurrency().
std::vector<BacktestResult> run_sweep(const TickColumns& cols, const std::vector<BacktestParams>& params,
                                      std::size_t threads = 0);

}  // namespace tickforge
