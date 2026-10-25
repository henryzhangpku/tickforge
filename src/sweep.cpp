#include "tickforge/sweep.hpp"

#include <future>

#include "tickforge/memory.hpp"
#include "tickforge/thread_pool.hpp"

namespace tickforge {

std::vector<BacktestParams> expand(const SweepGrid& g) {
  std::vector<BacktestParams> out;
  out.reserve(g.windows.size() * g.entry_z.size() * g.exit_z.size() * g.ewma_lambda.size() * g.ofi_max.size());
  for (int w : g.windows)
    for (double ez : g.entry_z)
      for (double xz : g.exit_z)
        for (double lam : g.ewma_lambda)
          for (double ofi : g.ofi_max) {
            BacktestParams p;
            p.strategy = {w, lam, ez, xz, ofi};
            p.exec = g.exec;
            out.push_back(p);
          }
  return out;
}

std::vector<BacktestResult> run_sweep(const TickColumns& cols, const std::vector<BacktestParams>& params,
                                      std::size_t threads) {
  std::vector<BacktestResult> results(params.size());
  if (params.empty()) return results;
  ThreadPool pool(threads);
  std::vector<std::future<void>> pending;
  pending.reserve(params.size());
  for (std::size_t i = 0; i < params.size(); ++i) {
    pending.push_back(pool.submit([&cols, &params, &results, i] {
      // Each task owns its arena: no shared allocator, no contention, and the
      // whole scratch space goes away in one release at the end.
      Arena arena(backtest_scratch_bytes(params[i]));
      results[i] = run_backtest(cols, params[i], arena.resource());
    }));
  }
  for (auto& f : pending) f.get();  // rethrows the first task exception, if any
  return results;
}

}  // namespace tickforge
