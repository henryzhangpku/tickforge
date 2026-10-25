#include "tickforge/sweep.hpp"

#include <charconv>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tickforge/memory.hpp"
#include "tickforge/thread_pool.hpp"
#include "tickforge/work_stealing_pool.hpp"

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

namespace {

template <class Pool>
void run_on(Pool& pool, std::span<const TickColumns> universe, const std::vector<BacktestParams>& params,
            std::vector<SweepRow>& rows) {
  std::vector<std::future<void>> pending;
  pending.reserve(rows.size());
  for (auto& row : rows) {
    pending.push_back(pool.submit([&universe, &params, &row] {
      // Each task owns its arena: no shared allocator, no contention, and the
      // whole scratch space goes away in one release at the end.
      const BacktestParams& p = params[row.param];
      Arena arena(backtest_scratch_bytes(p));
      row.result = run_backtest(universe[row.symbol], p, arena.resource());
    }));
  }
  for (auto& f : pending) f.get();  // rethrows the first task exception, if any
}

}  // namespace

std::vector<SweepRow> run_sweep(std::span<const TickColumns> universe, const std::vector<BacktestParams>& params,
                                const SweepOptions& opt) {
  if (opt.shard.count == 0 || opt.shard.index >= opt.shard.count) throw std::invalid_argument("bad shard");
  std::vector<SweepRow> rows;
  const std::size_t total = universe.size() * params.size();
  for (std::size_t i = 0; i < total; ++i)
    if (opt.shard.owns(i)) rows.push_back({i, i / params.size(), i % params.size(), {}});
  if (rows.empty()) return rows;
  if (opt.pool == PoolKind::WorkStealing) {
    WorkStealingPool pool(opt.threads);
    run_on(pool, universe, params, rows);
  } else {
    ThreadPool pool(opt.threads);
    run_on(pool, universe, params, rows);
  }
  return rows;
}

std::vector<BacktestResult> run_sweep(const TickColumns& cols, const std::vector<BacktestParams>& params,
                                      std::size_t threads, PoolKind pool) {
  const auto rows = run_sweep(std::span<const TickColumns>(&cols, 1), params, {threads, pool, {}});
  std::vector<BacktestResult> out;
  out.reserve(rows.size());
  for (const auto& r : rows) out.push_back(r.result);
  return out;
}

namespace {
std::size_t parse_index(const char* name, const char* v) {
  std::size_t out = 0;
  const std::string_view s(v);
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  if (ec != std::errc{} || ptr != s.data() + s.size())
    throw std::invalid_argument(std::string(name) + "='" + std::string(s) + "' is not a non-negative integer");
  return out;
}
}  // namespace

Shard shard_from_env(const std::function<const char*(const char*)>& getenv) {
  static constexpr const char* kPairs[][2] = {{"TICKFORGE_SHARD", "TICKFORGE_SHARDS"},
                                             {"JOB_COMPLETION_INDEX", "TICKFORGE_SHARDS"},
                                             {"CLOUD_RUN_TASK_INDEX", "CLOUD_RUN_TASK_COUNT"},
                                             {"AWS_BATCH_JOB_ARRAY_INDEX", "TICKFORGE_SHARDS"}};
  for (const auto& pair : kPairs) {
    const char* idx = getenv(pair[0]);
    if (idx == nullptr || *idx == '\0') continue;
    const char* cnt = getenv(pair[1]);
    if (cnt == nullptr || *cnt == '\0') throw std::invalid_argument(std::string(pair[0]) + " set but " + pair[1] + " is not");
    Shard s{parse_index(pair[0], idx), parse_index(pair[1], cnt)};
    if (s.count == 0 || s.index >= s.count) throw std::invalid_argument("shard index out of range");
    return s;
  }
  return {};
}

Shard shard_from_env() {
  return shard_from_env([](const char* name) -> const char* { return std::getenv(name); });
}

}  // namespace tickforge
