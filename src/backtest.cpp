#include "tickforge/backtest.hpp"

#include <optional>
#include <stdexcept>

#include "tickforge/memory.hpp"

namespace tickforge {

BacktestResult run_backtest(const TickColumns& cols, const BacktestParams& p, std::pmr::memory_resource* mr,
                            std::span<double> equity_out) {
  if (!equity_out.empty() && equity_out.size() != cols.size_ticks())
    throw std::invalid_argument("equity_out must have one slot per tick");
  std::optional<Arena> arena;
  if (mr == nullptr) mr = arena.emplace(backtest_scratch_bytes(p)).resource();

  Backtester bt(p, MeanReversion(p.strategy), mr);
  const std::size_t n = cols.size_ticks();
  if (equity_out.empty()) {
    for (std::size_t i = 0; i < n; ++i) bt.on_tick(cols[i]);
  } else {
    for (std::size_t i = 0; i < n; ++i) bt.on_tick(cols[i], &equity_out[i]);
  }
  return bt.finish();
}

}  // namespace tickforge
