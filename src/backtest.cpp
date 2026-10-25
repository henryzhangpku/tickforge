#include "tickforge/backtest.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#include "tickforge/memory.hpp"
#include "tickforge/spsc_ring.hpp"

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

BacktestResult run_backtest_pipelined(const TickColumns& cols, const BacktestParams& p) {
  using Ring = SpscRing<Tick, 4096>;
  auto ring = std::make_unique<Ring>();  // ~100 KB of slots: keep it off the stack
  std::atomic<bool> done{false};

  std::jthread producer([&] {
    const std::size_t n = cols.size_ticks();
    for (std::size_t i = 0; i < n; ++i) {
      const Tick t = cols[i];
      while (!ring->try_push(t)) std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  });

  Arena arena(backtest_scratch_bytes(p));
  Backtester bt(p, MeanReversion(p.strategy), arena.resource());
  const auto feed = [&bt](const Tick& t) noexcept { bt.on_tick(t); };
  for (;;) {
    if (ring->consume(feed) > 0) continue;
    // Empty. Only stop once the producer has finished *and* the ring is still
    // empty after seeing that: the acquire on `done` orders the final pushes.
    if (done.load(std::memory_order_acquire)) {
      while (ring->consume(feed) > 0) {
      }
      break;
    }
    std::this_thread::yield();
  }
  return bt.finish();
}

}  // namespace tickforge
