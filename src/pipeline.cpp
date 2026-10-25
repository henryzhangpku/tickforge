// Two-thread replay: a producer thread feeds ticks through the SPSC ring to
// the backtester on the calling thread. Kept apart from backtest.cpp so
// single-threaded targets (the WebAssembly demo) can build without threads.
#include <atomic>
#include <memory>
#include <thread>

#include "tickforge/backtest.hpp"
#include "tickforge/memory.hpp"
#include "tickforge/spsc_ring.hpp"

namespace tickforge {

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
