// Proves the backtest hot loop does not touch the heap.
//
// This binary replaces every global allocation function (throwing, nothrow,
// aligned) with counting versions over malloc/free, so new/delete pairs stay
// consistent under AddressSanitizer. Setup (feature buffers, arena) may
// allocate; once the backtester is constructed, feeding it ticks must leave
// the counter unchanged.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <memory_resource>
#include <new>

#include "fixtures.hpp"
#include "tickforge/backtest.hpp"
#include "tickforge/memory.hpp"

namespace {
std::atomic<std::size_t> g_news{0};
}

void* operator new(std::size_t n) {
  g_news.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  g_news.fetch_add(1, std::memory_order_relaxed);
  const auto al = static_cast<std::size_t>(a);
  if (void* p = std::aligned_alloc(al, (n + al - 1) / al * al)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  g_news.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept { return ::operator new(n, t); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  g_news.fetch_add(1, std::memory_order_relaxed);
  const auto al = static_cast<std::size_t>(a);
  return std::aligned_alloc(al, (n + al - 1) / al * al);
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t& t) noexcept {
  return ::operator new(n, a, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

using namespace tickforge;

TEST(NoAlloc, CounterSeesOrdinaryAllocations) {
  const std::size_t before = g_news.load();
  // Call the function directly: new-expressions may be elided by the optimiser.
  void* p = ::operator new(16);
  ::operator delete(p);
  EXPECT_GT(g_news.load(), before);
}

TEST(NoAlloc, HotLoopAllocatesNothing) {
  const auto t = test::small_ticks(200'000);
  BacktestParams p;
  p.strategy.window = 300;
  p.strategy.entry_z = 0.5;

  // The arena's upstream is null: if the backtester ever outgrew its
  // pre-sized scratch, allocation would throw instead of silently using malloc.
  Arena arena(backtest_scratch_bytes(p), std::pmr::null_memory_resource());
  Backtester bt(p, MeanReversion(p.strategy), arena.resource());

  const std::size_t before = g_news.load();
  const auto cols = t.view();
  for (std::size_t i = 0; i < cols.size_ticks(); ++i) bt.on_tick(cols[i]);
  const auto r = bt.finish();
  const std::size_t after = g_news.load();

  EXPECT_EQ(after - before, 0U) << "operator new called on the hot path";
  EXPECT_GT(r.trades, 0U);  // and it did real work
}
