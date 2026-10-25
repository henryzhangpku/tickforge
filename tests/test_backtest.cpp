#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "fixtures.hpp"
#include "tickforge/backtest.hpp"
#include "tickforge/plugins.hpp"

using namespace tickforge;

namespace {
BacktestParams fast_params() {
  BacktestParams p;
  p.strategy.window = 200;
  p.strategy.entry_z = 1.5;
  return p;
}
}  // namespace

TEST(Backtest, TradesAndAccountsConsistently) {
  const auto t = test::small_ticks(100'000);
  const auto p = fast_params();
  std::vector<double> equity(t.price.size());
  const auto r = run_backtest(t.view(), p, nullptr, equity);
  EXPECT_EQ(r.ticks, t.price.size());
  EXPECT_GT(r.trades, 0U);
  EXPECT_NEAR(r.fees, static_cast<double>(r.shares) * p.exec.fee_per_share, 1e-6);
  EXPECT_LE(std::llabs(r.final_position), p.exec.lot_size);
  EXPECT_GT(r.exec_cost, 0.0);  // crossing the spread is never free
  // Final equity on the curve equals reported P&L.
  EXPECT_DOUBLE_EQ(equity.back(), r.pnl);
  // Max drawdown is the worst peak-to-trough on that curve.
  double peak = 0.0, dd = 0.0;
  for (double e : equity) {
    peak = std::max(peak, e);
    dd = std::max(dd, peak - e);
  }
  EXPECT_DOUBLE_EQ(dd, r.max_drawdown);
}

TEST(Backtest, NoTradesWithoutSignal) {
  // A constant price never deviates from its VWAP, so the strategy never trades.
  TickVectors t;
  for (int i = 0; i < 5000; ++i)
    t.push_back({i, 50.0, 100, static_cast<std::int8_t>(i % 2 ? 1 : -1), 49.99, 50.01, 500, 500});
  const auto r = run_backtest(t.view(), fast_params());
  EXPECT_EQ(r.trades, 0U);
  EXPECT_DOUBLE_EQ(r.pnl, 0.0);
}

TEST(Backtest, CostsOnlyHurt) {
  const auto t = test::small_ticks(100'000);
  auto no_cost = fast_params();
  no_cost.exec.fee_per_share = 0.0;
  no_cost.exec.slippage_bps = 0.0;
  no_cost.exec.impact_bps_per_lot = 0.0;
  auto costly = no_cost;
  costly.exec.fee_per_share = 0.01;
  costly.exec.slippage_bps = 2.0;
  costly.exec.impact_bps_per_lot = 1.0;
  const auto a = run_backtest(t.view(), no_cost);
  const auto b = run_backtest(t.view(), costly);
  EXPECT_EQ(a.trades, b.trades);  // costs do not change decisions
  EXPECT_LT(b.pnl, a.pnl);
}

TEST(Backtest, PipelinedMatchesDirect) {
  const auto t = test::small_ticks(60'000);
  const auto p = fast_params();
  EXPECT_EQ(run_backtest(t.view(), p), run_backtest_pipelined(t.view(), p));
}

TEST(Backtest, VirtualStrategyMatchesCrtp) {
  const auto t = test::small_ticks(30'000);
  const auto p = fast_params();
  VirtualMeanReversion plugin(p.strategy);
  BasicBacktester<VirtualStrategyRef> bt(p, VirtualStrategyRef(plugin), std::pmr::new_delete_resource());
  for (std::size_t i = 0; i < t.price.size(); ++i) bt.on_tick(t.view()[i]);
  EXPECT_EQ(bt.finish(), run_backtest(t.view(), p));
}

TEST(Backtest, RejectsWrongEquityLength) {
  const auto t = test::small_ticks(100);
  std::vector<double> equity(5);
  EXPECT_THROW(run_backtest(t.view(), fast_params(), nullptr, equity), std::invalid_argument);
}

TEST(Plugins, RegistryCreatesByName) {
  const auto t = test::small_ticks(30'000);
  const auto p = fast_params();
  auto plugin = make_strategy("mean_reversion", p.strategy);
  ASSERT_NE(plugin, nullptr);
  BasicBacktester<VirtualStrategyRef> bt(p, VirtualStrategyRef(*plugin), std::pmr::new_delete_resource());
  for (std::size_t i = 0; i < t.price.size(); ++i) bt.on_tick(t.view()[i]);
  EXPECT_EQ(bt.finish(), run_backtest(t.view(), p));
  EXPECT_EQ(make_strategy("no_such_strategy", p.strategy), nullptr);
}

TEST(Plugins, CustomStrategyRegisters) {
  struct AlwaysLong final : IStrategy {
    int on_tick(const Tick&, const Features&, int) noexcept override { return 1; }
  };
  EXPECT_TRUE(register_strategy("always_long", [](const StrategyParams&) { return std::make_unique<AlwaysLong>(); }));
  EXPECT_FALSE(register_strategy("always_long", [](const StrategyParams&) { return std::make_unique<AlwaysLong>(); }));
  const auto names = strategy_names();
  EXPECT_NE(std::find(names.begin(), names.end(), "always_long"), names.end());

  const auto t = test::small_ticks(1000);
  const auto p = fast_params();
  auto s = make_strategy("always_long", p.strategy);
  BasicBacktester<VirtualStrategyRef> bt(p, VirtualStrategyRef(*s), std::pmr::new_delete_resource());
  for (std::size_t i = 0; i < t.price.size(); ++i) bt.on_tick(t.view()[i]);
  const auto r = bt.finish();
  EXPECT_EQ(r.trades, 1U);
  EXPECT_EQ(r.final_position, p.exec.lot_size);
}

// ---- execution model --------------------------------------------------------

namespace {

// A strategy that wants to be long from its first tick onward.
struct GoLong : StrategyBase<GoLong> {
  int decide(const Tick&, const Features&, int) const noexcept { return 1; }
};

TickVectors flat_market(int n, std::int64_t gap_ns, std::uint32_t ask_size) {
  TickVectors t;
  for (int i = 0; i < n; ++i)
    t.push_back({i * gap_ns, 100.0, 100, 1, 99.99, 100.01, 1000, ask_size});
  return t;
}

BacktestResult run_go_long(const TickVectors& t, const ExecParams& e) {
  BacktestParams p;
  p.strategy.window = 5;
  p.exec = e;
  BasicBacktester<GoLong> bt(p, GoLong{}, std::pmr::new_delete_resource());
  const auto c = t.view();
  for (std::size_t i = 0; i < c.size_ticks(); ++i) bt.on_tick(c[i]);
  return bt.finish();
}

}  // namespace

TEST(Execution, BuysPayTheAskPlusImpact) {
  ExecParams e;
  e.lot_size = 100;
  e.fee_per_share = 0.0;
  e.impact_bps_per_lot = 2.0;
  const auto r = run_go_long(flat_market(10, 1'000'000, 1000), e);
  ASSERT_EQ(r.trades, 1U);
  EXPECT_EQ(r.final_position, 100);
  // 100 shares at 100.01 * (1 + 2bp): cost vs mid = 100 * (100.03 - 100.00).
  EXPECT_NEAR(r.exec_cost, 100 * (100.01 * 1.0002 - 100.0), 1e-9);
  EXPECT_NEAR(r.pnl, -r.exec_cost, 1e-9);
}

TEST(Execution, PartialFillsAtTheTouch) {
  ExecParams e;
  e.lot_size = 500;  // want 500 shares, only 200 displayed per tick
  const auto r = run_go_long(flat_market(10, 1'000'000, 200), e);
  EXPECT_EQ(r.final_position, 500);
  EXPECT_EQ(r.trades, 3U);         // 200 + 200 + 100
  EXPECT_EQ(r.partial_fills, 2U);  // the first two left a remainder

  e.respect_touch_size = false;    // ignore displayed size: one fill
  EXPECT_EQ(run_go_long(flat_market(10, 1'000'000, 200), e).trades, 1U);
}

TEST(Execution, LatencyDelaysTheFill) {
  // Ticks every 1ms. The order is decided on tick 0; with 5.5ms latency the
  // first eligible tick is t=6ms, so the position is held for 4 of 10 ticks.
  ExecParams e;
  e.latency_ns = 5'500'000;
  std::vector<double> eq(10);
  BacktestParams p;
  p.strategy.window = 5;
  p.exec = e;
  BasicBacktester<GoLong> bt(p, GoLong{}, std::pmr::new_delete_resource());
  const auto t = flat_market(10, 1'000'000, 1000);
  for (std::size_t i = 0; i < 10; ++i) bt.on_tick(t.view()[i], &eq[i]);
  for (std::size_t i = 0; i < 6; ++i) EXPECT_EQ(eq[i], 0.0) << i;  // not filled yet
  EXPECT_LT(eq[6], 0.0);                                             // filled at t=6ms, paid the spread
  EXPECT_EQ(bt.finish().final_position, 100);
}

TEST(Execution, NeverFillsOnTheDecisionTick) {
  // Even with zero latency, an order decided on the last tick never fills.
  const auto r = run_go_long(flat_market(1, 1'000'000, 1000), {});
  EXPECT_EQ(r.trades, 0U);
  EXPECT_EQ(r.final_position, 0);
}

TEST(Backtest, AnnualisedSharpeScalesWithTimeSpan) {
  const auto t = test::small_ticks(100'000);
  const auto r = run_backtest(t.view(), fast_params());
  ASSERT_NE(r.sharpe, 0.0);
  const double span_s = static_cast<double>(t.ts.back() - t.ts.front()) * 1e-9;
  const double buckets = std::floor(static_cast<double>(t.ts.size()) / 1000.0);
  EXPECT_NEAR(r.sharpe_annual, r.sharpe * std::sqrt(buckets * kTradingSecondsPerYear / span_s), 1e-9);
}
