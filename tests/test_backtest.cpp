#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "fixtures.hpp"
#include "tickforge/backtest.hpp"

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
  EXPECT_EQ(r.final_position % p.exec.lot_size, 0);
  EXPECT_LE(std::llabs(r.final_position), p.exec.lot_size);
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
  for (int i = 0; i < 5000; ++i) {
    t.ts.push_back(i);
    t.price.push_back(50.0);
    t.size.push_back(100);
    t.side.push_back(static_cast<std::int8_t>(i % 2 ? 1 : -1));
  }
  const auto r = run_backtest(t.view(), fast_params());
  EXPECT_EQ(r.trades, 0U);
  EXPECT_DOUBLE_EQ(r.pnl, 0.0);
}

TEST(Backtest, CostsOnlyHurt) {
  const auto t = test::small_ticks(100'000);
  auto no_cost = fast_params();
  no_cost.exec.fee_per_share = 0.0;
  no_cost.exec.slippage_bps = 0.0;
  auto costly = no_cost;
  costly.exec.fee_per_share = 0.01;
  costly.exec.slippage_bps = 2.0;
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
