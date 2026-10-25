// Event-driven backtest: one tick in, the working order filled against that
// tick's quote, features updated, P&L marked at mid, strategy asked for a
// target. Orders decided on a tick fill on later ticks only (no look-ahead).
//
// BasicBacktester is templated on the strategy so the default path
// (MeanReversion, CRTP) inlines completely; VirtualStrategyRef plugs a
// runtime IStrategy into the same loop for comparison.
//
// All per-run state is allocated in the constructor from the memory_resource
// you pass (normally an Arena). on_tick() never allocates; tests/test_no_alloc
// replaces global operator new to prove it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory_resource>
#include <span>
#include <utility>

#include "tickforge/features.hpp"
#include "tickforge/memory.hpp"
#include "tickforge/strategy.hpp"
#include "tickforge/tick_store.hpp"

namespace tickforge {

// Execution model. Orders are market orders that cross the spread:
//  * latency: an order decided at tick t becomes eligible at ts(t) + latency_ns,
//    and never on the tick that triggered it (no look-ahead);
//  * price: buys pay the ask, sells hit the bid, plus `slippage_bps` and a
//    linear temporary impact of `impact_bps_per_lot` per lot in the fill;
//  * size: with `respect_touch_size`, a fill takes at most the size displayed
//    at the touch; the rest stays working and fills on later ticks (partial fills);
//  * fees per share on every fill.
struct ExecParams {
  std::int64_t lot_size = 100;       // shares per unit of target position
  double fee_per_share = 0.0005;     // commission + exchange fee, $/share
  double slippage_bps = 0.0;         // extra cost on every fill, against you
  double impact_bps_per_lot = 0.1;   // temporary impact, bps per lot filled
  std::int64_t latency_ns = 0;       // decision-to-eligible delay (0 = next tick)
  bool respect_touch_size = true;    // partial fills limited by displayed size
};

struct BacktestParams {
  StrategyParams strategy;
  ExecParams exec;
  std::uint32_t sharpe_bucket = 1000;  // ticks per P&L sample for the Sharpe estimate
};

// Seconds in a trading year (252 sessions of 6.5 hours), for annualising.
inline constexpr double kTradingSecondsPerYear = 252.0 * 6.5 * 3600.0;

struct BacktestResult {
  double pnl = 0.0;              // final equity marked at mid, net of all costs, $
  double fees = 0.0;             // $
  double exec_cost = 0.0;        // $ paid versus mid: half-spread + impact + slippage
  double sharpe = 0.0;           // mean/std of per-bucket P&L
  double sharpe_annual = 0.0;    // sharpe * sqrt(buckets per trading year), from the data's timestamps
  double max_drawdown = 0.0;     // $, peak-to-trough on tick-level equity
  std::uint64_t trades = 0;      // fills (a partially filled order counts once per fill)
  std::uint64_t partial_fills = 0;
  std::uint64_t shares = 0;      // absolute shares traded
  std::int64_t final_position = 0;
  std::uint64_t ticks = 0;

  bool operator==(const BacktestResult&) const = default;
};

// Bytes of arena scratch one backtest needs; used to size Arenas up front.
inline std::size_t backtest_scratch_bytes(const BacktestParams& p) noexcept {
  const auto w = static_cast<std::size_t>(std::max(p.strategy.window, 1));
  return 3 * w * sizeof(double) + 4 * kCacheLine;  // vwap(2) + ofi(1) rings + headroom
}

template <class Strategy>
class BasicBacktester {
 public:
  BasicBacktester(const BacktestParams& p, Strategy strategy, std::pmr::memory_resource* mr)
      : p_(p),
        features_(static_cast<std::size_t>(std::max(p.strategy.window, 1)), p.strategy.ewma_lambda, mr),
        strategy_(std::move(strategy)) {}

  // Hot path. Optionally writes this tick's equity to *equity_out.
  void on_tick(const Tick& t, double* equity_out = nullptr) noexcept {
    if (r_.ticks == 0) first_ts_ = t.ts;
    // 1. Work the outstanding order against this tick's quote.
    if (pending_ != 0 && t.ts >= eligible_ts_) fill(t);
    // 2. Features and mark-to-market at mid.
    const Features f = features_.update(t);
    const double mid = t.mid();
    const double equity = cash_ + static_cast<double>(position_) * mid;
    if (equity_out != nullptr) *equity_out = equity;
    peak_ = std::max(peak_, equity);
    r_.max_drawdown = std::max(r_.max_drawdown, peak_ - equity);
    if (++in_bucket_ == p_.sharpe_bucket) {
      record_bucket(equity - bucket_start_);
      bucket_start_ = equity;
      in_bucket_ = 0;
    }
    last_mid_ = mid;
    last_ts_ = t.ts;
    ++r_.ticks;
    // 3. Ask the strategy for a target; re-route the order only if it changed.
    const int target = strategy_.on_tick(t, f, target_);
    if (target != target_) {
      target_ = target;
      pending_ = static_cast<std::int64_t>(target) * p_.exec.lot_size - position_;
      eligible_ts_ = t.ts + std::max<std::int64_t>(p_.exec.latency_ns, 1);
    }
  }

  BacktestResult finish() const noexcept {
    BacktestResult r = r_;
    r.pnl = cash_ + static_cast<double>(position_) * last_mid_;
    r.final_position = position_;
    if (n_buckets_ > 1) {
      const double var = m2_ / static_cast<double>(n_buckets_ - 1);
      r.sharpe = var > 0.0 ? mean_ / std::sqrt(var) : 0.0;
      const double span_s = static_cast<double>(last_ts_ - first_ts_) * 1e-9;
      if (span_s > 0.0) {
        const double buckets_per_year = static_cast<double>(n_buckets_) * kTradingSecondsPerYear / span_s;
        r.sharpe_annual = r.sharpe * std::sqrt(buckets_per_year);
      }
    }
    return r;
  }

 private:
  void fill(const Tick& t) noexcept {
    const bool buy = pending_ > 0;
    const std::int64_t want = buy ? pending_ : -pending_;
    std::int64_t qty = want;
    if (p_.exec.respect_touch_size) {
      qty = std::min<std::int64_t>(want, buy ? t.ask_size : t.bid_size);
      if (qty <= 0) return;  // nothing displayed; keep working
    }
    const double lots = static_cast<double>(qty) / static_cast<double>(p_.exec.lot_size);
    const double cost_bps = p_.exec.slippage_bps + p_.exec.impact_bps_per_lot * lots;
    const double touch = buy ? t.ask : t.bid;
    const double exec_px = buy ? touch * (1.0 + cost_bps * 1e-4) : touch * (1.0 - cost_bps * 1e-4);
    const double signed_qty = buy ? static_cast<double>(qty) : -static_cast<double>(qty);
    const double fee = p_.exec.fee_per_share * static_cast<double>(qty);
    cash_ -= signed_qty * exec_px + fee;
    position_ += buy ? qty : -qty;
    pending_ -= buy ? qty : -qty;
    r_.fees += fee;
    r_.exec_cost += std::abs(exec_px - t.mid()) * static_cast<double>(qty);
    r_.shares += static_cast<std::uint64_t>(qty);
    ++r_.trades;
    if (qty < want) ++r_.partial_fills;
  }

  void record_bucket(double pnl) noexcept {  // Welford, so no buffer needed
    ++n_buckets_;
    const double d = pnl - mean_;
    mean_ += d / static_cast<double>(n_buckets_);
    m2_ += d * (pnl - mean_);
  }

  BacktestParams p_;
  FeatureSet features_;
  Strategy strategy_;
  BacktestResult r_{};
  double cash_ = 0.0;
  std::int64_t position_ = 0;
  std::int64_t pending_ = 0;
  std::int64_t eligible_ts_ = 0;
  int target_ = 0;
  double last_mid_ = 0.0;
  std::int64_t first_ts_ = 0;
  std::int64_t last_ts_ = 0;
  double peak_ = 0.0;
  std::uint32_t in_bucket_ = 0;
  double bucket_start_ = 0.0;
  std::uint64_t n_buckets_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;
};

using Backtester = BasicBacktester<MeanReversion>;

// Runs a full backtest over `cols`. If `equity_out` is non-empty it must have
// cols.size_ticks() elements and receives the per-tick equity curve. If `mr`
// is null a right-sized Arena is created internally.
BacktestResult run_backtest(const TickColumns& cols, const BacktestParams& p,
                            std::pmr::memory_resource* mr = nullptr, std::span<double> equity_out = {});

// Same result, but ticks travel producer -> SpscRing -> consumer across two
// threads, as they would from a live feed handler.
BacktestResult run_backtest_pipelined(const TickColumns& cols, const BacktestParams& p);

}  // namespace tickforge
