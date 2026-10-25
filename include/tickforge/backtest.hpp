// Event-driven backtest: one tick in, features updated, strategy asked for a
// target, orders filled on the *next* tick (no look-ahead), P&L marked.
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

struct ExecParams {
  std::int64_t lot_size = 100;     // shares per unit of target position
  double fee_per_share = 0.0005;   // commission + exchange fee, $/share
  double slippage_bps = 0.5;       // paid on every fill, against you
};

struct BacktestParams {
  StrategyParams strategy;
  ExecParams exec;
  std::uint32_t sharpe_bucket = 1000;  // ticks per P&L sample for the Sharpe estimate
};

struct BacktestResult {
  double pnl = 0.0;            // final equity, net of fees and slippage, $
  double fees = 0.0;           // $
  double sharpe = 0.0;         // mean/std of per-bucket P&L; not annualised
  double max_drawdown = 0.0;   // $, peak-to-trough on tick-level equity
  std::uint64_t trades = 0;    // fills
  std::uint64_t shares = 0;    // absolute shares traded
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
    // 1. Fill the order decided on the previous tick at this tick's price.
    if (pending_ != 0) fill(t.price);
    // 2. Features and mark-to-market.
    const Features f = features_.update(t);
    const double equity = cash_ + static_cast<double>(position_) * t.price;
    if (equity_out != nullptr) *equity_out = equity;
    peak_ = std::max(peak_, equity);
    r_.max_drawdown = std::max(r_.max_drawdown, peak_ - equity);
    if (++in_bucket_ == p_.sharpe_bucket) {
      record_bucket(equity - bucket_start_);
      bucket_start_ = equity;
      in_bucket_ = 0;
    }
    last_price_ = t.price;
    ++r_.ticks;
    // 3. Ask the strategy for a target; trade the difference next tick.
    const int current = static_cast<int>(position_ / p_.exec.lot_size);
    const int target = strategy_.on_tick(t, f, current);
    pending_ = static_cast<std::int64_t>(target) * p_.exec.lot_size - position_;
  }

  BacktestResult finish() const noexcept {
    BacktestResult r = r_;
    r.pnl = cash_ + static_cast<double>(position_) * last_price_;
    r.final_position = position_;
    if (n_buckets_ > 1) {
      const double var = m2_ / static_cast<double>(n_buckets_ - 1);
      r.sharpe = var > 0.0 ? mean_ / std::sqrt(var) : 0.0;
    }
    return r;
  }

 private:
  void fill(double px) noexcept {
    const double slip = p_.exec.slippage_bps * 1e-4;
    const double exec_px = pending_ > 0 ? px * (1.0 + slip) : px * (1.0 - slip);
    const auto qty = static_cast<double>(pending_);
    const auto abs_qty = static_cast<std::uint64_t>(std::llabs(pending_));
    const double fee = p_.exec.fee_per_share * static_cast<double>(abs_qty);
    cash_ -= qty * exec_px + fee;
    position_ += pending_;
    r_.fees += fee;
    r_.shares += abs_qty;
    ++r_.trades;
    pending_ = 0;
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
  double last_price_ = 0.0;
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
