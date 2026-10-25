// Streaming features, O(1) per tick, no allocation after construction.
//
// Two dispatch styles live side by side on purpose:
//
//  * FeatureBase<Derived> (CRTP): the call resolves at compile time, so the
//    compiler can inline `update` into the backtest loop and keep state in
//    registers. This is what the hot path uses.
//  * IFeature (virtual): a runtime-pluggable interface for code that is
//    loaded or chosen at runtime. VirtualFeature<F> adapts any CRTP feature
//    to it. Each call is an indirect branch the optimiser cannot see through.
//
// bench/bench_main.cpp measures the gap; docs/index.html discusses it.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <utility>
#include <vector>

#include "tickforge/tick_store.hpp"

namespace tickforge {

template <class Derived>
class FeatureBase {
 public:
  double update(const Tick& t) noexcept { return self().on_tick(t); }
  double value() const noexcept { return self().current(); }

 private:
  Derived& self() noexcept { return static_cast<Derived&>(*this); }
  const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }
};

// Volume-weighted average price over the last `window` ticks.
class RollingVwap : public FeatureBase<RollingVwap> {
 public:
  RollingVwap(std::size_t window, std::pmr::memory_resource* mr)
      : pv_(window, 0.0, mr), vol_(window, 0.0, mr) {}

  double on_tick(const Tick& t) noexcept {
    const double v = static_cast<double>(t.size);
    const double pv = t.price * v;
    sum_pv_ += pv - pv_[idx_];
    sum_v_ += v - vol_[idx_];
    pv_[idx_] = pv;
    vol_[idx_] = v;
    if (++idx_ == pv_.size()) {
      idx_ = 0;
      // Running sums drift by rounding error; re-sum once per window
      // (amortised O(1)) so long backtests stay exact-ish.
      sum_pv_ = 0.0;
      sum_v_ = 0.0;
      for (std::size_t i = 0; i < pv_.size(); ++i) {
        sum_pv_ += pv_[i];
        sum_v_ += vol_[i];
      }
    }
    if (count_ < pv_.size()) ++count_;
    return current();
  }
  double current() const noexcept { return sum_v_ > 0.0 ? sum_pv_ / sum_v_ : 0.0; }
  bool warm() const noexcept { return count_ == pv_.size(); }

 private:
  std::pmr::vector<double> pv_;
  std::pmr::vector<double> vol_;
  std::size_t idx_ = 0;
  std::size_t count_ = 0;
  double sum_pv_ = 0.0;
  double sum_v_ = 0.0;
};

// Exponentially weighted volatility of tick log returns (RiskMetrics style).
class EwmaVol : public FeatureBase<EwmaVol> {
 public:
  explicit EwmaVol(double lambda) noexcept : lambda_(lambda) {}

  double on_tick(const Tick& t) noexcept {
    if (prev_ > 0.0) {
      const double r = std::log(t.price / prev_);
      var_ = lambda_ * var_ + (1.0 - lambda_) * r * r;
    }
    prev_ = t.price;
    return current();
  }
  double current() const noexcept { return std::sqrt(var_); }

 private:
  double lambda_;
  double var_ = 0.0;
  double prev_ = 0.0;
};

// Trade-sign order-flow imbalance over the last `window` ticks:
// (buyer-initiated volume - seller-initiated volume) / total volume, in [-1, 1].
// Integer sums, so no drift.
class OrderFlowImbalance : public FeatureBase<OrderFlowImbalance> {
 public:
  OrderFlowImbalance(std::size_t window, std::pmr::memory_resource* mr) : signed_(window, 0, mr) {}

  double on_tick(const Tick& t) noexcept {
    const std::int64_t s = static_cast<std::int64_t>(t.size) * t.side;
    const std::int64_t old = signed_[idx_];
    sum_signed_ += s - old;
    sum_abs_ += (s < 0 ? -s : s) - (old < 0 ? -old : old);
    signed_[idx_] = s;
    if (++idx_ == signed_.size()) idx_ = 0;
    return current();
  }
  double current() const noexcept {
    return sum_abs_ > 0 ? static_cast<double>(sum_signed_) / static_cast<double>(sum_abs_) : 0.0;
  }

 private:
  std::pmr::vector<std::int64_t> signed_;
  std::size_t idx_ = 0;
  std::int64_t sum_signed_ = 0;
  std::int64_t sum_abs_ = 0;
};

struct Features {
  double vwap = 0.0;
  double vol = 0.0;
  double ofi = 0.0;
  bool warm = false;
};

// The three features the toy strategy consumes, updated together.
class FeatureSet {
 public:
  FeatureSet(std::size_t window, double ewma_lambda, std::pmr::memory_resource* mr)
      : vwap_(window, mr), vol_(ewma_lambda), ofi_(window, mr) {}

  Features update(const Tick& t) noexcept {
    return {vwap_.update(t), vol_.update(t), ofi_.update(t), vwap_.warm()};
  }

 private:
  RollingVwap vwap_;
  EwmaVol vol_;
  OrderFlowImbalance ofi_;
};

// ---- Runtime-polymorphic counterpart, for plugins --------------------------

class IFeature {
 public:
  virtual ~IFeature() = default;
  virtual double update(const Tick& t) noexcept = 0;
};

template <class F>
class VirtualFeature final : public IFeature {
 public:
  template <class... Args>
  explicit VirtualFeature(Args&&... args) : f_(std::forward<Args>(args)...) {}
  double update(const Tick& t) noexcept override { return f_.update(t); }

 private:
  F f_;
};

}  // namespace tickforge
