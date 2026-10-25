// A deliberately simple mean-reversion toy. It exists to give the engine
// something realistic-shaped to run, not to make money.
//
// Signal: z = log(mid / vwap) / (vol * sqrt(window)), i.e. the deviation from
// the rolling VWAP in units of the volatility you would expect over the window.
// Go short when z > entry_z, long when z < -entry_z, flat when |z| < exit_z.
// Order-flow filter: skip a new entry if trade flow is strongly against it
// (|ofi| > ofi_max in the direction of the move).
#pragma once

#include <cmath>

#include "tickforge/features.hpp"

namespace tickforge {

struct StrategyParams {
  int window = 500;
  double ewma_lambda = 0.995;
  double entry_z = 1.0;
  double exit_z = 0.25;
  double ofi_max = 1.0;  // 1.0 disables the filter
};

template <class Derived>
class StrategyBase {
 public:
  // Returns the desired position in lots: -1, 0 or +1.
  int on_tick(const Tick& t, const Features& f, int current) noexcept {
    return static_cast<Derived&>(*this).decide(t, f, current);
  }
};

class MeanReversion : public StrategyBase<MeanReversion> {
 public:
  explicit MeanReversion(const StrategyParams& p) noexcept
      : p_(p), scale_(std::sqrt(static_cast<double>(p.window))) {}

  int decide(const Tick& t, const Features& f, int current) const noexcept {
    if (!f.warm || f.vol <= 0.0 || f.vwap <= 0.0) return current;
    const double z = std::log(t.mid() / f.vwap) / (f.vol * scale_);
    if (current == 0) {
      const bool no_filter = p_.ofi_max >= 1.0;
      if (z > p_.entry_z && (no_filter || f.ofi < p_.ofi_max)) return -1;   // rich, buyers not stampeding
      if (z < -p_.entry_z && (no_filter || f.ofi > -p_.ofi_max)) return 1;  // cheap, sellers not stampeding
      return 0;
    }
    if (std::abs(z) < p_.exit_z) return 0;
    if (current > 0 && z > p_.entry_z) return -1;  // flip on the opposite extreme
    if (current < 0 && z < -p_.entry_z) return 1;
    return current;
  }

 private:
  StrategyParams p_;
  double scale_;
};

// ---- Runtime-polymorphic counterpart, for plugins --------------------------

class IStrategy {
 public:
  virtual ~IStrategy() = default;
  virtual int on_tick(const Tick& t, const Features& f, int current) noexcept = 0;
};

// Lets the templated backtester drive an IStrategy*; every call goes through
// the vtable. Used by the CRTP-vs-virtual benchmark.
class VirtualStrategyRef {
 public:
  explicit VirtualStrategyRef(IStrategy& s) noexcept : s_(&s) {}
  int on_tick(const Tick& t, const Features& f, int current) noexcept { return s_->on_tick(t, f, current); }

 private:
  IStrategy* s_;
};

class VirtualMeanReversion final : public IStrategy {
 public:
  explicit VirtualMeanReversion(const StrategyParams& p) noexcept : impl_(p) {}
  int on_tick(const Tick& t, const Features& f, int current) noexcept override {
    return impl_.on_tick(t, f, current);
  }

 private:
  MeanReversion impl_;
};

}  // namespace tickforge
