// WebAssembly entry points for the browser demo (docs/index.html).
//
// The same engine code as the native build, compiled with Emscripten and run
// inside a Web Worker. Results live in WASM linear memory; JavaScript reads
// them through typed-array views at the pointers returned here, the browser
// equivalent of the zero-copy NumPy views in python/bindings.cpp.
//
// Single-threaded on purpose: WASM threads need SharedArrayBuffer, which needs
// cross-origin-isolation headers that GitHub Pages cannot send.
#include <emscripten/emscripten.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory_resource>
#include <vector>

#include "tickforge/backtest.hpp"
#include "tickforge/memory.hpp"
#include "tickforge/synth.hpp"

namespace tf = tickforge;

namespace {

tf::TickVectors g_ticks;
std::vector<double> g_equity;
std::vector<std::int32_t> g_position;
std::vector<double> g_stats;
std::vector<double> g_sweep;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

tf::BacktestParams params(int window, double entry_z, double exit_z, double fee_per_share, double impact_bps_per_lot,
                          double latency_ms, int respect_touch_size) {
  tf::BacktestParams p;
  p.strategy.window = std::max(window, 2);
  p.strategy.entry_z = entry_z;
  p.strategy.exit_z = exit_z;
  p.exec.fee_per_share = fee_per_share;
  p.exec.impact_bps_per_lot = impact_bps_per_lot;
  p.exec.latency_ns = static_cast<std::int64_t>(latency_ms * 1e6);
  p.exec.respect_touch_size = respect_touch_size != 0;
  return p;
}

}  // namespace

extern "C" {

// Generates `n` synthetic ticks; returns the generation time in ms.
EMSCRIPTEN_KEEPALIVE double tf_generate(int n, int seed) {
  const double t0 = now_ms();
  tf::SynthParams p;
  p.n_ticks = static_cast<std::uint64_t>(std::max(n, 1));
  p.seed = static_cast<std::uint64_t>(seed);
  g_ticks = tf::generate_ticks(p);
  return now_ms() - t0;
}

EMSCRIPTEN_KEEPALIVE int tf_size() { return static_cast<int>(g_ticks.price.size()); }
EMSCRIPTEN_KEEPALIVE const double* tf_price() { return g_ticks.price.data(); }
EMSCRIPTEN_KEEPALIVE const double* tf_bid() { return g_ticks.bid.data(); }
EMSCRIPTEN_KEEPALIVE const double* tf_ask() { return g_ticks.ask.data(); }
EMSCRIPTEN_KEEPALIVE const std::int8_t* tf_side() { return g_ticks.side.data(); }
EMSCRIPTEN_KEEPALIVE double tf_span_seconds() {
  return g_ticks.ts.empty() ? 0.0 : static_cast<double>(g_ticks.ts.back() - g_ticks.ts.front()) * 1e-9;
}

// One backtest over the generated ticks. Fills per-tick equity and position
// arrays and a stats block; returns a pointer to the stats block:
// [pnl, fees, exec_cost, sharpe, sharpe_annual, max_drawdown, trades,
//  partial_fills, shares, final_position, elapsed_ms]
EMSCRIPTEN_KEEPALIVE const double* tf_backtest(int window, double entry_z, double exit_z, double fee_per_share,
                                              double impact_bps_per_lot, double latency_ms, int respect_touch_size) {
  const auto p = params(window, entry_z, exit_z, fee_per_share, impact_bps_per_lot, latency_ms, respect_touch_size);
  const auto cols = g_ticks.view();
  const std::size_t n = cols.size_ticks();
  g_equity.resize(n);
  g_position.resize(n);
  const double t0 = now_ms();
  tf::Arena arena(tf::backtest_scratch_bytes(p));
  tf::Backtester bt(p, tf::MeanReversion(p.strategy), arena.resource());
  for (std::size_t i = 0; i < n; ++i) {
    bt.on_tick(cols[i], &g_equity[i]);
    g_position[i] = static_cast<std::int32_t>(bt.position());
  }
  const auto r = bt.finish();
  const double elapsed = now_ms() - t0;
  g_stats = {r.pnl,
             r.fees,
             r.exec_cost,
             r.sharpe,
             r.sharpe_annual,
             r.max_drawdown,
             static_cast<double>(r.trades),
             static_cast<double>(r.partial_fills),
             static_cast<double>(r.shares),
             static_cast<double>(r.final_position),
             elapsed};
  return g_stats.data();
}

EMSCRIPTEN_KEEPALIVE const double* tf_equity() { return g_equity.data(); }
EMSCRIPTEN_KEEPALIVE const std::int32_t* tf_position() { return g_position.data(); }

// One grid point of a sweep (the worker loops so it can report progress).
// Returns a pointer to [pnl, sharpe_annual, trades, exec_cost, max_drawdown, elapsed_ms].
EMSCRIPTEN_KEEPALIVE const double* tf_sweep_point(int window, double entry_z, double exit_z, double fee_per_share,
                                                 double impact_bps_per_lot, double latency_ms, int respect_touch_size) {
  const auto p = params(window, entry_z, exit_z, fee_per_share, impact_bps_per_lot, latency_ms, respect_touch_size);
  const double t0 = now_ms();
  const auto r = tf::run_backtest(g_ticks.view(), p);
  g_sweep = {r.pnl, r.sharpe_annual, static_cast<double>(r.trades), r.exec_cost, r.max_drawdown, now_ms() - t0};
  return g_sweep.data();
}

}  // extern "C"
