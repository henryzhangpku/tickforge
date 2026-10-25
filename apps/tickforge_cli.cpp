// Command-line driver.
//
//   tickforge_cli gen <file> [n_ticks] [seed]   write a synthetic tick file
//   tickforge_cli backtest <file>               one backtest, default params
//   tickforge_cli pipeline <file>               same, through the SPSC ring
//   tickforge_cli sweep <file> [threads]        default grid on a thread pool
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

#include "tickforge/backtest.hpp"
#include "tickforge/sweep.hpp"
#include "tickforge/synth.hpp"
#include "tickforge/tick_store.hpp"

namespace tf = tickforge;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: tickforge_cli gen <file> [n_ticks] [seed]\n"
               "       tickforge_cli backtest|pipeline <file>\n"
               "       tickforge_cli sweep <file> [threads]\n");
  return 2;
}

void print(const tf::BacktestResult& r) {
  std::printf("pnl=%.2f fees=%.2f sharpe=%.4f max_dd=%.2f trades=%llu final_pos=%lld ticks=%llu\n", r.pnl, r.fees,
              r.sharpe, r.max_drawdown, static_cast<unsigned long long>(r.trades),
              static_cast<long long>(r.final_position), static_cast<unsigned long long>(r.ticks));
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3) return usage();
  const std::string_view cmd = argv[1];
  const std::string path = argv[2];

  if (cmd == "gen") {
    tf::SynthParams p;
    if (argc > 3) p.n_ticks = std::strtoull(argv[3], nullptr, 10);
    if (argc > 4) p.seed = std::strtoull(argv[4], nullptr, 10);
    const auto t = tf::generate_ticks(p);
    tf::write_tick_file(path, t.view(), p.seed);
    std::printf("wrote %llu ticks to %s\n", static_cast<unsigned long long>(p.n_ticks), path.c_str());
    return 0;
  }

  const tf::MappedTickFile file(path);
  const auto t0 = std::chrono::steady_clock::now();
  if (cmd == "backtest" || cmd == "pipeline") {
    const tf::BacktestParams p;
    const auto r = cmd == "backtest" ? tf::run_backtest(file.columns(), p) : tf::run_backtest_pipelined(file.columns(), p);
    const double s = seconds_since(t0);
    print(r);
    std::printf("%.3f s, %.1f M ticks/s\n", s, static_cast<double>(file.size()) / s / 1e6);
    return 0;
  }
  if (cmd == "sweep") {
    const std::size_t threads = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;
    const auto params = tf::expand(tf::SweepGrid{});
    const auto results = tf::run_sweep(file.columns(), params, threads);
    const double s = seconds_since(t0);
    for (std::size_t i = 0; i < params.size(); ++i) {
      const auto& sp = params[i].strategy;
      std::printf("window=%4d entry=%.2f exit=%.2f  ", sp.window, sp.entry_z, sp.exit_z);
      print(results[i]);
    }
    std::printf("%zu backtests in %.3f s\n", params.size(), s);
    return 0;
  }
  return usage();
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
