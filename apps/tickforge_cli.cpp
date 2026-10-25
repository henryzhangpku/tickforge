// Command-line driver.
//
//   tickforge_cli gen <file> [n_ticks] [seed]                 synthetic tick file
//   tickforge_cli gen-universe <dir> <n_symbols> [n_ticks] [seed]
//   tickforge_cli import <trades.csv> <file> [symbol]          bring your own data
//   tickforge_cli info <file>
//   tickforge_cli backtest <file>                              one backtest, default params
//   tickforge_cli pipeline <file>                              same, through the SPSC ring
//   tickforge_cli sweep <file|dir>... [options]                default grid
//       --threads N        worker threads (default: all cores)
//       --pool ws|central  work-stealing (default) or single shared queue
//       --shard I/N        run only tasks with index % N == I
//       --shard-from-env   read the shard from TICKFORGE_SHARD(S), JOB_COMPLETION_INDEX,
//                          CLOUD_RUN_TASK_INDEX/COUNT or AWS_BATCH_JOB_ARRAY_INDEX
//       --csv              machine-readable rows (one per task) instead of a table
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tickforge/backtest.hpp"
#include "tickforge/sweep.hpp"
#include "tickforge/synth.hpp"
#include "tickforge/tick_store.hpp"

namespace tf = tickforge;
namespace fs = std::filesystem;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: tickforge_cli gen <file> [n_ticks] [seed]\n"
               "       tickforge_cli gen-universe <dir> <n_symbols> [n_ticks] [seed]\n"
               "       tickforge_cli import <trades.csv> <file> [symbol]\n"
               "       tickforge_cli info|backtest|pipeline <file>\n"
               "       tickforge_cli sweep <file|dir>... [--threads N] [--pool ws|central]\n"
               "                           [--shard I/N | --shard-from-env] [--csv]\n");
  return 2;
}

std::uint64_t to_u64(const char* s) {
  char* end = nullptr;
  const auto v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') throw std::invalid_argument(std::string("not a number: ") + s);
  return v;
}

void print(const tf::BacktestResult& r) {
  std::printf("pnl=%.2f fees=%.2f exec_cost=%.2f sharpe=%.4f sharpe_ann=%.2f max_dd=%.2f trades=%llu partial=%llu "
              "final_pos=%lld\n",
              r.pnl, r.fees, r.exec_cost, r.sharpe, r.sharpe_annual, r.max_drawdown,
              static_cast<unsigned long long>(r.trades), static_cast<unsigned long long>(r.partial_fills),
              static_cast<long long>(r.final_position));
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Expands directories to their *.tf files, sorted, so every shard of a
// distributed sweep sees the same universe in the same order.
std::vector<fs::path> collect(const std::vector<std::string>& args) {
  std::vector<fs::path> out;
  for (const auto& a : args) {
    if (fs::is_directory(a)) {
      std::vector<fs::path> files;
      for (const auto& e : fs::directory_iterator(a))
        if (e.path().extension() == ".tf") files.push_back(e.path());
      std::sort(files.begin(), files.end());
      out.insert(out.end(), files.begin(), files.end());
    } else {
      out.emplace_back(a);
    }
  }
  if (out.empty()) throw std::invalid_argument("no tick files given");
  return out;
}

int sweep(int argc, char** argv) {
  std::vector<std::string> inputs;
  tf::SweepOptions opt;
  bool csv = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view a = argv[i];
    const auto next = [&]() -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument(std::string(a) + " needs a value");
      return argv[++i];
    };
    if (a == "--threads") {
      opt.threads = to_u64(next());
    } else if (a == "--pool") {
      const std::string_view v = next();
      if (v == "ws") opt.pool = tf::PoolKind::WorkStealing;
      else if (v == "central") opt.pool = tf::PoolKind::Central;
      else throw std::invalid_argument("--pool must be ws or central");
    } else if (a == "--shard") {
      const std::string v = next();
      const auto slash = v.find('/');
      if (slash == std::string::npos) throw std::invalid_argument("--shard wants I/N");
      opt.shard = {to_u64(v.substr(0, slash).c_str()), to_u64(v.substr(slash + 1).c_str())};
    } else if (a == "--shard-from-env") {
      opt.shard = tf::shard_from_env();
    } else if (a == "--csv") {
      csv = true;
    } else if (a.starts_with("--")) {
      throw std::invalid_argument("unknown option " + std::string(a));
    } else {
      inputs.emplace_back(a);
    }
  }

  const auto paths = collect(inputs);
  std::vector<tf::MappedTickFile> files;
  std::vector<tf::TickColumns> universe;
  files.reserve(paths.size());
  universe.reserve(paths.size());
  for (const auto& p : paths) universe.push_back(files.emplace_back(p).columns());
  const auto params = tf::expand(tf::SweepGrid{});

  const auto t0 = std::chrono::steady_clock::now();
  const auto rows = tf::run_sweep(universe, params, opt);
  const double s = seconds_since(t0);

  if (csv) {
    std::printf("index,symbol,window,entry_z,exit_z,ewma_lambda,ofi_max,pnl,fees,exec_cost,sharpe,sharpe_annual,"
                "max_drawdown,trades,partial_fills,final_position\n");
    for (const auto& r : rows) {
      const auto& sp = params[r.param].strategy;
      const auto& b = r.result;
      // %.17g round-trips doubles exactly, so merged shards compare byte-for-byte.
      std::printf("%zu,%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%llu,%llu,%lld\n", r.index,
                  files[r.symbol].symbol().c_str(), sp.window, sp.entry_z, sp.exit_z, sp.ewma_lambda, sp.ofi_max,
                  b.pnl, b.fees, b.exec_cost, b.sharpe, b.sharpe_annual, b.max_drawdown,
                  static_cast<unsigned long long>(b.trades), static_cast<unsigned long long>(b.partial_fills),
                  static_cast<long long>(b.final_position));
    }
    std::fprintf(stderr, "shard %zu/%zu: %zu backtests in %.3f s\n", opt.shard.index, opt.shard.count, rows.size(), s);
    return 0;
  }
  for (const auto& r : rows) {
    const auto& sp = params[r.param].strategy;
    std::printf("%-8s window=%4d entry=%.2f exit=%.2f  ", files[r.symbol].symbol().c_str(), sp.window, sp.entry_z,
                sp.exit_z);
    print(r.result);
  }
  std::printf("%zu backtests over %zu symbol(s) in %.3f s\n", rows.size(), files.size(), s);
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 3) return usage();
  const std::string_view cmd = argv[1];

  if (cmd == "gen") {
    tf::SynthParams p;
    if (argc > 3) p.n_ticks = to_u64(argv[3]);
    if (argc > 4) p.seed = to_u64(argv[4]);
    const auto t = tf::generate_ticks(p);
    tf::write_tick_file(argv[2], t.view(), p.seed, "SYN");
    std::printf("wrote %llu ticks to %s\n", static_cast<unsigned long long>(p.n_ticks), argv[2]);
    return 0;
  }
  if (cmd == "gen-universe") {
    if (argc < 4) return usage();
    tf::SynthParams p;
    if (argc > 4) p.n_ticks = to_u64(argv[4]);
    if (argc > 5) p.seed = to_u64(argv[5]);
    const auto paths = tf::generate_universe(argv[2], to_u64(argv[3]), p);
    std::printf("wrote %zu files of %llu ticks to %s\n", paths.size(), static_cast<unsigned long long>(p.n_ticks),
                argv[2]);
    return 0;
  }
  if (cmd == "import") {
    if (argc < 4) return usage();
    const auto t = tf::read_tick_csv(argv[2]);
    tf::write_tick_file(argv[3], t.view(), 0, argc > 4 ? argv[4] : "");
    std::printf("imported %zu ticks from %s into %s\n", t.price.size(), argv[2], argv[3]);
    return 0;
  }
  if (cmd == "sweep") return sweep(argc, argv);

  const tf::MappedTickFile file(argv[2]);
  if (cmd == "info") {
    const auto& c = file.columns();
    std::printf("symbol=%s ticks=%zu seed=%llu", file.symbol().c_str(), file.size(),
                static_cast<unsigned long long>(file.header().seed));
    if (file.size() > 0)
      std::printf(" span=%.1f s first=%.4f last=%.4f", static_cast<double>(c.ts.back() - c.ts.front()) * 1e-9,
                  c.price.front(), c.price.back());
    std::printf("\n");
    return 0;
  }
  if (cmd == "backtest" || cmd == "pipeline") {
    const auto t0 = std::chrono::steady_clock::now();
    const tf::BacktestParams p;
    const auto r = cmd == "backtest" ? tf::run_backtest(file.columns(), p) : tf::run_backtest_pipelined(file.columns(), p);
    const double s = seconds_since(t0);
    print(r);
    std::printf("%.3f s, %.1f M ticks/s\n", s, static_cast<double>(file.size()) / s / 1e6);
    return 0;
  }
  return usage();
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
