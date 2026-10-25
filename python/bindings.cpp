// pybind11 bindings: `import tickforge`.
//
// Zero-copy in both directions:
//  * TickStore columns are NumPy arrays whose data pointer is the mmap itself;
//    the array's base object keeps the TickStore (and so the mapping) alive.
//    They are marked read-only because the pages are mapped PROT_READ.
//  * Sweep/backtest results are built in std::vectors, then handed to NumPy
//    with a capsule that owns the vector: NumPy frees it, nothing is copied.
//  * rolling_features takes any C-contiguous float64 array without copying.
//
// The GIL is released while C++ works, so Python threads keep running.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tickforge/backtest.hpp"
#include "tickforge/batch_features.hpp"
#include "tickforge/sweep.hpp"
#include "tickforge/synth.hpp"
#include "tickforge/tick_store.hpp"

namespace py = pybind11;
namespace tf = tickforge;

namespace {

// Moves a vector to the heap and lets NumPy own it via a capsule.
template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v) {
  auto* heap = new std::vector<T>(std::move(v));
  py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<T>*>(p); });
  return py::array_t<T>(static_cast<py::ssize_t>(heap->size()), heap->data(), owner);
}

// A read-only NumPy view into memory owned by `base`.
template <class T>
py::array_t<T> view(std::span<const T> s, py::handle base) {
  py::array_t<T> a(static_cast<py::ssize_t>(s.size()), s.data(), base);
  a.attr("setflags")(py::arg("write") = false);
  return a;
}

struct TickStore {
  explicit TickStore(const std::string& path) : file(path), path(path) {}
  tf::MappedTickFile file;
  std::string path;
};

tf::ExecParams exec_params(const py::kwargs& kw) {
  tf::ExecParams e;
  for (const auto& item : kw) {
    const auto key = item.first.cast<std::string>();
    if (key == "fee_per_share") e.fee_per_share = item.second.cast<double>();
    else if (key == "slippage_bps") e.slippage_bps = item.second.cast<double>();
    else if (key == "impact_bps_per_lot") e.impact_bps_per_lot = item.second.cast<double>();
    else if (key == "latency_ns") e.latency_ns = item.second.cast<std::int64_t>();
    else if (key == "lot_size") e.lot_size = item.second.cast<std::int64_t>();
    else if (key == "respect_touch_size") e.respect_touch_size = item.second.cast<bool>();
    else throw py::type_error("unexpected keyword argument '" + key + "'");
  }
  return e;
}

tf::PoolKind pool_kind(const std::string& pool) {
  if (pool == "work_stealing") return tf::PoolKind::WorkStealing;
  if (pool == "central") return tf::PoolKind::Central;
  throw std::invalid_argument("pool must be 'work_stealing' or 'central'");
}

py::dict result_dict(const tf::BacktestResult& r) {
  py::dict d;
  d["pnl"] = r.pnl;
  d["fees"] = r.fees;
  d["exec_cost"] = r.exec_cost;
  d["sharpe"] = r.sharpe;
  d["sharpe_annual"] = r.sharpe_annual;
  d["max_drawdown"] = r.max_drawdown;
  d["trades"] = r.trades;
  d["partial_fills"] = r.partial_fills;
  d["shares"] = r.shares;
  d["final_position"] = r.final_position;
  d["ticks"] = r.ticks;
  return d;
}

// `stores` is a TickStore or a list of them (a universe). Rows are
// symbol-major: row = symbol * n_params + param.
py::dict sweep(const py::object& stores, std::vector<int> windows, std::vector<double> entry_z,
               std::vector<double> exit_z, std::vector<double> ewma_lambda, std::vector<double> ofi_max,
               std::size_t threads, const std::string& pool, const py::kwargs& exec_kw) {
  std::vector<const TickStore*> list;
  if (py::isinstance<TickStore>(stores)) {
    list.push_back(&stores.cast<const TickStore&>());
  } else {
    for (const auto& h : stores) list.push_back(&h.cast<const TickStore&>());
  }
  std::vector<tf::TickColumns> universe;
  universe.reserve(list.size());
  for (const auto* st : list) universe.push_back(st->file.columns());

  tf::SweepGrid g{std::move(windows), std::move(entry_z), std::move(exit_z), std::move(ewma_lambda),
                  std::move(ofi_max), exec_params(exec_kw)};
  const auto params = tf::expand(g);
  std::vector<tf::SweepRow> rows;
  {
    py::gil_scoped_release nogil;
    rows = tf::run_sweep(universe, params, {threads, pool_kind(pool), {}});
  }
  const std::size_t n = rows.size();
  std::vector<std::int32_t> sym(n), w(n);
  std::vector<double> ez(n), xz(n), lam(n), ofi(n), pnl(n), fees(n), cost(n), sharpe(n), sharpe_ann(n), dd(n);
  std::vector<std::uint64_t> trades(n), partial(n);
  std::vector<std::int64_t> pos(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& sp = params[rows[i].param].strategy;
    const auto& r = rows[i].result;
    sym[i] = static_cast<std::int32_t>(rows[i].symbol);
    w[i] = sp.window;
    ez[i] = sp.entry_z;
    xz[i] = sp.exit_z;
    lam[i] = sp.ewma_lambda;
    ofi[i] = sp.ofi_max;
    pnl[i] = r.pnl;
    fees[i] = r.fees;
    cost[i] = r.exec_cost;
    sharpe[i] = r.sharpe;
    sharpe_ann[i] = r.sharpe_annual;
    dd[i] = r.max_drawdown;
    trades[i] = r.trades;
    partial[i] = r.partial_fills;
    pos[i] = r.final_position;
  }
  py::dict d;
  d["symbol"] = to_numpy(std::move(sym));
  d["window"] = to_numpy(std::move(w));
  d["entry_z"] = to_numpy(std::move(ez));
  d["exit_z"] = to_numpy(std::move(xz));
  d["ewma_lambda"] = to_numpy(std::move(lam));
  d["ofi_max"] = to_numpy(std::move(ofi));
  d["pnl"] = to_numpy(std::move(pnl));
  d["fees"] = to_numpy(std::move(fees));
  d["exec_cost"] = to_numpy(std::move(cost));
  d["sharpe"] = to_numpy(std::move(sharpe));
  d["sharpe_annual"] = to_numpy(std::move(sharpe_ann));
  d["max_drawdown"] = to_numpy(std::move(dd));
  d["trades"] = to_numpy(std::move(trades));
  d["partial_fills"] = to_numpy(std::move(partial));
  d["final_position"] = to_numpy(std::move(pos));
  return d;
}

py::tuple backtest(const TickStore& store, int window, double entry_z, double exit_z, double ewma_lambda,
                   double ofi_max, const py::kwargs& exec_kw) {
  tf::BacktestParams p;
  p.strategy = {window, ewma_lambda, entry_z, exit_z, ofi_max};
  p.exec = exec_params(exec_kw);
  std::vector<double> equity(store.file.size());
  tf::BacktestResult r;
  {
    py::gil_scoped_release nogil;
    r = tf::run_backtest(store.file.columns(), p, nullptr, equity);
  }
  return py::make_tuple(result_dict(r), to_numpy(std::move(equity)));
}

py::tuple rolling_features(const py::array_t<double, py::array::c_style | py::array::forcecast>& prices,
                           std::size_t window, double ewma_lambda, const std::string& backend) {
  if (prices.ndim() != 2) throw std::invalid_argument("prices must be 2-D (n_ticks, n_symbols)");
  const auto T = prices.shape(0);
  const auto S = prices.shape(1);
  py::array_t<double> mean({T, S});
  py::array_t<double> vol({T, S});
  const auto n = static_cast<std::size_t>(T * S);
  tf::Backend want = tf::Backend::Auto;
  if (backend == "cpu") want = tf::Backend::Cpu;
  else if (backend == "cuda") want = tf::Backend::Cuda;
  else if (backend != "auto") throw std::invalid_argument("backend must be 'auto', 'cpu' or 'cuda'");
  tf::Backend used;
  {
    py::gil_scoped_release nogil;
    used = tf::rolling_features_batch({prices.data(), n}, static_cast<std::size_t>(S), {window, ewma_lambda},
                                      {mean.mutable_data(), n}, {vol.mutable_data(), n}, want);
  }
  return py::make_tuple(mean, vol, used == tf::Backend::Cuda ? "cuda" : "cpu");
}

}  // namespace

PYBIND11_MODULE(tickforge, m) {
  m.doc() = "tickforge: C++20 tick research engine (learning project, synthetic data)";

  m.def(
      "generate",
      [](const std::string& path, std::uint64_t n_ticks, std::uint64_t seed, const std::string& symbol) {
        tf::SynthParams p;
        p.n_ticks = n_ticks;
        p.seed = seed;
        py::gil_scoped_release nogil;
        const auto t = tf::generate_ticks(p);
        tf::write_tick_file(path, t.view(), seed, symbol);
      },
      py::arg("path"), py::arg("n_ticks") = 1'000'000, py::arg("seed") = 42, py::arg("symbol") = "SYN",
      "Write a deterministic synthetic tick file (trades plus the quote in force).");

  m.def(
      "generate_universe",
      [](const std::string& dir, std::size_t n_symbols, std::uint64_t n_ticks, std::uint64_t seed) {
        tf::SynthParams p;
        p.n_ticks = n_ticks;
        p.seed = seed;
        std::vector<std::filesystem::path> paths;
        {
          py::gil_scoped_release nogil;
          paths = tf::generate_universe(dir, n_symbols, p);
        }
        std::vector<std::string> out;
        for (const auto& x : paths) out.push_back(x.string());
        return out;
      },
      py::arg("dir"), py::arg("n_symbols"), py::arg("n_ticks") = 1'000'000, py::arg("seed") = 42,
      "Write one synthetic file per symbol into `dir`; returns the paths.");

  m.def(
      "import_csv",
      [](const std::string& csv, const std::string& path, const std::string& symbol) {
        py::gil_scoped_release nogil;
        const auto t = tf::read_tick_csv(csv);
        tf::write_tick_file(path, t.view(), 0, symbol);
        return t.price.size();
      },
      py::arg("csv"), py::arg("path"), py::arg("symbol") = "",
      "Convert a CSV (ts,price,size,side[,bid,ask,bid_size,ask_size]) to a tick file; returns the tick count.");

  py::class_<TickStore>(m, "TickStore", "A memory-mapped tick file. Column properties are zero-copy NumPy views.")
      .def(py::init<const std::string&>(), py::arg("path"))
      .def("__len__", [](const TickStore& s) { return s.file.size(); })
      .def("__repr__", [](const TickStore& s) {
        return "<TickStore " + s.file.symbol() + " '" + s.path + "' ticks=" + std::to_string(s.file.size()) + ">";
      })
      .def_property_readonly("ts", [](py::object self) { return view(self.cast<TickStore&>().file.columns().ts, self); })
      .def_property_readonly("price", [](py::object self) { return view(self.cast<TickStore&>().file.columns().price, self); })
      .def_property_readonly("size", [](py::object self) { return view(self.cast<TickStore&>().file.columns().size, self); })
      .def_property_readonly("side", [](py::object self) { return view(self.cast<TickStore&>().file.columns().side, self); })
      .def_property_readonly("bid", [](py::object self) { return view(self.cast<TickStore&>().file.columns().bid, self); })
      .def_property_readonly("ask", [](py::object self) { return view(self.cast<TickStore&>().file.columns().ask, self); })
      .def_property_readonly("bid_size", [](py::object self) { return view(self.cast<TickStore&>().file.columns().bid_size, self); })
      .def_property_readonly("ask_size", [](py::object self) { return view(self.cast<TickStore&>().file.columns().ask_size, self); })
      .def_property_readonly("symbol", [](const TickStore& s) { return s.file.symbol(); })
      .def_property_readonly("seed", [](const TickStore& s) { return s.file.header().seed; });

  const tf::StrategyParams sp;
  constexpr const char* kExecDoc =
      "Execution keywords (all optional): fee_per_share, slippage_bps, impact_bps_per_lot, latency_ns, lot_size, "
      "respect_touch_size.";
  m.def("sweep", &sweep, py::arg("stores"), py::arg("windows") = std::vector<int>{200, 500, 1000},
        py::arg("entry_z") = std::vector<double>{0.5, 1.0, 1.5}, py::arg("exit_z") = std::vector<double>{0.0, 0.25},
        py::arg("ewma_lambda") = std::vector<double>{sp.ewma_lambda}, py::arg("ofi_max") = std::vector<double>{1.0},
        py::arg("threads") = 0, py::arg("pool") = "work_stealing",
        (std::string("Run the cartesian product of the parameter lists for one TickStore or a list of them, on a "
                     "thread pool. Returns a dict of NumPy arrays, one row per (symbol, parameter set), "
                     "symbol-major with windows varying slowest. ") +
         kExecDoc)
            .c_str());

  m.def("backtest", &backtest, py::arg("store"), py::arg("window") = sp.window, py::arg("entry_z") = sp.entry_z,
        py::arg("exit_z") = sp.exit_z, py::arg("ewma_lambda") = sp.ewma_lambda, py::arg("ofi_max") = sp.ofi_max,
        (std::string("One backtest. Returns (stats dict, per-tick equity array). ") + kExecDoc).c_str());

  m.def("rolling_features", &rolling_features, py::arg("prices"), py::arg("window") = 100,
        py::arg("ewma_lambda") = 0.995, py::arg("backend") = "auto",
        "Rolling mean and EWMA vol for a (n_ticks, n_symbols) price matrix. Returns (mean, vol, backend_used).");

  m.def("cuda_available", &tf::cuda_available);
}
