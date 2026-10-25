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

tf::ExecParams exec_params(double fee_per_share, double slippage_bps, std::int64_t lot_size) {
  tf::ExecParams e;
  e.fee_per_share = fee_per_share;
  e.slippage_bps = slippage_bps;
  e.lot_size = lot_size;
  return e;
}

py::dict result_dict(const tf::BacktestResult& r) {
  py::dict d;
  d["pnl"] = r.pnl;
  d["fees"] = r.fees;
  d["sharpe"] = r.sharpe;
  d["max_drawdown"] = r.max_drawdown;
  d["trades"] = r.trades;
  d["shares"] = r.shares;
  d["final_position"] = r.final_position;
  d["ticks"] = r.ticks;
  return d;
}

py::dict sweep(const TickStore& store, std::vector<int> windows, std::vector<double> entry_z,
               std::vector<double> exit_z, std::vector<double> ewma_lambda, std::vector<double> ofi_max,
               double fee_per_share, double slippage_bps, std::int64_t lot_size, std::size_t threads) {
  tf::SweepGrid g{std::move(windows), std::move(entry_z), std::move(exit_z), std::move(ewma_lambda),
                  std::move(ofi_max), exec_params(fee_per_share, slippage_bps, lot_size)};
  const auto params = tf::expand(g);
  std::vector<tf::BacktestResult> results;
  {
    py::gil_scoped_release nogil;
    results = tf::run_sweep(store.file.columns(), params, threads);
  }
  const std::size_t n = params.size();
  std::vector<std::int32_t> w(n);
  std::vector<double> ez(n), xz(n), lam(n), ofi(n), pnl(n), fees(n), sharpe(n), dd(n);
  std::vector<std::uint64_t> trades(n);
  std::vector<std::int64_t> pos(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& sp = params[i].strategy;
    w[i] = sp.window;
    ez[i] = sp.entry_z;
    xz[i] = sp.exit_z;
    lam[i] = sp.ewma_lambda;
    ofi[i] = sp.ofi_max;
    pnl[i] = results[i].pnl;
    fees[i] = results[i].fees;
    sharpe[i] = results[i].sharpe;
    dd[i] = results[i].max_drawdown;
    trades[i] = results[i].trades;
    pos[i] = results[i].final_position;
  }
  py::dict d;
  d["window"] = to_numpy(std::move(w));
  d["entry_z"] = to_numpy(std::move(ez));
  d["exit_z"] = to_numpy(std::move(xz));
  d["ewma_lambda"] = to_numpy(std::move(lam));
  d["ofi_max"] = to_numpy(std::move(ofi));
  d["pnl"] = to_numpy(std::move(pnl));
  d["fees"] = to_numpy(std::move(fees));
  d["sharpe"] = to_numpy(std::move(sharpe));
  d["max_drawdown"] = to_numpy(std::move(dd));
  d["trades"] = to_numpy(std::move(trades));
  d["final_position"] = to_numpy(std::move(pos));
  return d;
}

py::tuple backtest(const TickStore& store, int window, double entry_z, double exit_z, double ewma_lambda,
                   double ofi_max, double fee_per_share, double slippage_bps, std::int64_t lot_size) {
  tf::BacktestParams p;
  p.strategy = {window, ewma_lambda, entry_z, exit_z, ofi_max};
  p.exec = exec_params(fee_per_share, slippage_bps, lot_size);
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
      [](const std::string& path, std::uint64_t n_ticks, std::uint64_t seed) {
        tf::SynthParams p;
        p.n_ticks = n_ticks;
        p.seed = seed;
        py::gil_scoped_release nogil;
        const auto t = tf::generate_ticks(p);
        tf::write_tick_file(path, t.view(), seed);
      },
      py::arg("path"), py::arg("n_ticks") = 1'000'000, py::arg("seed") = 42,
      "Write a deterministic synthetic tick file.");

  py::class_<TickStore>(m, "TickStore", "A memory-mapped tick file. Column properties are zero-copy NumPy views.")
      .def(py::init<const std::string&>(), py::arg("path"))
      .def("__len__", [](const TickStore& s) { return s.file.size(); })
      .def("__repr__", [](const TickStore& s) {
        return "<TickStore '" + s.path + "' ticks=" + std::to_string(s.file.size()) + ">";
      })
      .def_property_readonly("ts", [](py::object self) { return view(self.cast<TickStore&>().file.columns().ts, self); })
      .def_property_readonly("price", [](py::object self) { return view(self.cast<TickStore&>().file.columns().price, self); })
      .def_property_readonly("size", [](py::object self) { return view(self.cast<TickStore&>().file.columns().size, self); })
      .def_property_readonly("side", [](py::object self) { return view(self.cast<TickStore&>().file.columns().side, self); })
      .def_property_readonly("seed", [](const TickStore& s) { return s.file.header().seed; });

  const tf::ExecParams e;
  const tf::StrategyParams sp;
  m.def("sweep", &sweep, py::arg("store"), py::arg("windows") = std::vector<int>{200, 500, 1000},
        py::arg("entry_z") = std::vector<double>{0.5, 1.0, 1.5}, py::arg("exit_z") = std::vector<double>{0.0, 0.25},
        py::arg("ewma_lambda") = std::vector<double>{sp.ewma_lambda}, py::arg("ofi_max") = std::vector<double>{1.0},
        py::arg("fee_per_share") = e.fee_per_share, py::arg("slippage_bps") = e.slippage_bps,
        py::arg("lot_size") = e.lot_size, py::arg("threads") = 0,
        "Run the cartesian product of the parameter lists on a thread pool. Returns a dict of NumPy arrays, "
        "one row per parameter combination (windows vary slowest).");

  m.def("backtest", &backtest, py::arg("store"), py::arg("window") = sp.window, py::arg("entry_z") = sp.entry_z,
        py::arg("exit_z") = sp.exit_z, py::arg("ewma_lambda") = sp.ewma_lambda, py::arg("ofi_max") = sp.ofi_max,
        py::arg("fee_per_share") = e.fee_per_share, py::arg("slippage_bps") = e.slippage_bps,
        py::arg("lot_size") = e.lot_size, "One backtest. Returns (stats dict, per-tick equity array).");

  m.def("rolling_features", &rolling_features, py::arg("prices"), py::arg("window") = 100,
        py::arg("ewma_lambda") = 0.995, py::arg("backend") = "auto",
        "Rolling mean and EWMA vol for a (n_ticks, n_symbols) price matrix. Returns (mean, vol, backend_used).");

  m.def("cuda_available", &tf::cuda_available);
}
