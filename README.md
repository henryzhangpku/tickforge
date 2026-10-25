<p align="center"><img src="docs/assets/logo.svg" width="88" alt="tickforge logo"></p>

<h1 align="center">tickforge</h1>

<p align="center">
<a href="https://github.com/henryzhangpku/tickforge/actions/workflows/ci.yml"><img src="https://github.com/henryzhangpku/tickforge/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://henryzhangpku.github.io/tickforge/"><img src="https://github.com/henryzhangpku/tickforge/actions/workflows/pages.yml/badge.svg" alt="Pages"></a>
<a href="https://colab.research.google.com/github/henryzhangpku/tickforge/blob/main/python/examples/gpu_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the GPU notebook in Colab"></a>
</p>

A C++20 research engine for tick data: a memory-mapped columnar store, arena memory, a lock-free pipeline, an event-driven backtest with spread-crossing fills, work-stealing parameter sweeps that shard across a cluster, zero-copy Python bindings, a CUDA kernel, and a WebAssembly build that runs in the browser.

**Live demo: <https://henryzhangpku.github.io/tickforge/>.** The engine compiled to WebAssembly backtests a million ticks on your CPU as you move the sliders, WebGL2 draws them, and a parameter sweep fills in a heatmap live.

> **Learning project.** All data is synthetic, generated from a seed (or bring your own via CSV). The strategy is a deliberately simple toy; its results on synthetic data say more about how the data was generated than about markets. Nothing here is investment advice.

```python
import tickforge

tickforge.generate("ticks.tf", n_ticks=2_000_000, seed=42)
store = tickforge.TickStore("ticks.tf")          # mmap; store.price, .bid, .ask are zero-copy NumPy views
res = tickforge.sweep(store,                      # all cores, GIL released
                      windows=[200, 500, 1000], entry_z=[0.5, 1.0, 1.5], exit_z=[0.0, 0.25],
                      latency_ns=5_000_000, impact_bps_per_lot=0.5)
best = res["sharpe"].argmax()                     # NumPy arrays, one row per grid point
```

## What's in it, and why

| Piece | Where | Concern it demonstrates |
|---|---|---|
| Columnar tick file (trades + quote in force), `mmap`, `std::span` views; CSV importer; per-symbol universes | `include/tickforge/tick_store.hpp`, `src/tick_store.cpp` | Hardware/software interface and data engineering: SoA layout, 64-byte aligned columns, page cache, `MADV_SEQUENTIAL`, zero-copy reads, validated ingest |
| Arena (`std::pmr::monotonic_buffer_resource`), aligned buffers, counting resource | `include/tickforge/memory.hpp`, `tests/test_no_alloc.cpp` | Memory management: no allocation on the hot path, *proven* by a test that replaces global `operator new` |
| Lock-free SPSC ring (acquire/release, cached indices, batched consume) | `include/tickforge/spsc_ring.hpp`, `src/pipeline.cpp` | Low-level concurrency: memory ordering, false sharing, cache-line traffic |
| Streaming features: rolling VWAP, EWMA volatility, order-flow imbalance | `include/tickforge/features.hpp` | O(1) per tick, no allocation, numerically re-anchored |
| CRTP strategy/feature dispatch vs virtual plugin interface and registry | `include/tickforge/strategy.hpp`, `include/tickforge/plugins.hpp` | Design patterns and the measured cost of each |
| Event-driven backtest: fills at the bid/ask after latency, partial fills at displayed size, linear impact, fees; P&L at mid; drawdown; Sharpe (per bucket and annualised) | `include/tickforge/backtest.hpp` | Research tooling that is correct by construction (no look-ahead) and honest about costs |
| Central-queue thread pool and work-stealing pool, futures; sweeps over symbols × parameters | `include/tickforge/thread_pool.hpp`, `include/tickforge/work_stealing_pool.hpp`, `src/sweep.cpp` | Parallel and async patterns; deterministic results at any thread count |
| Sharded sweeps: one image, shard index from Kubernetes, Cloud Run or AWS Batch | `deploy/`, `scripts/merge_shards.py` | Cloud and production: the same job verified on a real Kubernetes cluster in CI |
| pybind11 module | `python/bindings.cpp`, `python/examples/research_example.py` | C++ and Python together: buffer-protocol NumPy views, capsule-owned results, GIL release |
| CUDA rolling-feature kernel + CPU fallback | `src/batch_features.cu`, `src/batch_features.cpp`, `python/examples/gpu_colab.ipynb` | GPU computing: coalesced time-major layout, one thread per symbol |
| WebAssembly build + WebGL2 renderer | `web/wasm_api.cpp`, `docs/assets/` | The engine in a browser worker, results read straight out of WASM memory; GPU-drawn charts (OpenGL ES 3.0) |
| CMake presets; CI with gcc, clang, ASan/UBSan, TSan, clang-tidy, nvcc, kind, Docker, GHCR, benchmarks | `CMakePresets.json`, `.github/workflows/` | Production engineering: reproducible builds, sanitizers, static analysis, deployable artefacts |

The design notes (fill model, CRTP vs virtual, work stealing vs one queue, the SPSC memory-ordering argument, the CUDA mental model) are on the [project page](https://henryzhangpku.github.io/tickforge/#notes).

## Build and run

Requirements: a C++20 compiler (gcc 13 or clang 18 tested), CMake 3.25+, Ninja. GoogleTest, Google Benchmark and pybind11 are fetched by CMake.

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release

B=./build/release/tickforge_cli
$B gen ticks.tf 2000000 42                 # deterministic synthetic data (trades + quotes)
$B backtest ticks.tf                       # one backtest, default parameters
$B pipeline ticks.tf                       # same result through the SPSC ring, two threads
$B sweep ticks.tf                          # default grid on a work-stealing pool
$B gen-universe universe/ 8 1000000        # eight symbols
$B sweep universe/ --csv --shard 0/4       # one shard of a distributed sweep
$B import my_trades.csv mine.tf ABC        # bring your own data
./build/release/tickforge_bench            # microbenchmarks
```

CSV import expects a header with `ts,price,size,side` (nanosecond timestamps, side ±1) and optionally `bid,ask,bid_size,ask_size`, in any column order.

Python:

```sh
pip install numpy
cmake --preset python && cmake --build --preset python
PYTHONPATH=build/python python3 python/examples/research_example.py
PYTHONPATH=build/python python3 -m unittest discover python/tests
```

Other presets: `debug`, `asan` (AddressSanitizer + UBSan), `tsan` (ThreadSanitizer), `cuda` (needs `nvcc`). WebAssembly: `scripts/build_wasm.sh` with Emscripten on the path, then serve `docs/`.

### Containers and clusters

```sh
docker run --rm ghcr.io/henryzhangpku/tickforge:latest        # public image, published by CI
kubectl apply -f deploy/k8s/sweep-job.yaml                    # 4-way Indexed Job
for p in $(kubectl get pods -l app=tickforge -o name); do kubectl logs $p; done > shards.log
python3 scripts/merge_shards.py shards.log > sweep.csv        # byte-identical to a one-machine run
```

`deploy/cloudrun/job.yaml` and `deploy/aws-batch/job-definition.json` run the same image under Cloud Run Jobs and AWS Batch array jobs; see [`deploy/README.md`](deploy/README.md).

## Results

Measured on a GitHub Actions runner (AMD EPYC 7763, 4 vCPU = 2 cores × SMT, gcc 13, Release), re-measured on every push; the [project page](https://henryzhangpku.github.io/tickforge/#benchmarks) shows the latest run and a dev-container run, each labelled with its machine. Shared VMs are noisy, so treat differences under ~10% as noise.

| What | Result |
|---|---|
| Backtest, one thread (features, strategy, quote-aware fills, P&L) | ~29–32 M ticks/s (≈ 33 ns/tick) |
| Same backtest in the browser (WebAssembly, one core) | ~12–16 M ticks/s; results identical to native |
| Same backtest through the two-thread SPSC pipeline | equal to direct replay (producer fully overlapped) |
| Sweep, 32 backtests × 1 M ticks | 1.11 s on 1 thread → 0.43 s on 4 (2.0× at 2 threads; SMT limits the rest) |
| Fork-join tree of 32k tiny tasks | work stealing 2.5× faster than one shared queue |
| Coarse sweep tasks | work stealing and one shared queue tie |
| Feature updates, CRTP vs virtual | 55 vs 40 M ticks/s |
| Backtest, CRTP strategy vs virtual plugin | within noise (one predictable indirect call per tick) |
| Price-column scan through `mmap` | ~8.5 GB/s (warm page cache) |
| Batch features on CPU, 4096 symbols × 4096 ticks | ~80 M cells/s |

Correctness checks worth pointing at:

- `tests/test_no_alloc.cpp`: zero calls to `operator new` while 200k ticks go through a constructed backtester.
- `tests/test_sweep.cpp`: sweeps bit-identical across 1–8 threads and both pools; shards partition the work exactly.
- `tests/test_backtest.cpp`: buys pay the ask plus impact; partial fills at the touch; latency delays fills; never fills on the decision tick; pipelined == direct; virtual == CRTP.
- `tests/test_batch_features.cpp`: CUDA output equals the CPU reference (runs where a GPU exists; see the Colab notebook).
- CI job `kubernetes`: the sweep runs as an Indexed Job on a kind cluster; the merged pod output is byte-compared with a single-machine run.
- Pages workflow: the WebAssembly engine's P&L must equal the native build's, and the live site must serve the new commit.

## Limitations

- Synthetic data by default. The CSV importer accepts your own data; there is no full order book, so queue position and hidden liquidity are not modelled.
- Impact is linear and temporary; no permanent impact or adverse selection.
- The synthetic price is mean-reverting by construction, so Sharpe ratios on it are far higher than anything real; compare them with each other, not with the world.
- GitHub's free runners have no GPU: CI compiles the CUDA kernel with `nvcc` and runs the suite (GPU comparisons skip themselves); the [Colab notebook](https://colab.research.google.com/github/henryzhangpku/tickforge/blob/main/python/examples/gpu_colab.ipynb) runs it on a real T4.
- Cloud Run and AWS Batch definitions are provided but not executed in CI (no cloud account); the Kubernetes job verifies the same image and sharding contract.
- The browser build is single-threaded: WebAssembly threads need cross-origin-isolation headers that GitHub Pages cannot send.
- POSIX and little-endian only.

## License

MIT, see [LICENSE](LICENSE). Learning project, synthetic data, not investment advice.
