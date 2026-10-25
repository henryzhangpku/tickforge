# tickforge

[![CI](https://github.com/henryzhangpku/tickforge/actions/workflows/ci.yml/badge.svg)](https://github.com/henryzhangpku/tickforge/actions/workflows/ci.yml)
[![Pages](https://github.com/henryzhangpku/tickforge/actions/workflows/pages.yml/badge.svg)](https://henryzhangpku.github.io/tickforge/)

A small C++20 research engine for tick data: a memory-mapped columnar store, arena memory, a lock-free pipeline, an event-driven backtest, parallel parameter sweeps, zero-copy Python bindings, and an optional CUDA kernel.

**Live page with architecture, benchmark charts and design notes: <https://henryzhangpku.github.io/tickforge/>**

> **Learning project.** All data is synthetic, generated from a seed. The strategy is a deliberately simple toy; that it makes money on this data says something about how the data was generated, not about markets. Nothing here is investment advice.

```python
import tickforge

tickforge.generate("ticks.tf", n_ticks=2_000_000, seed=42)
store = tickforge.TickStore("ticks.tf")          # mmap; store.price is a zero-copy NumPy view
res = tickforge.sweep(store,                      # 18 backtests on all cores, GIL released
                      windows=[200, 500, 1000], entry_z=[0.5, 1.0, 1.5], exit_z=[0.0, 0.25])
best = res["sharpe"].argmax()                     # results are NumPy arrays, no copies
```

## What's in it, and why

| Piece | Where | Concern it demonstrates |
|---|---|---|
| Columnar tick file, `mmap`, `std::span` views | `include/tickforge/tick_store.hpp`, `src/tick_store.cpp` | Hardware/software interface: SoA layout, 64-byte aligned columns, page cache, `MADV_SEQUENTIAL`, zero-copy reads |
| Arena (`std::pmr::monotonic_buffer_resource`), aligned buffers, counting resource | `include/tickforge/memory.hpp`, `tests/test_no_alloc.cpp` | Memory management: no allocation on the hot path, *proven* by a test that replaces global `operator new` |
| Lock-free SPSC ring (acquire/release, cached indices, batched consume) | `include/tickforge/spsc_ring.hpp` | Low-level concurrency: memory ordering, false sharing, cache-line traffic |
| Streaming features: rolling VWAP, EWMA volatility, order-flow imbalance | `include/tickforge/features.hpp` | O(1) per tick, no allocation, numerically re-anchored |
| CRTP strategy/feature dispatch vs virtual plugin interface + registry | `include/tickforge/strategy.hpp`, `include/tickforge/plugins.hpp` | Design patterns and the cost of each, measured |
| Event-driven backtest: next-tick fills, fees, slippage, P&L, drawdown, Sharpe | `include/tickforge/backtest.hpp` | Correct-by-construction research tooling (no look-ahead) |
| Thread pool with futures; parameter sweeps | `include/tickforge/thread_pool.hpp`, `src/sweep.cpp` | Parallel/async patterns, deterministic results at any thread count |
| pybind11 module | `python/bindings.cpp`, `python/examples/research_example.py` | C++ and Python together: buffer-protocol NumPy views, capsule-owned results, GIL release |
| CUDA rolling-feature kernel + CPU fallback | `src/batch_features.cu`, `src/batch_features.cpp` | GPU computing: coalesced time-major layout, one thread per symbol |
| CMake presets, CI (gcc, clang, ASan/UBSan, TSan, clang-tidy, benchmarks), Docker, Pages | `CMakePresets.json`, `.github/workflows/`, `Dockerfile` | Production engineering: reproducible builds, sanitizers, static analysis, a container that tests itself while building |

The design notes (CRTP vs virtual, why a central-queue pool rather than work stealing, the SPSC memory-ordering argument, the CUDA mental model) are on the [Pages site](https://henryzhangpku.github.io/tickforge/#design-notes).

## Build and run

Requirements: a C++20 compiler (gcc 13 or clang 18 tested), CMake 3.25+, Ninja. Dependencies (GoogleTest, Google Benchmark, pybind11) are fetched by CMake.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release

./build/release/tickforge_cli gen ticks.tf 2000000 42     # deterministic synthetic data
./build/release/tickforge_cli backtest ticks.tf
./build/release/tickforge_cli pipeline ticks.tf            # same result through the SPSC ring
./build/release/tickforge_cli sweep ticks.tf               # default grid on all cores
./build/release/tickforge_bench                            # microbenchmarks
```

Python module:

```sh
pip install numpy
cmake --preset python && cmake --build --preset python
PYTHONPATH=build/python python3 python/examples/research_example.py
PYTHONPATH=build/python python3 -m unittest discover python/tests
```

Other presets: `debug`, `asan` (AddressSanitizer + UBSan), `tsan` (ThreadSanitizer), `cuda` (needs `nvcc`; adds the GPU path to `rolling_features_batch`, chosen automatically when a device is present).

Docker:

```sh
docker build -t tickforge .     # builds and runs the test suite inside the build stage
docker run --rm tickforge       # generates data, runs a backtest, a sweep, and the Python example
```

## Results

From the committed run on a 4-vCPU cloud dev container (Intel Xeon @ 2.8 GHz, gcc 13, Release); CI re-measures on a GitHub runner for every push to `main` and the Pages site shows both, each labelled with its machine. Shared VMs are noisy: treat differences under ~10% as noise.

| What | Result |
|---|---|
| Backtest, one thread (features + strategy + fills + P&L) | ~21 M ticks/s (≈ 48 ns/tick) |
| Same backtest through the two-thread SPSC pipeline | ~20–22 M ticks/s (producer fully overlapped) |
| SPSC ring alone, 2 threads | ~54–66 M items/s |
| Sweep, 32 backtests × 1 M ticks | 1.50 s on 1 thread → 0.44 s on 4 threads (3.4×, 86% efficiency) |
| Feature updates, CRTP vs virtual | 44 vs 36 M ticks/s (virtual 20–50% slower across runs) |
| Backtest, CRTP strategy vs virtual plugin | within noise (one predictable indirect call per tick) |
| Price-column scan through `mmap` | ~5 GB/s, 630 M prices/s (warm page cache) |
| Scratch allocation, arena vs `new`/`delete` (3 × 4 KB vectors) | 131 ns vs 196 ns |
| Batch features on CPU, 512 symbols × 4096 ticks | ~70 M cells/s |

Correctness checks worth pointing at:

- `tests/test_no_alloc.cpp` — zero calls to `operator new` while 200k ticks go through a constructed backtester.
- `tests/test_sweep.cpp` — sweep output is bit-identical with 1, 2, 3, 4 and 8 threads.
- `tests/test_backtest.cpp` — pipelined == direct; virtual plugin == CRTP; costs never change decisions and only lower P&L; reported drawdown equals the drawdown of the equity curve.
- `tests/test_features.cpp` — rolling VWAP matches brute force over 10k ticks.
- `python/tests/test_bindings.py` — columns are read-only views sharing memory with the mmap and stay valid after the `TickStore` is dropped.

## Limitations

- Synthetic data only; one instrument per file; trades only (no quotes, no order book).
- Fill model is simple: market orders fill at the next tick's price plus fixed slippage; no queue position, partial fills or impact.
- Sharpe is per 1,000-tick bucket and not annualised.
- The CUDA kernel is written but not run in CI (no GPU runners). It mirrors the CPU fallback, which is tested; agreement is expected to within a few ULP of `log`.
- POSIX and little-endian only.
- The thread pool is a single shared queue, not work stealing (the right call for coarse tasks; see the design notes).
- Cloud deployment is limited to the Docker image and CI; there is no managed-service deployment.

## License

MIT, see [LICENSE](LICENSE). Learning project, synthetic data, not investment advice.
