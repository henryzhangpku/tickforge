"""Notebook-style walk-through of the Python API.

Run from the repo root after building with -DTICKFORGE_PYTHON=ON:

    PYTHONPATH=build/release python3 python/examples/research_example.py

Everything here is synthetic data and a toy strategy. It shows the workflow
(generate -> map -> sweep -> inspect), not a trading idea.
"""

import os
import shutil
import tempfile
import time

import numpy as np

import tickforge

# %% 1. Generate a deterministic synthetic tick file and map it.
path = os.path.join(tempfile.gettempdir(), "tickforge_example.tf")
tickforge.generate(path, n_ticks=2_000_000, seed=42)
store = tickforge.TickStore(path)
print(store)

# Columns are NumPy views straight into the mmap: no copy, read-only.
px = store.price
print(f"price column: dtype={px.dtype} len={len(px):,} owns_data={px.flags.owndata} "
      f"writeable={px.flags.writeable}")
spread_bps = 1e4 * (store.ask - store.bid) / ((store.ask + store.bid) / 2)
print(f"first/last price: {px[0]:.2f} / {px[-1]:.2f}, mean size {store.size.mean():.0f}, "
      f"median spread {np.median(spread_bps):.2f} bp")

# %% 2. A parameter sweep on all cores. Results come back as NumPy arrays.
t0 = time.perf_counter()
res = tickforge.sweep(
    store,
    windows=[200, 500, 1000, 2000],
    entry_z=[0.5, 0.75, 1.0, 1.5],
    exit_z=[0.0, 0.25],
)
dt = time.perf_counter() - t0
n = len(res["pnl"])
print(f"\n{n} backtests x {len(store):,} ticks in {dt:.2f} s "
      f"({n * len(store) / dt / 1e6:.0f} M ticks/s aggregate)")

# %% 3. Rank by Sharpe. Fills cross the spread, so high-turnover settings pay for it.
order = np.argsort(-res["sharpe"])
print("\nTop 5 and bottom 3 by Sharpe (per 1,000-tick bucket):")
print(f"{'window':>7} {'entry':>6} {'exit':>5} {'pnl':>10} {'exec cost':>10} {'sharpe':>7} {'trades':>7}")
for i in list(order[:5]) + list(order[-3:]):
    print(f"{res['window'][i]:>7} {res['entry_z'][i]:>6.2f} {res['exit_z'][i]:>5.2f} "
          f"{res['pnl'][i]:>10,.0f} {res['exec_cost'][i]:>10,.0f} {res['sharpe'][i]:>7.3f} {res['trades'][i]:>7}")

# %% 4. Same sweep with 1 thread gives identical numbers (determinism).
res1 = tickforge.sweep(store, windows=[200, 500], entry_z=[1.0], exit_z=[0.25], threads=1)
resN = tickforge.sweep(store, windows=[200, 500], entry_z=[1.0], exit_z=[0.25], threads=0)
assert np.array_equal(res1["pnl"], resN["pnl"]), "sweep must not depend on thread count"
print("\n1-thread and N-thread sweeps are bit-identical")

# %% 5. How much does 50 ms of latency and heavier impact cost the best setting?
best = order[0]
kw = dict(window=int(res["window"][best]), entry_z=float(res["entry_z"][best]), exit_z=float(res["exit_z"][best]))
base, _ = tickforge.backtest(store, **kw)
slow, _ = tickforge.backtest(store, latency_ns=50_000_000, impact_bps_per_lot=1.0, **kw)
print(f"\nbest setting: pnl {base['pnl']:,.0f} -> {slow['pnl']:,.0f} with 50 ms latency and 1 bp/lot impact")

# %% 6. One backtest with its equity curve.
stats, equity = tickforge.backtest(store, **kw)
print(f"\nbest params backtest: pnl={stats['pnl']:,.0f} trades={stats['trades']} "
      f"max_dd={stats['max_drawdown']:,.0f}; equity curve shape {equity.shape}")

# %% 7. A small universe: the same grid over 4 symbols in one call.
udir = tempfile.mkdtemp()
universe = [tickforge.TickStore(p) for p in tickforge.generate_universe(udir, 4, n_ticks=500_000)]
u = tickforge.sweep(universe, windows=[500, 1000], entry_z=[0.5, 1.0], exit_z=[0.25])
for s_idx, st in enumerate(universe):
    rows = u["symbol"] == s_idx
    print(f"{st.symbol}: best sharpe {u['sharpe'][rows].max():.3f} over {rows.sum()} settings")

# %% 8. Batch rolling features across many symbols (CUDA if built, else CPU).
rng = np.random.default_rng(0)
prices = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, size=(10_000, 256)), axis=0))
mean, vol, backend = tickforge.rolling_features(prices, window=100)
print(f"\nrolling features on {prices.shape} via {backend}: "
      f"mean[-1,:3]={np.round(mean[-1, :3], 3)}, vol[-1,:3]={np.round(vol[-1, :3], 6)}")

os.remove(path)
shutil.rmtree(udir)
