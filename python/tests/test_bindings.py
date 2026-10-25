"""Tests for the Python bindings. Run with:  PYTHONPATH=build/release python3 -m unittest discover python/tests"""

import gc
import os
import tempfile
import unittest

import numpy as np

import tickforge


class BindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.gettempdir(), f"tickforge_pytest_{os.getpid()}.tf")
        tickforge.generate(cls.path, n_ticks=50_000, seed=3)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path)

    def test_columns_are_zero_copy_readonly_views(self):
        store = tickforge.TickStore(self.path)
        px = store.price
        self.assertEqual(len(px), len(store))
        self.assertFalse(px.flags.owndata)
        self.assertFalse(px.flags.writeable)
        with self.assertRaises(ValueError):
            px[0] = 1.0
        # Two views of the same column share memory: both point at the mmap.
        self.assertTrue(np.shares_memory(px, store.price))
        self.assertTrue(set(np.unique(store.side)) <= {-1, 1})
        self.assertTrue(np.all(store.ask > store.bid))
        self.assertTrue(np.array_equal(store.price, np.where(store.side > 0, store.ask, store.bid)))
        self.assertEqual(store.symbol, "SYN")
        self.assertTrue(np.all(np.diff(store.ts) >= 0))

    def test_view_keeps_mapping_alive(self):
        px = tickforge.TickStore(self.path).price  # store has no other reference
        gc.collect()
        self.assertGreater(float(px.sum()), 0.0)  # would crash if unmapped

    def test_generate_is_deterministic(self):
        other = self.path + ".b"
        try:
            tickforge.generate(other, n_ticks=50_000, seed=3)
            self.assertTrue(np.array_equal(tickforge.TickStore(other).price, tickforge.TickStore(self.path).price))
        finally:
            os.remove(other)

    def test_sweep_shapes_and_determinism(self):
        store = tickforge.TickStore(self.path)
        kw = dict(windows=[100, 300], entry_z=[0.5, 1.0, 1.5], exit_z=[0.0, 0.25])
        a = tickforge.sweep(store, threads=1, pool="central", **kw)
        b = tickforge.sweep(store, threads=4, pool="work_stealing", **kw)
        self.assertEqual(len(a["pnl"]), 12)
        self.assertEqual(a["pnl"].dtype, np.float64)
        self.assertTrue(a["pnl"].flags.owndata is False)  # owned by the capsule, not copied
        for k in a:
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertEqual(list(a["window"][:6]), [100] * 6)

    def test_backtest_matches_sweep_row(self):
        store = tickforge.TickStore(self.path)
        sw = tickforge.sweep(store, windows=[300], entry_z=[1.0], exit_z=[0.25])
        stats, equity = tickforge.backtest(store, window=300, entry_z=1.0, exit_z=0.25)
        self.assertEqual(stats["pnl"], sw["pnl"][0])
        self.assertEqual(equity.shape, (len(store),))
        self.assertEqual(equity[-1], stats["pnl"])

    def test_execution_keywords(self):
        store = tickforge.TickStore(self.path)
        cheap, _ = tickforge.backtest(store, window=300, entry_z=0.5, impact_bps_per_lot=0.0, fee_per_share=0.0)
        dear, _ = tickforge.backtest(store, window=300, entry_z=0.5, impact_bps_per_lot=5.0, fee_per_share=0.01)
        self.assertLess(dear["pnl"], cheap["pnl"])
        self.assertGreater(dear["exec_cost"], cheap["exec_cost"])
        slow, _ = tickforge.backtest(store, window=300, entry_z=0.5, latency_ns=50_000_000)
        self.assertNotEqual(slow["pnl"], cheap["pnl"])
        with self.assertRaises(TypeError):
            tickforge.backtest(store, not_a_param=1)
        with self.assertRaises(ValueError):
            tickforge.sweep(store, pool="nope")

    def test_universe_sweep(self):
        d = tempfile.mkdtemp()
        paths = tickforge.generate_universe(d, 3, n_ticks=20_000, seed=5)
        stores = [tickforge.TickStore(p) for p in paths]
        self.assertEqual([s.symbol for s in stores], ["SYN000", "SYN001", "SYN002"])
        res = tickforge.sweep(stores, windows=[100, 200], entry_z=[1.0], exit_z=[0.25])
        self.assertEqual(list(res["symbol"]), [0, 0, 1, 1, 2, 2])
        one = tickforge.sweep(stores[2], windows=[100, 200], entry_z=[1.0], exit_z=[0.25])
        self.assertTrue(np.array_equal(res["pnl"][4:], one["pnl"]))
        for p in paths:
            os.remove(p)
        os.rmdir(d)

    def test_import_csv(self):
        csv = self.path + ".csv"
        out = self.path + ".imported"
        with open(csv, "w") as f:
            f.write("ts,price,size,side,bid,ask,bid_size,ask_size\n")
            f.write("1,10.02,100,1,10.00,10.02,300,300\n2,10.00,200,-1,10.00,10.02,300,300\n")
        try:
            self.assertEqual(tickforge.import_csv(csv, out, symbol="ABC"), 2)
            st = tickforge.TickStore(out)
            self.assertEqual(st.symbol, "ABC")
            self.assertEqual(list(st.size), [100, 200])
        finally:
            os.remove(csv)
            if os.path.exists(out):
                os.remove(out)

    def test_rolling_features(self):
        rng = np.random.default_rng(1)
        prices = 100 + rng.random((500, 7))
        mean, vol, backend = tickforge.rolling_features(prices, window=20, backend="cpu")
        self.assertEqual(backend, "cpu")
        self.assertEqual(mean.shape, prices.shape)
        np.testing.assert_allclose(mean[100], prices[81:101].mean(axis=0), rtol=1e-12)
        self.assertTrue(np.all(vol >= 0))
        with self.assertRaises(ValueError):
            tickforge.rolling_features(prices[0])
        if not tickforge.cuda_available():
            with self.assertRaises(RuntimeError):
                tickforge.rolling_features(prices, backend="cuda")


if __name__ == "__main__":
    unittest.main()
