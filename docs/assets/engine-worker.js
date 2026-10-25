// Runs the tickforge engine (C++ compiled to WebAssembly) off the main thread.
//
// Results are read straight out of WASM linear memory through typed-array
// views, then copied once into transferable buffers for the UI thread.
/* global importScripts, createTickforge */
"use strict";

importScripts("../wasm/tickforge.js");

let M = null;
let n = 0;
let base = 0;
let sweepToken = 0;
const pending = [];

// In a worker Emscripten resolves the .wasm against the worker's URL, not the
// loader's, so point it at the right directory.
createTickforge({ locateFile: (f) => "../wasm/" + f }).then((mod) => {
  M = mod;
  postMessage({ type: "ready" });
  pending.splice(0).forEach(handle);
});

onmessage = (e) => (M ? handle(e.data) : pending.push(e.data));

const f64 = (ptr, len) => new Float64Array(M.HEAPF64.buffer, ptr, len);

function args(p) {
  return [p.window, p.entryZ, p.exitZ, p.fee, p.impact, p.latencyMs, p.touch ? 1 : 0];
}

function handle(msg) {
  switch (msg.cmd) {
    case "generate": {
      sweepToken++; // a new dataset invalidates any running sweep
      const genMs = M._tf_generate(msg.n, msg.seed);
      n = M._tf_size();
      const bid = f64(M._tf_bid(), n);
      const ask = f64(M._tf_ask(), n);
      base = 0.5 * (bid[0] + ask[0]);
      // Float32 relative to the first mid keeps full tick precision on the GPU.
      const bid32 = new Float32Array(n);
      const ask32 = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        bid32[i] = bid[i] - base;
        ask32[i] = ask[i] - base;
      }
      postMessage(
        { type: "data", id: msg.id, n, base, genMs, spanSeconds: M._tf_span_seconds(), bid: bid32, ask: ask32 },
        [bid32.buffer, ask32.buffer]
      );
      break;
    }
    case "backtest": {
      const sp = M._tf_backtest(...args(msg.p));
      const s = f64(sp, 11);
      const stats = {
        pnl: s[0], fees: s[1], execCost: s[2], sharpe: s[3], sharpeAnnual: s[4], maxDrawdown: s[5],
        trades: s[6], partialFills: s[7], shares: s[8], finalPosition: s[9], elapsedMs: s[10],
      };
      const eq = f64(M._tf_equity(), n);
      const pos = new Int32Array(M.HEAP32.buffer, M._tf_position(), n);
      const equity = new Float32Array(n);
      for (let i = 0; i < n; i++) equity[i] = eq[i];
      // Fills = position changes. Buys print at the ask, sells at the bid.
      const bid = f64(M._tf_bid(), n);
      const ask = f64(M._tf_ask(), n);
      const fx = [];
      const fy = [];
      const fs = [];
      let prev = 0;
      for (let i = 0; i < n; i++) {
        const d = pos[i] - prev;
        if (d !== 0) {
          fx.push(i);
          fy.push((d > 0 ? ask[i] : bid[i]) - base);
          fs.push(d > 0 ? 1 : -1);
        }
        prev = pos[i];
      }
      const fillX = Float32Array.from(fx);
      const fillY = Float32Array.from(fy);
      const fillSide = Int8Array.from(fs);
      const position = Int32Array.from(pos);
      postMessage(
        { type: "backtest", id: msg.id, stats, equity, position, fillX, fillY, fillSide },
        [equity.buffer, position.buffer, fillX.buffer, fillY.buffer, fillSide.buffer]
      );
      break;
    }
    case "sweep": {
      const token = ++sweepToken;
      const cells = [];
      msg.windows.forEach((w, i) => msg.entries.forEach((z, j) => cells.push([i, j, w, z])));
      let k = 0;
      const t0 = performance.now();
      // One grid point per macrotask, so a newer command can interrupt.
      const step = () => {
        if (token !== sweepToken) return;
        if (k === cells.length) {
          postMessage({ type: "sweepDone", id: msg.id, ms: performance.now() - t0, count: cells.length });
          return;
        }
        const [i, j, w, z] = cells[k++];
        const r = f64(M._tf_sweep_point(...args({ ...msg.p, window: w, entryZ: z })), 6);
        postMessage({ type: "cell", id: msg.id, i, j, pnl: r[0], sharpe: r[1], trades: r[2], execCost: r[3], ms: r[5] });
        setTimeout(step, 0);
      };
      step();
      break;
    }
    default:
      break;
  }
}
