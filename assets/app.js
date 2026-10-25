// tickforge browser demo: UI thread.
//
//  * The engine (C++ -> WebAssembly) runs in engine-worker.js.
//  * The tick chart is WebGL2: the bid/ask band is a triangle strip, the mid a
//    line strip and fills are GL points shaped in the fragment shader. Zoom and
//    pan only change two uniforms, so a million ticks redraw at frame rate.
//  * The equity chart is a 2D canvas, downsampled to min/max per pixel column.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const tip = $("tip");
  const fmtInt = (v) => Math.round(v).toLocaleString();
  const fmt = (v, d = 1) => v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const money = (v) => (v < 0 ? "−$" : "$") + fmtInt(Math.abs(v));

  // ---- theme ---------------------------------------------------------------
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("tf-theme");
    if (saved) root.dataset.theme = saved;
  } catch (e) { /* storage unavailable */ }
  $("theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("tf-theme", root.dataset.theme); } catch (e) { /* ignore */ }
    redrawAll();
  });

  // ---- color helpers -------------------------------------------------------
  function parseColor(c) {
    const ctx = parseColor.ctx || (parseColor.ctx = document.createElement("canvas").getContext("2d"));
    ctx.fillStyle = "#000";
    ctx.fillStyle = c;
    const s = ctx.fillStyle;
    if (s.startsWith("#")) {
      const n = parseInt(s.slice(1), 16);
      return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
    }
    const m = s.match(/rgba?\(([^)]+)\)/);
    const p = m[1].split(",").map(Number);
    return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
  }
  const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
  const toCss = (c) => `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;

  // ---- state ---------------------------------------------------------------
  const S = {
    n: 0, base: 0, spanSeconds: 0, bid: null, ask: null, mid: null,
    equity: null, position: null, fills: null, stats: null,
    view: { x0: 0, x1: 1 }, blocks: null, hover: -1,
  };

  // ---- controls ------------------------------------------------------------
  const ctl = {
    window: { el: $("window"), out: $("o-window"), f: (v) => fmtInt(v) + " ticks" },
    entry: { el: $("entry"), out: $("o-entry"), f: (v) => fmt(+v, 2) },
    exit: { el: $("exit"), out: $("o-exit"), f: (v) => fmt(+v, 2) },
    latency: { el: $("latency"), out: $("o-latency"), f: (v) => v + " ms" },
    impact: { el: $("impact"), out: $("o-impact"), f: (v) => fmt(+v, 1) + " bp" },
    fee: { el: $("fee"), out: $("o-fee"), f: (v) => "$" + (+v).toFixed(4) },
  };
  Object.values(ctl).forEach((c) => {
    const upd = () => (c.out.textContent = c.f(c.el.value));
    upd();
    c.el.addEventListener("input", () => { upd(); scheduleBacktest(); });
  });
  $("touch").addEventListener("change", () => scheduleBacktest());
  $("n").addEventListener("change", generate);
  $("seed").addEventListener("change", generate);

  const params = () => ({
    window: +ctl.window.el.value, entryZ: +ctl.entry.el.value, exitZ: +ctl.exit.el.value,
    latencyMs: +ctl.latency.el.value, impact: +ctl.impact.el.value, fee: +ctl.fee.el.value, touch: $("touch").checked,
  });

  // ---- worker --------------------------------------------------------------
  let worker = null;
  let seq = 0;
  let latestBacktest = 0;
  let latestSweep = 0;
  let btTimer = 0;
  let sweepTimer = 0;
  try {
    worker = new Worker("assets/engine-worker.js");
  } catch (e) {
    $("engine-status").textContent = "This browser cannot start a Web Worker; the live demo is unavailable.";
  }
  if (worker) {
    worker.onmessage = (e) => onWorker(e.data);
    worker.onerror = () => ($("engine-status").textContent = "The WebAssembly engine failed to load.");
  }

  function generate() {
    if (!worker) return;
    $("engine-status").textContent = "Generating synthetic ticks in WebAssembly…";
    worker.postMessage({ cmd: "generate", id: ++seq, n: +$("n").value, seed: +$("seed").value || 0 });
  }
  function scheduleBacktest(delay = 60) {
    clearTimeout(btTimer);
    btTimer = setTimeout(runBacktest, delay);
    clearTimeout(sweepTimer);
    sweepTimer = setTimeout(runSweep, 900); // execution settings changed: refresh the heatmap
  }
  function runBacktest() {
    if (!worker || !S.n) return;
    latestBacktest = ++seq;
    worker.postMessage({ cmd: "backtest", id: latestBacktest, p: params() });
  }

  function onWorker(m) {
    if (m.type === "ready") {
      $("engine-status").textContent = "C++ engine loaded (WebAssembly, " + (navigator.hardwareConcurrency || "?") + " cores available)";
      generate();
    } else if (m.type === "data") {
      S.n = m.n; S.base = m.base; S.spanSeconds = m.spanSeconds; S.bid = m.bid; S.ask = m.ask;
      S.mid = new Float32Array(m.n);
      for (let i = 0; i < m.n; i++) S.mid[i] = 0.5 * (m.bid[i] + m.ask[i]);
      buildBlocks();
      S.view = { x0: 0, x1: m.n - 1 };
      gl.setData();
      $("k-ticks").textContent = m.n >= 1e6 ? fmt(m.n / 1e6, m.n % 1e6 ? 1 : 0) + "M" : fmtInt(m.n / 1e3) + "k";
      $("engine-status").textContent = `Generated ${fmtInt(m.n)} ticks in ${fmt(m.genMs, 0)} ms (WebAssembly)`;
      runBacktest();
      runSweep();
    } else if (m.type === "backtest") {
      if (m.id !== latestBacktest) return;
      S.stats = m.stats; S.equity = m.equity; S.position = m.position;
      S.fills = { x: m.fillX, y: m.fillY, side: m.fillSide };
      gl.setFills();
      showStats();
      redrawAll();
    } else if (m.type === "cell") {
      if (m.id !== latestSweep) return;
      sweepCell(m);
    } else if (m.type === "sweepDone") {
      if (m.id !== latestSweep) return;
      $("sweep-status").textContent = `${m.count} backtests × ${fmtInt(S.n)} ticks in ${fmt(m.ms / 1000, 2)} s on one core (${fmt((m.count * S.n) / m.ms / 1e3, 1)}M ticks/s).`;
      $("sweep-bar").style.width = "100%";
    }
  }

  function showStats() {
    const s = S.stats;
    const set = (id, text, sign) => {
      const el = $(id);
      el.textContent = text;
      el.className = "v" + (sign > 0 ? " pos" : sign < 0 ? " neg" : "");
    };
    set("t-pnl", money(s.pnl), s.pnl);
    set("t-sharpe", fmt(s.sharpeAnnual, 1), s.sharpeAnnual);
    set("t-trades", fmtInt(s.trades) + (s.partialFills ? ` (${fmtInt(s.partialFills)})` : ""));
    set("t-cost", money(s.execCost));
    set("t-dd", money(s.maxDrawdown));
    const mts = S.n / s.elapsedMs / 1e3;
    set("t-speed", fmt(s.elapsedMs, 0) + " ms");
    $("k-wasm").textContent = fmt(mts, 1) + "M ticks/s";
    $("engine-info").innerHTML = `Engine: C++20 → WebAssembly in a Web Worker.<br><b>${fmtInt(S.n)} ticks in ${fmt(s.elapsedMs, 1)} ms</b> = ${fmt(mts, 1)}M ticks/s, including feature updates, the strategy, fills and P&amp;L for every tick.`;
  }

  // ---- y-range acceleration: min/max per block of 256 ticks ----------------
  const BLOCK = 256;
  function buildBlocks() {
    const nb = Math.ceil(S.n / BLOCK);
    const lo = new Float32Array(nb), hi = new Float32Array(nb);
    for (let b = 0; b < nb; b++) {
      let mn = Infinity, mx = -Infinity;
      const end = Math.min(S.n, (b + 1) * BLOCK);
      for (let i = b * BLOCK; i < end; i++) {
        if (S.bid[i] < mn) mn = S.bid[i];
        if (S.ask[i] > mx) mx = S.ask[i];
      }
      lo[b] = mn; hi[b] = mx;
    }
    S.blocks = { lo, hi };
  }
  function yRange(x0, x1) {
    let mn = Infinity, mx = -Infinity;
    const a = Math.max(0, Math.floor(x0)), b = Math.min(S.n - 1, Math.ceil(x1));
    let i = a;
    while (i <= b) {
      if (i % BLOCK === 0 && i + BLOCK - 1 <= b) {
        const k = i / BLOCK;
        if (S.blocks.lo[k] < mn) mn = S.blocks.lo[k];
        if (S.blocks.hi[k] > mx) mx = S.blocks.hi[k];
        i += BLOCK;
      } else {
        if (S.bid[i] < mn) mn = S.bid[i];
        if (S.ask[i] > mx) mx = S.ask[i];
        i++;
      }
    }
    const pad = Math.max((mx - mn) * 0.08, 0.02);
    return [mn - pad, mx + pad];
  }

  // ---- WebGL2 chart --------------------------------------------------------
  const gl = (function () {
    const canvas = $("price-gl");
    const ctx = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
    const api = { ok: !!ctx, ctx, setData() {}, setFills() {}, draw() {} };
    if (!ctx) {
      $("price-box").insertAdjacentHTML("beforeend", '<p style="position:absolute;inset:auto 0 45% 0;text-align:center;color:var(--ink-3)">WebGL2 is not available in this browser.</p>');
      return api;
    }
    const vs = `#version 300 es
      in vec2 a_pos;
      uniform vec2 u_origin; uniform vec2 u_scale; uniform float u_size;
      void main() {
        vec2 p = (a_pos - u_origin) * u_scale - 1.0;   // subtract first: exact for integer x in float32
        gl_Position = vec4(p, 0.0, 1.0);
        gl_PointSize = u_size;
      }`;
    const fs = `#version 300 es
      precision mediump float;
      uniform vec4 u_color; uniform int u_shape;
      out vec4 o;
      void main() {
        if (u_shape != 0) {
          vec2 c = gl_PointCoord;                       // (0,0) top-left
          float half_w = abs(c.x - 0.5) * 2.0;
          float lim = u_shape == 1 ? c.y : 1.0 - c.y;   // 1: up triangle, 2: down
          if (half_w > lim) discard;
        }
        o = u_color;
      }`;
    const compile = (type, src) => {
      const s = ctx.createShader(type);
      ctx.shaderSource(s, src);
      ctx.compileShader(s);
      if (!ctx.getShaderParameter(s, ctx.COMPILE_STATUS)) throw new Error(ctx.getShaderInfoLog(s));
      return s;
    };
    const prog = ctx.createProgram();
    ctx.attachShader(prog, compile(ctx.VERTEX_SHADER, vs));
    ctx.attachShader(prog, compile(ctx.FRAGMENT_SHADER, fs));
    ctx.linkProgram(prog);
    const loc = {
      pos: ctx.getAttribLocation(prog, "a_pos"),
      origin: ctx.getUniformLocation(prog, "u_origin"), scale: ctx.getUniformLocation(prog, "u_scale"),
      size: ctx.getUniformLocation(prog, "u_size"), color: ctx.getUniformLocation(prog, "u_color"),
      shape: ctx.getUniformLocation(prog, "u_shape"),
    };
    const mkBuf = () => {
      const vao = ctx.createVertexArray();
      const buf = ctx.createBuffer();
      ctx.bindVertexArray(vao);
      ctx.bindBuffer(ctx.ARRAY_BUFFER, buf);
      ctx.enableVertexAttribArray(loc.pos);
      ctx.vertexAttribPointer(loc.pos, 2, ctx.FLOAT, false, 0, 0);
      ctx.bindVertexArray(null);
      return { vao, buf, count: 0 };
    };
    const band = mkBuf(), mid = mkBuf(), buys = mkBuf(), sells = mkBuf();
    const upload = (b, data, count) => {
      ctx.bindBuffer(ctx.ARRAY_BUFFER, b.buf);
      ctx.bufferData(ctx.ARRAY_BUFFER, data, ctx.STATIC_DRAW);
      b.count = count;
    };

    api.setData = () => {
      const n = S.n;
      const strip = new Float32Array(n * 4);
      const line = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) {
        strip[4 * i] = i; strip[4 * i + 1] = S.bid[i]; strip[4 * i + 2] = i; strip[4 * i + 3] = S.ask[i];
        line[2 * i] = i; line[2 * i + 1] = S.mid[i];
      }
      upload(band, strip, 2 * n);
      upload(mid, line, n);
      upload(buys, new Float32Array(0), 0);
      upload(sells, new Float32Array(0), 0);
    };
    api.setFills = () => {
      const f = S.fills;
      const b = [], s = [];
      for (let k = 0; k < f.x.length; k++) (f.side[k] > 0 ? b : s).push(f.x[k], f.y[k]);
      upload(buys, new Float32Array(b), b.length / 2);
      upload(sells, new Float32Array(s), s.length / 2);
    };
    api.draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.viewport(0, 0, canvas.width, canvas.height);
      ctx.clearColor(0, 0, 0, 0);
      ctx.clear(ctx.COLOR_BUFFER_BIT);
      if (!S.n) return;
      ctx.enable(ctx.BLEND);
      // Blend colour normally but accumulate alpha additively; plain
      // SRC_ALPHA/ONE_MINUS_SRC_ALPHA would square the alpha of translucent
      // marks and they would almost vanish when the canvas is composited.
      ctx.blendFuncSeparate(ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA, ctx.ONE, ctx.ONE_MINUS_SRC_ALPHA);
      ctx.useProgram(prog);
      const { x0, x1 } = S.view;
      const [y0, y1] = yRange(x0, x1);
      S.yr = [y0, y1];
      ctx.uniform2f(loc.origin, x0, y0);
      ctx.uniform2f(loc.scale, 2 / Math.max(x1 - x0, 1e-9), 2 / (y1 - y0));
      ctx.uniform1i(loc.shape, 0);
      ctx.uniform1f(loc.size, 1);
      const visible = x1 - x0;
      ctx.uniform4fv(loc.color, parseColor(css("--band")));
      ctx.bindVertexArray(band.vao);
      // The buffer holds (i, bid_i), (i, ask_i) pairs. Zoomed in it is a
      // triangle strip; zoomed out each quad is far narrower than a pixel and
      // would cover no pixel centres, so draw the same pairs as vertical lines,
      // which always rasterise.
      const ticksPerPixel = visible / canvas.width;
      ctx.drawArrays(ticksPerPixel > 0.5 ? ctx.LINES : ctx.TRIANGLE_STRIP, 0, band.count);
      ctx.uniform4fv(loc.color, parseColor(css("--mid")));
      ctx.bindVertexArray(mid.vao);
      ctx.drawArrays(ctx.LINE_STRIP, 0, mid.count);
      // Fills: a surface-coloured ring first, then the marker (2px separation).
      const size = (visible < 5000 ? 13 : visible < 50000 ? 10 : visible < 300000 ? 7 : 5) * dpr;
      const surface = parseColor(css("--bg-2"));
      for (const [buf, shape, color] of [[buys, 1, "--series-1"], [sells, 2, "--series-2"]]) {
        if (!buf.count) continue;
        ctx.bindVertexArray(buf.vao);
        ctx.uniform1i(loc.shape, shape);
        ctx.uniform1f(loc.size, size + 4 * dpr);
        ctx.uniform4fv(loc.color, surface);
        ctx.drawArrays(ctx.POINTS, 0, buf.count);
        ctx.uniform1f(loc.size, size);
        ctx.uniform4fv(loc.color, parseColor(css(color)));
        ctx.drawArrays(ctx.POINTS, 0, buf.count);
      }
      ctx.bindVertexArray(null);
    };
    return api;
  })();

  // ---- overlay (axes, crosshair) -------------------------------------------
  function niceStep(range, target) {
    const raw = range / target;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  function sessionTime(i) {
    const secs = 9.5 * 3600 + (i / Math.max(S.n - 1, 1)) * S.spanSeconds;
    const d = Math.floor(secs / 86400);
    const r = secs % 86400;
    const hh = String(Math.floor(r / 3600)).padStart(2, "0"), mm = String(Math.floor((r % 3600) / 60)).padStart(2, "0");
    const ss = String(Math.floor(r % 60)).padStart(2, "0");
    return (d ? `d${d + 1} ` : "") + `${hh}:${mm}` + (S.view.x1 - S.view.x0 < 20000 ? `:${ss}` : "");
  }
  function prep2d(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    return { c, w, h };
  }
  function drawOverlay() {
    const { c, w, h } = prep2d($("price-ov"));
    if (!S.n || !S.yr) return;
    const { x0, x1 } = S.view;
    const [y0, y1] = S.yr;
    const X = (i) => ((i - x0) / (x1 - x0)) * w, Y = (v) => h - ((v - y0) / (y1 - y0)) * h;
    c.font = "11px Inter, sans-serif";
    c.fillStyle = css("--ink-3");
    c.strokeStyle = css("--grid");
    c.lineWidth = 1;
    // Gridlines at round absolute prices (the GPU data is relative to the first mid).
    const ys = niceStep(y1 - y0, 5);
    const dec = ys < 0.01 ? 3 : 2;
    for (let k = Math.ceil((S.base + y0) / ys); k * ys <= S.base + y1; k++) {
      const price = k * ys;
      const y = Math.round(Y(price - S.base)) + 0.5;
      c.globalAlpha = 0.6; c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); c.globalAlpha = 1;
      c.fillText(price.toFixed(dec), 6, y - 3);
    }
    const xs = niceStep(x1 - x0, Math.max(2, Math.floor(w / 140)));
    c.textAlign = "center";
    for (let i = Math.ceil(x0 / xs) * xs; i <= x1; i += xs) c.fillText(sessionTime(i), Math.max(28, Math.min(w - 28, X(i))), h - 6);
    c.textAlign = "left";
    $("view-info").textContent = `${fmtInt(x1 - x0 + 1)} ticks in view`;
    if (S.hover >= 0) {
      const x = Math.round(X(S.hover)) + 0.5;
      c.strokeStyle = css("--ink-3");
      c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke();
      c.setLineDash([]);
    }
  }

  function drawEquity() {
    const { c, w, h } = prep2d($("equity"));
    if (!S.equity) return;
    const { x0, x1 } = S.view;
    const a = Math.max(0, Math.floor(x0)), b = Math.min(S.n - 1, Math.ceil(x1));
    let mn = 0, mx = 0;
    const cols = Math.max(1, Math.floor(w));
    const lo = new Float32Array(cols).fill(Infinity), hi = new Float32Array(cols).fill(-Infinity);
    const per = (b - a + 1) / cols;
    for (let i = a; i <= b; i++) {
      const k = Math.min(cols - 1, Math.floor((i - a) / per));
      const v = S.equity[i];
      if (v < lo[k]) lo[k] = v;
      if (v > hi[k]) hi[k] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const pad = Math.max((mx - mn) * 0.1, 1);
    mn -= pad; mx += pad;
    const Y = (v) => h - 4 - ((v - mn) / (mx - mn)) * (h - 8);
    const accent = css("--accent-2");
    c.beginPath();
    let started = false;
    for (let k = 0; k < cols; k++) if (hi[k] > -Infinity) { started ? c.lineTo(k, Y(hi[k])) : c.moveTo(k, Y(hi[k])); started = true; }
    for (let k = cols - 1; k >= 0; k--) if (lo[k] < Infinity) c.lineTo(k, Y(lo[k]));
    c.closePath();
    c.fillStyle = accent; c.globalAlpha = 0.25; c.fill(); c.globalAlpha = 1;
    c.strokeStyle = accent; c.lineWidth = 1.5;
    c.beginPath(); started = false;
    for (let k = 0; k < cols; k++) if (hi[k] > -Infinity) { const v = (hi[k] + lo[k]) / 2; started ? c.lineTo(k, Y(v)) : c.moveTo(k, Y(v)); started = true; }
    c.stroke();
    const z = Math.round(Y(0)) + 0.5;
    c.strokeStyle = css("--ink-3"); c.setLineDash([4, 4]); c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, z); c.lineTo(w, z); c.stroke(); c.setLineDash([]);
    c.font = "11px Inter, sans-serif";
    const label = "equity  max " + money(mx - pad) + " · min " + money(mn + pad);
    c.fillStyle = css("--bg-2"); c.globalAlpha = 0.85; c.fillRect(2, 2, c.measureText(label).width + 10, 17); c.globalAlpha = 1;
    c.fillStyle = css("--ink-3"); c.fillText(label, 6, 14);
    if (S.hover >= 0) {
      const x = Math.round(((S.hover - x0) / (x1 - x0)) * w) + 0.5;
      c.setLineDash([3, 3]); c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); c.setLineDash([]);
    }
  }

  let raf = 0;
  function redrawAll() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      gl.draw();
      drawOverlay();
      drawEquity();
    });
  }
  new ResizeObserver(redrawAll).observe($("price-box"));
  window.tickforge = { state: S, gl, redraw: redrawAll }; // handy from the devtools console

  // ---- interaction: zoom, pan, hover (linked across both charts) -----------
  function zoomAt(frac, factor) {
    if (!S.n) return;
    const { x0, x1 } = S.view;
    const span = x1 - x0;
    const c = x0 + frac * span;
    let ns = Math.min(S.n - 1, Math.max(50, span * factor));
    let a = c - frac * ns;
    a = Math.max(0, Math.min(a, S.n - 1 - ns));
    S.view = { x0: a, x1: a + ns };
    redrawAll();
  }
  function bindInteractions(box) {
    let drag = null;
    box.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = box.getBoundingClientRect();
      zoomAt((e.clientX - r.left) / r.width, Math.exp(e.deltaY * 0.0015));
    }, { passive: false });
    box.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, v: { ...S.view } };
      box.setPointerCapture(e.pointerId);
    });
    box.addEventListener("pointermove", (e) => {
      const r = box.getBoundingClientRect();
      if (drag) {
        const span = drag.v.x1 - drag.v.x0;
        let a = drag.v.x0 - ((e.clientX - drag.x) / r.width) * span;
        a = Math.max(0, Math.min(a, S.n - 1 - span));
        S.view = { x0: a, x1: a + span };
      }
      if (!S.n) return;
      const i = Math.round(S.view.x0 + ((e.clientX - r.left) / r.width) * (S.view.x1 - S.view.x0));
      S.hover = Math.max(0, Math.min(S.n - 1, i));
      showTip(e, S.hover);
      redrawAll();
    });
    const end = () => (drag = null);
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
    box.addEventListener("pointerleave", () => { S.hover = -1; tip.style.display = "none"; redrawAll(); });
    box.addEventListener("dblclick", resetView);
  }
  function showTip(e, i) {
    const pos = S.position ? S.position[i] : 0;
    const eq = S.equity ? S.equity[i] : 0;
    tip.innerHTML = `<b>tick ${fmtInt(i)}</b> · ${sessionTime(i)}<br>bid ${(S.base + S.bid[i]).toFixed(2)} · ask ${(S.base + S.ask[i]).toFixed(2)}<br>position ${pos > 0 ? "+" : ""}${pos} · equity ${money(eq)}`;
    tip.style.display = "block";
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px";
    tip.style.top = e.clientY + 14 + "px";
  }
  function resetView() { if (S.n) { S.view = { x0: 0, x1: S.n - 1 }; redrawAll(); } }
  bindInteractions($("price-box"));
  bindInteractions($("equity-box"));
  $("zoom-in").addEventListener("click", () => zoomAt(0.5, 0.4));
  $("zoom-out").addEventListener("click", () => zoomAt(0.5, 2.5));
  $("zoom-reset").addEventListener("click", resetView);

  // ---- sweep heatmap ---------------------------------------------------------
  const WINDOWS = [100, 200, 350, 500, 750, 1000, 1500, 2000];
  const ENTRIES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
  const heat = $("heat");
  const cells = [];
  heat.style.gridTemplateColumns = `56px repeat(${ENTRIES.length}, 1fr)`;
  heat.insertAdjacentHTML("beforeend", '<div class="hdr rowh">w \\ z</div>' + ENTRIES.map((z) => `<div class="hdr">${z.toFixed(2)}</div>`).join(""));
  WINDOWS.forEach((w, i) => {
    heat.insertAdjacentHTML("beforeend", `<div class="hdr rowh">${w}</div>`);
    cells[i] = [];
    ENTRIES.forEach((z, j) => {
      const b = document.createElement("button");
      b.className = "cell";
      b.type = "button";
      b.setAttribute("aria-label", `window ${w}, entry ${z}`);
      b.addEventListener("click", () => {
        ctl.window.el.value = w; ctl.entry.el.value = z;
        ctl.window.out.textContent = ctl.window.f(w); ctl.entry.out.textContent = ctl.entry.f(z);
        clearTimeout(btTimer); runBacktest(); markSelected();
        $("demo").scrollIntoView({ behavior: "smooth" });
      });
      b.addEventListener("mousemove", (e) => {
        if (!b._r) return;
        const r = b._r;
        tip.innerHTML = `<b>window ${w}, entry ${z}</b><br>Sharpe (ann.) ${fmt(r.sharpe, 1)}<br>P&amp;L ${money(r.pnl)} · ${fmtInt(r.trades)} fills<br>spread + impact paid ${money(r.execCost)}`;
        tip.style.display = "block";
        tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
        tip.style.top = e.clientY + 14 + "px";
      });
      b.addEventListener("mouseleave", () => (tip.style.display = "none"));
      heat.appendChild(b);
      cells[i][j] = b;
    });
  });
  function markSelected() {
    const w = +ctl.window.el.value, z = +ctl.entry.el.value;
    WINDOWS.forEach((ww, i) => ENTRIES.forEach((zz, j) => cells[i][j].classList.toggle("sel", ww === w && Math.abs(zz - z) < 1e-9)));
  }
  ctl.window.el.addEventListener("input", markSelected);
  ctl.entry.el.addEventListener("input", markSelected);
  let done = 0;
  function runSweep() {
    if (!worker || !S.n) return;
    latestSweep = ++seq;
    done = 0;
    cells.flat().forEach((b) => { b._r = null; b.classList.add("pending"); b.style.background = ""; b.textContent = ""; });
    $("sweep-bar").style.width = "0";
    $("sweep-status").textContent = "Running…";
    worker.postMessage({ cmd: "sweep", id: latestSweep, windows: WINDOWS, entries: ENTRIES, p: params() });
    markSelected();
  }
  $("sweep-run").addEventListener("click", runSweep);
  function sweepCell(m) {
    const b = cells[m.i][m.j];
    b._r = m; // sharpe is annualised by the engine
    b.classList.remove("pending");
    done++;
    $("sweep-bar").style.width = (100 * done) / (WINDOWS.length * ENTRIES.length) + "%";
    $("sweep-status").textContent = `${done} / ${WINDOWS.length * ENTRIES.length} backtests…`;
    paintHeat();
  }
  function paintHeat() {
    // Each side of zero scales to its own extreme (the legend shows both), so
    // one very negative cell does not wash out the differences among the winners.
    const all = cells.flat().filter((b) => b._r).map((b) => b._r.sharpe);
    const hiPos = Math.max(1e-9, ...all.filter((v) => v > 0));
    const hiNeg = Math.max(1e-9, ...all.filter((v) => v < 0).map((v) => -v));
    $("sc-min").textContent = "−" + fmt(hiNeg, 0);
    $("sc-max").textContent = "+" + fmt(hiPos, 0);
    const pos = parseColor(css("--pos")), neg = parseColor(css("--neg")), mid = parseColor(css("--neutral"));
    cells.flat().forEach((b) => {
      if (!b._r) return;
      const v = b._r.sharpe;
      const t = v >= 0 ? Math.min(1, v / hiPos) : -Math.min(1, -v / hiNeg);
      const c = t >= 0 ? mix(mid, pos, t) : mix(mid, neg, -t);
      b.style.background = toCss(c);
      const lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
      b.style.color = lum > 0.55 ? "#0d1424" : "#ffffff";
      b.textContent = b._r.trades ? fmt(b._r.sharpe, 0) : "–";
    });
  }
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { paintHeat(); redrawAll(); });
  $("theme").addEventListener("click", () => setTimeout(paintHeat, 0));

  // ---- benchmark charts (SVG) ----------------------------------------------
  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs, parent) => {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const hover = (node, html) => {
    node.addEventListener("mousemove", (ev) => {
      tip.innerHTML = html; tip.style.display = "block";
      tip.style.left = Math.min(ev.clientX + 12, window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = ev.clientY + 12 + "px";
    });
    node.addEventListener("mouseleave", () => (tip.style.display = "none"));
  };
  const find = (bs, name) => bs.find((b) => b.name === name);
  const roundedBar = (x, y, w, h, r, vertical) =>
    vertical
      ? `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`
      : `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;

  function drawScaling(data) {
    const svg = $("chart-scaling");
    svg.replaceChildren();
    const runs = data.benchmarks
      .filter((b) => b.name.startsWith("BM_Sweep/threads:"))
      .map((b) => ({ t: parseInt(b.name.split("threads:")[1], 10), ms: b.real_time }))
      .sort((a, b) => a.t - b.t);
    if (!runs.length) return;
    runs.forEach((r) => (r.s = runs[0].ms / r.ms));
    const W = 560, H = 260, L = 40, R = 10, T = 18, B = 34;
    const maxT = runs[runs.length - 1].t;
    const maxY = Math.max(maxT, ...runs.map((r) => r.s)) * 1.12;
    const y = (v) => T + (H - T - B) * (1 - v / maxY);
    const band = (W - L - R) / runs.length;
    const xc = (i) => L + band * (i + 0.5);
    for (let g = 0; g <= Math.floor(maxY); g++) {
      el("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), stroke: "var(--grid)" }, svg);
      el("text", { x: L - 8, y: y(g) + 4, "text-anchor": "end", class: "svgtext" }, svg).textContent = g + "×";
    }
    const bw = Math.min(54, band * 0.55);
    runs.forEach((r, i) => {
      const top = y(r.s), h = y(0) - top;
      el("path", { d: roundedBar(xc(i) - bw / 2, top, bw, h, Math.min(4, h / 2), true), fill: "var(--series-1)" }, svg);
      el("text", { x: xc(i), y: top - 6, "text-anchor": "middle", class: "svgtext strong" }, svg).textContent = fmt(r.s, 2) + "×";
      el("text", { x: xc(i), y: H - B + 18, "text-anchor": "middle", class: "svgtext" }, svg).textContent = r.t + (r.t === 1 ? " thread" : " threads");
      const hit = el("rect", { x: xc(i) - band / 2, y: T, width: band, height: H - T - B, fill: "transparent" }, svg);
      hover(hit, `<b>${r.t} thread${r.t > 1 ? "s" : ""}</b><br>${fmt(r.ms, 0)} ms for 32 backtests<br>speed-up ${fmt(r.s, 2)}× (ideal ${r.t}×)`);
    });
    el("polyline", { points: runs.map((r, i) => `${xc(i)},${y(r.t)}`).join(" "), fill: "none", stroke: "var(--ideal)", "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    el("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: "var(--ink-3)" }, svg);
    const last = runs[runs.length - 1];
    $("scaling-caption").textContent = `${maxT} threads: ${fmt(last.s, 2)}× (${fmt((100 * last.s) / last.t, 0)}% efficiency). Bit-identical results at every thread count. On machines whose logical CPUs are hyperthread pairs, scaling flattens past the physical core count.`;
  }

  const PAIRS = [
    { label: "Feature updates", a: ["CRTP", "BM_FeaturesCrtp"], b: ["virtual", "BM_FeaturesVirtual"] },
    { label: "Backtest dispatch", a: ["CRTP", "BM_BacktestCrtp"], b: ["virtual plugin", "BM_BacktestVirtual"] },
    { label: "Replay mode", a: ["direct", "BM_BacktestCrtp"], b: ["SPSC pipeline", "BM_BacktestPipelined/real_time"] },
    { label: "Fork-join, 32k tiny tasks", a: ["work stealing", "BM_PoolTreeStealing/threads:4/real_time"], b: ["one queue", "BM_PoolTreeCentral/threads:4/real_time"] },
    { label: "Sweep, 32 backtests", a: ["work stealing", "BM_Sweep/threads:4/real_time"], b: ["one queue", "BM_SweepCentral/real_time"] },
    { label: "SPSC ring alone", a: ["2 threads", "BM_SpscRingTransfer/real_time"] },
  ];
  function drawThroughput(data) {
    const svg = $("chart-throughput");
    svg.replaceChildren();
    const rows = [];
    PAIRS.forEach((p) => [p.a, p.b].filter(Boolean).forEach((s, j) => {
      const b = find(data.benchmarks, s[1]);
      if (b && b.items_per_second) rows.push({ group: p.label, name: s[0], v: b.items_per_second / 1e6, series: j, first: j === 0, bench: b });
    }));
    // Each group is normalised to its own maximum: the groups measure different things.
    const groupMax = {};
    rows.forEach((r) => (groupMax[r.group] = Math.max(groupMax[r.group] || 0, r.v)));
    const W = 560, L = 150, R = 64, rowH = 22, gap = 12;
    let yPos = 6;
    rows.forEach((r, i) => { if (r.first && i) yPos += gap; r.y = yPos; yPos += rowH; });
    svg.setAttribute("viewBox", `0 0 ${W} ${yPos + 6}`);
    rows.forEach((r) => {
      if (r.first) el("text", { x: 0, y: r.y + 15, class: "svgtext strong" }, svg).textContent = r.group;
      const w = Math.max(3, ((W - L - R) * r.v) / groupMax[r.group]), h = rowH - 7;
      el("text", { x: L - 8, y: r.y + 15, "text-anchor": "end", class: "svgtext" }, svg).textContent = r.name;
      el("path", { d: roundedBar(L, r.y + 3, w, h, Math.min(4, w / 2), false), fill: r.series ? "var(--series-2)" : "var(--series-1)" }, svg);
      el("text", { x: L + w + 6, y: r.y + 15, class: "svgtext" }, svg).textContent = fmt(r.v, r.v < 10 ? 2 : 1) + "M";
      const hit = el("rect", { x: 0, y: r.y, width: W, height: rowH, fill: "transparent" }, svg);
      hover(hit, `<b>${r.group}</b> · ${r.name}<br>${fmt(r.v, 2)} M items/s<br>${fmt(r.bench.real_time, 2)} ${r.bench.time_unit} per iteration`);
    });
  }
  function drawTable(data) {
    const tb = document.querySelector("#bench-table tbody");
    tb.replaceChildren();
    data.benchmarks.forEach((b) => {
      let thr = "";
      if (b.items_per_second) thr = fmt(b.items_per_second / 1e6, 1) + " M items/s";
      if (b.bytes_per_second) thr += (thr ? " · " : "") + fmt(b.bytes_per_second / 1e9, 2) + " GB/s";
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><code>${b.name}</code></td><td class="num">${fmt(b.real_time, b.real_time < 10 ? 2 : 1)} ${b.time_unit}</td><td class="num">${thr || "—"}</td>`;
      tb.appendChild(tr);
    });
  }
  function describe(m) {
    const parts = [`<b>${m.label}</b>`, m.cpu, `${m.num_cpus} logical CPUs`];
    if (m.caches && m.caches.length) parts.push(m.caches.join(", "));
    if (m.compiler) parts.push(m.compiler);
    if (m.commit) parts.push(`commit <code>${m.commit.slice(0, 7)}</code>`);
    if (m.date) parts.push(m.date.slice(0, 10));
    return parts.join(" · ");
  }
  function renderBench(data) {
    $("machine-desc").innerHTML = describe(data.machine);
    drawScaling(data);
    drawThroughput(data);
    drawTable(data);
  }
  Promise.all(["data/bench-ci.json", "data/bench-local.json"].map((f) => fetch(f).then((r) => (r.ok ? r.json() : null)).catch(() => null)))
    .then((sets) => {
      const loaded = sets.filter(Boolean);
      if (!loaded.length) { $("machine-desc").textContent = "Benchmark data not available."; return; }
      const tabs = $("machine-tabs");
      loaded.forEach((d, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = d.machine.label;
        b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
        b.addEventListener("click", () => {
          tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
          b.setAttribute("aria-pressed", "true");
          renderBench(d);
        });
        tabs.appendChild(b);
      });
      renderBench(loaded[0]);
      const ci = loaded[0].benchmarks;
      const bt = find(ci, "BM_BacktestCrtp");
      if (bt) $("k-native").textContent = fmt(bt.items_per_second / 1e6, 1) + "M ticks/s";
      const ws = find(ci, "BM_PoolTreeStealing/threads:4/real_time"), cq = find(ci, "BM_PoolTreeCentral/threads:4/real_time");
      $("k-stealing").textContent = ws && cq ? fmt(ws.items_per_second / cq.items_per_second, 1) + "× faster" : "—";
    });
})();
