#!/usr/bin/env python3
"""Convert Google Benchmark JSON into the compact file the Pages site reads.

    bench_to_site.py <benchmark.json> <out.json> --label "GitHub Actions ubuntu-24.04"

The machine description comes from the benchmark's own context block plus
/proc/cpuinfo when available, so every chart on the site names the hardware
it was measured on.
"""

import argparse
import datetime as dt
import json
import platform


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--label", required=True)
    ap.add_argument("--compiler", default="")
    ap.add_argument("--commit", default="")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        raw = json.load(f)
    ctx = raw.get("context", {})
    caches = [
        f"L{c['level']} {c['type'].lower()} {c['size'] // 1024} KiB" + (f" x{c['num_sharing']}" if c.get("num_sharing", 0) > 1 else "")
        for c in ctx.get("caches", [])
    ]
    out = {
        "machine": {
            "label": args.label,
            "cpu": cpu_model(),
            "num_cpus": ctx.get("num_cpus"),
            "mhz_per_cpu": ctx.get("mhz_per_cpu"),
            "caches": caches,
            "build_type": ctx.get("library_build_type"),
            "compiler": args.compiler,
            "commit": args.commit,
            "date": ctx.get("date") or dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "benchmarks": [
            {
                "name": b["name"],
                "real_time": b["real_time"],
                "cpu_time": b["cpu_time"],
                "time_unit": b["time_unit"],
                "items_per_second": b.get("items_per_second"),
                "bytes_per_second": b.get("bytes_per_second"),
            }
            for b in raw["benchmarks"]
            if b.get("run_type", "iteration") == "iteration"
        ],
    }
    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
