#!/usr/bin/env python3
"""StarryVector validation harness.

Runs the compiled `starry_bench` binary over a matrix of configurations,
collects the JSON results and renders a self-contained HTML report that
can be opened directly in any browser (charts are embedded as base64
PNGs, so the file works offline and can be shared as a single artifact).

Usage (from anywhere):
    python3 validation/run_validation.py            # full matrix
    python3 validation/run_validation.py --quick    # smoke-sized matrix
    python3 validation/run_validation.py --open     # open report in browser

Outputs are written to validation/out/:
    report.html     the human-readable report
    results.json    the raw collected benchmark results
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_DIR.parent
DEFAULT_BENCH = REPO_ROOT / "build" / "bin" / "starry_bench"
OUT_DIR = VALIDATION_DIR / "out"

# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------
# Each case is (label, dict-of-bench-flags).  The full matrix is kept small
# enough to finish in a couple of minutes on a laptop; the quick matrix is
# for smoke-testing the harness itself.


def build_matrix(quick: bool) -> list[dict]:
    cases: list[dict] = []

    def add(label: str, **flags) -> None:
        base = {"dim": 128, "n": 100_000, "k": 10, "metric": "l2",
                "queries": 200, "threads": 1, "seed": 42}
        base.update(flags)
        cases.append({"label": label, "flags": base})

    if quick:
        for n in (10_000, 100_000):
            add(f"scale-n={n}", n=n)
        for threads in (1, 4):
            add(f"threads={threads}", n=50_000, threads=threads, queries=100)
        add("metric=cosine", metric="cosine", n=50_000, queries=100)
        return cases

    # 1. Dataset size scaling (single thread) - shows O(n) scan cost.
    for n in (10_000, 50_000, 100_000, 500_000, 1_000_000):
        add(f"scale-n={n}", n=n, queries=100 if n >= 500_000 else 200)

    # 2. Thread scaling (fixed dataset) - shows read-path parallelism.
    for threads in (1, 2, 4, 8):
        add(f"threads={threads}", n=200_000, threads=threads, queries=200)

    # 3. Dimension scaling - RAG embeddings are commonly 64..1536-d.
    for dim in (64, 128, 256, 768):
        add(f"dim={dim}", dim=dim, queries=100)

    # 4. Metric comparison at a fixed size.
    for metric in ("l2", "ip", "cosine"):
        add(f"metric={metric}", metric=metric, queries=200)

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_case(bench: Path, case: dict) -> dict:
    """Run one benchmark case and merge its JSON into the case record."""
    cmd = [str(bench)]
    for key, value in case["flags"].items():
        cmd += [f"--{key}", str(value)]
    print(f"  running {case['label']:<22} -> {' '.join(cmd[1:])}",
          flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"bench failed for case {case['label']}")
    result = json.loads(proc.stdout)
    return {"label": case["label"], "flags": case["flags"], "result": result}


def collect_machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_model": "unknown",
        "mem_total": "unknown",
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                kb = int(line.split()[1])
                info["mem_total"] = f"{kb / 1024 / 1024:.1f} GiB"
                break
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------
# Charts (matplotlib, Agg backend, embedded as base64)
# ---------------------------------------------------------------------------


def make_charts(runs: list[dict]) -> list[tuple[str, str]]:
    """Return a list of (title, base64-png) chart sections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts: list[tuple[str, str]] = []

    def render(fig, title: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        png = base64.b64encode(buf.getvalue()).decode()
        charts.append((title, png))

    def pick(prefix: str) -> list[dict]:
        return [r for r in runs if r["label"].startswith(prefix)]

    # -- 1. thread scaling -------------------------------------------------
    ts = pick("threads=")
    if ts:
        xs = [r["flags"]["threads"] for r in ts]
        qps = [r["result"]["search"]["qps"] for r in ts]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(xs, qps, "o-", label="measured QPS")
        ax.plot(xs, [qps[0] * t / xs[0] for t in xs], "--",
                label="ideal linear scaling")
        ax.set_xlabel("worker threads")
        ax.set_ylabel("queries / second")
        ax.set_title("Thread scaling (n=200k, dim=128, L2)")
        ax.legend()
        ax.grid(alpha=0.3)
        render(fig, "Thread scaling")

    # -- 2. dataset size scaling --------------------------------------------
    ns = pick("scale-n=")
    if ns:
        xs = [r["flags"]["n"] for r in ns]
        p50 = [r["result"]["search"]["p50_us"] / 1000.0 for r in ns]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(xs, p50, "s-", color="tab:red")
        ax.set_xscale("log")
        ax.set_xlabel("vectors in index (log scale)")
        ax.set_ylabel("p50 latency per query [ms]")
        ax.set_title("Query latency vs dataset size (1 thread, exact scan)")
        ax.grid(alpha=0.3, which="both")
        render(fig, "Dataset size scaling")

    # -- 3. dimension scaling -------------------------------------------------
    ds = pick("dim=")
    if ds:
        xs = [str(r["flags"]["dim"]) for r in ds]
        qps = [r["result"]["search"]["qps"] for r in ds]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(xs, qps, color="tab:purple")
        ax.set_xlabel("vector dimension")
        ax.set_ylabel("queries / second")
        ax.set_title("Throughput vs dimension (n=100k, 1 thread)")
        ax.grid(alpha=0.3, axis="y")
        render(fig, "Dimension scaling")

    # -- 4. metric comparison ---------------------------------------------------
    ms = pick("metric=")
    if ms:
        xs = [r["flags"]["metric"] for r in ms]
        qps = [r["result"]["search"]["qps"] for r in ms]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(xs, qps, color="tab:green")
        ax.set_xlabel("metric")
        ax.set_ylabel("queries / second")
        ax.set_title("Throughput by metric (n=100k, dim=128, 1 thread)")
        ax.grid(alpha=0.3, axis="y")
        render(fig, "Metric comparison")

    # -- 5. memory bandwidth -----------------------------------------------------
    if ts:
        gbps = [r["result"]["scan_gbps"] for r in ts]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar([str(r["flags"]["threads"]) for r in ts], gbps,
               color="tab:orange")
        ax.set_xlabel("worker threads")
        ax.set_ylabel("vector bytes scanned [GB/s]")
        ax.set_title("Effective scan bandwidth while searching")
        ax.grid(alpha=0.3, axis="y")
        render(fig, "Scan bandwidth")

    return charts


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>StarryVector validation report</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f5f6f8; color: #222; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 26px; } h2 { font-size: 20px; margin-top: 40px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
  .card { background: #fff; border-radius: 10px; padding: 14px 18px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); min-width: 150px; }
  .card .k { font-size: 12px; color: #777; text-transform: uppercase; }
  .card .v { font-size: 20px; font-weight: 600; margin-top: 2px; }
  .chart { background: #fff; border-radius: 10px; padding: 12px;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); margin: 14px 0; }
  .chart img { width: 100%; height: auto; display: block; }
  table { border-collapse: collapse; width: 100%; background: #fff;
          border-radius: 10px; overflow: hidden; font-size: 13px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  th, td { padding: 7px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { background: #2c3e50; color: #fff; font-weight: 600; }
  tr:nth-child(even) td { background: #f0f2f5; }
  .note { color: #666; font-size: 12px; margin-top: 6px; }
</style>
</head>
<body><div class="wrap">
"""

HTML_TAIL = "</div></body></html>\n"


def render_html(runs: list[dict], machine: dict, charts: list) -> str:
    parts = [HTML_HEAD]
    parts.append("<h1>StarryVector &mdash; validation report</h1>")
    parts.append(
        f'<p class="note">Generated {datetime.datetime.now().isoformat(timespec="seconds")} '
        f"&middot; exact brute-force index (recall = 100% by construction)</p>")

    best_qps = max((r["result"]["search"]["qps"] for r in runs), default=0)
    peak_gbps = max((r["result"]["scan_gbps"] for r in runs), default=0)
    fastest_build = max((r["result"]["build"]["vectors_per_second"]
                         for r in runs), default=0)
    total_vec = max((r["result"]["config"]["n"] for r in runs), default=0)

    parts.append("<div class='cards'>")
    for key, value in (
            ("peak QPS", f"{best_qps:,.0f}"),
            ("peak scan bandwidth", f"{peak_gbps:.1f} GB/s"),
            ("peak build rate", f"{fastest_build:,.0f} vec/s"),
            ("largest dataset", f"{total_vec:,} vectors"),
            ("cpu", machine["cpu_model"]),
            ("cores / memory", f"{machine['cpu_count']} / {machine['mem_total']}")):
        parts.append(f"<div class='card'><div class='k'>{key}</div>"
                     f"<div class='v'>{value}</div></div>")
    parts.append("</div>")

    parts.append("<h2>Charts</h2>")
    for title, png in charts:
        parts.append(f"<div class='chart'><img "
                     f"src='data:image/png;base64,{png}' alt='{title}'></div>")

    parts.append("<h2>All runs</h2><table>")
    parts.append("<tr><th>case</th><th>dim</th><th>n</th><th>metric</th>"
                 "<th>threads</th><th>build s</th><th>QPS</th>"
                 "<th>p50 &micro;s</th><th>p95 &micro;s</th><th>p99 &micro;s</th>"
                 "<th>GB/s</th></tr>")
    for r in runs:
        c, s = r["result"]["config"], r["result"]["search"]
        parts.append(
            f"<tr><td>{r['label']}</td><td>{c['dim']}</td><td>{c['n']:,}</td>"
            f"<td>{c['metric']}</td><td>{c['threads']}</td>"
            f"<td>{r['result']['build']['seconds']:.2f}</td>"
            f"<td>{s['qps']:,.0f}</td><td>{s['p50_us']:,.0f}</td>"
            f"<td>{s['p95_us']:,.0f}</td><td>{s['p99_us']:,.0f}</td>"
            f"<td>{r['result']['scan_gbps']:.1f}</td></tr>")
    parts.append("</table>")
    parts.append(
        "<p class='note'>Raw JSON: validation/out/results.json &middot; "
        f"host: {machine['platform']}</p>")

    parts.append(HTML_TAIL)
    return "".join(parts)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH,
                        help="path to the starry_bench binary")
    parser.add_argument("--quick", action="store_true",
                        help="run a reduced smoke matrix")
    parser.add_argument("--open", action="store_true",
                        help="open the report in the default browser")
    args = parser.parse_args()

    if not args.bench.is_file():
        raise SystemExit(
            f"bench binary not found: {args.bench}\n"
            f"build it first:  cmake -B build -DCMAKE_BUILD_TYPE=Release "
            f"&& cmake --build build -j")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    machine = collect_machine_info()
    print(f"host: {machine['cpu_model']} ({machine['cpu_count']} cores, "
          f"{machine['mem_total']})")

    matrix = build_matrix(args.quick)
    print(f"running {len(matrix)} benchmark cases ...")
    runs = [run_case(args.bench, case) for case in matrix]

    (OUT_DIR / "results.json").write_text(json.dumps(
        {"machine": machine, "runs": runs}, indent=2))

    charts = make_charts(runs)
    report_path = OUT_DIR / "report.html"
    report_path.write_text(render_html(runs, machine, charts))

    print(f"\nreport : {report_path}")
    print(f"raw    : {OUT_DIR / 'results.json'}")
    if args.open:
        webbrowser.open(report_path.as_uri())


if __name__ == "__main__":
    main()
