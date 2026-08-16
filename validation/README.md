# Validation harness

Runs `starry_bench` over a benchmark matrix and produces a single
self-contained HTML report (charts embedded as base64 PNGs — just open it
in any browser, no server needed).

## Usage

From the repository root:

```bash
# 1. build the core and the bench driver (Release!)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. run unit tests
ctest --test-dir build --output-on-failure

# 3. generate the report
python3 validation/run_validation.py          # full matrix, ~2 min
python3 validation/run_validation.py --quick  # smoke matrix, ~10 s
python3 validation/run_validation.py --open   # open in browser afterwards
```

Outputs land in `validation/out/` (git-ignored):

- `report.html` — human-readable report with charts and a full result table
- `results.json` — raw machine-readable results for tooling / trend tracking

## What is measured

| Case family | What it tells you |
|---|---|
| `scale-n=*` | query latency grows linearly with dataset size (exact scan) |
| `threads=*` | read-path parallelism; compare against ideal linear scaling |
| `dim=*` | cost per dimension; 768-d is the common RAG embedding size |
| `metric=*` | relative cost of L2 / inner product / cosine kernels |

The M0 index is brute force, so recall is **100% by construction** — the
report focuses on throughput (QPS), latency percentiles (p50/p95/p99) and
effective scan bandwidth (GB/s). Once approximate indexes (HNSW, IVF)
land, recall-vs-QPS Pareto curves will be added here.

## Requirements

- Python 3 with `matplotlib` (Debian/Ubuntu: `sudo apt install python3-matplotlib`)
