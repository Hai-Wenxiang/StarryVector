# StarryVector

[![CI](https://github.com/Hai-Wenxiang/StarryVector/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/StarryVector/actions/workflows/ci.yml)

**English** | [简体中文](README.zh-CN.md)

A high-performance **vector database for RAG (Retrieval-Augmented Generation)** for **Linux**, written in portable **C++17** (GCC 7+, Clang 5+). Windows is not a supported platform.

StarryVector provides efficient storage, indexing, and similarity search for high-dimensional embedding vectors, designed to serve as the retrieval backbone of RAG pipelines.

## Why StarryVector

- **Fast on CPU** — AVX2 distance kernels selected at runtime (with a scalar fallback), cosine search reduced to inner products via insert-time normalisation, and an HNSW graph index that searches 6x+ faster than a brute-force scan at 99.5% recall.
- **One binary, any x86 machine** — SIMD is dispatched at runtime (`__builtin_cpu_supports`), so a single build runs everywhere from old servers to new desktops. Set `STARRY_FORCE_SCALAR=1` to force the scalar path.
- **Real deletion** — tombstone-based soft delete works with both indexes; deleted rows disappear from searches immediately.
- **Single-file persistence** — `save()` / `load()` with a versioned, little-endian format (`STARRYV2` stores the HNSW graph; older `STARRYV1` flat files still load).
- **Zero dependencies** — the core library uses only the C++ standard library. The single vendored header (doctest) is a dev-only test dependency.
- **Simple API** — status codes instead of exceptions; RAII throughout.

## Features

- **Exact Search** — brute-force kNN with SIMD distance kernels (L2, inner product, cosine), 100% recall by construction. O(n log k) bounded-heap top-k selection.
- **HNSW ANN Index** — hierarchical navigable small-world graph (Malkov & Yashunin, 2016) with incremental inserts, diversity-based neighbour selection and a tunable `ef` search beam. Deterministic builds (fixed RNG seed) make indexes reproducible.
- **Persistence** — single-file `save()` / `load()` with tombstone-aware soft deletion.
- **IVF + Quantization** (planned) — second engine with memory-lean scans, validated against the exact core.
- **Metadata Filtering** (planned) — combine vector search with scalar attribute filters.

## Roadmap

- [x] Core vector storage & exact kNN search
- [x] SIMD distance kernels (AVX2, runtime dispatch, `STARRY_FORCE_SCALAR=1` to disable)
- [x] HNSW index
- [ ] IVF index with k-means clustering
- [ ] Metadata filtering
- [ ] WAL + mmap storage engine
- [ ] HTTP server (REST) interface
- [ ] Python client bindings

## Building

Requirements:

- Linux with a C++17 compliant compiler (GCC 7+ or Clang 5+)
- CMake 3.10+

```bash
git clone https://github.com/Hai-Wenxiang/StarryVector.git
cd StarryVector
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # unit tests
```

Build options:

| Option | Default | Description |
|---|---|---|
| `STARRY_DISABLE_SIMD` | `OFF` | Skip AVX2 kernels entirely (scalar only) |

### Install & consume

Install the library and headers (default prefix `/usr/local`, override with `--prefix`; on CMake < 3.15 use `make install`):

```bash
cmake --install build --prefix /path/to/install
```

Use from another CMake project (after install via `find_package`, or add this repository with `add_subdirectory` and link `starry_core` directly):

```cmake
find_package(StarryVector 0.1 REQUIRED)   # version policy: same major
target_link_libraries(my_app PRIVATE StarryVector::starry_core)
```

## Quick Start

```cpp
#include <starry/db.hpp>

int main() {
    // Exact brute-force database (100% recall)
    starry::VectorDB db(768);  // dimension of embeddings

    db.insert(1, {0.1f, 0.2f, /* ... */});   // returns starry::kOk
    db.insert(2, {0.3f, 0.4f, /* ... */});

    auto results = db.search({0.1f, 0.2f, /* ... */}, /*k=*/10);
    for (const auto& hit : results) {
        // hit.id, hit.distance — top-k nearest neighbors
    }

    db.remove(2);                             // soft delete
    db.save("myvectors.bin");                 // single-file persistence
}
```

Approximate search with the HNSW index:

```cpp
starry::VectorDB db(768, starry::kL2, starry::kHnswIndex);
db.set_search_ef(128);   // larger = higher recall, slower

for (/* every document chunk */) {
    db.insert(chunk_id, embedding);
}
auto hits = db.search(query_embedding, 10);   // approximate top-k
```

## Benchmarks & Validation

```bash
# Exact search benchmark (JSON report on stdout)
./build/bin/starry_bench --n 100000 --dim 128 --metric l2 --threads 1

# HNSW benchmark with recall against the exact core
./build/bin/starry_bench --index hnsw --ef 64 --dim 16 --n 100000

# Full benchmark matrix -> HTML report
python3 validation/run_validation.py --open
```

Reference numbers (single core, pinned, i5-13400, n=100k dim=128, see `notes/`
for methodology):

| Workload | QPS | Latency p50 | Recall@10 |
|---|---|---|---|
| Flat + AVX2 (L2) | ~390 | ~2.5 ms | 100% |
| Flat + AVX2 (cosine) | ~390 | ~2.5 ms | 100% |
| HNSW ef=64 (dim=16 clustered) | ~15,000 | ~60 us | 99.6% |

> Note: recall depends on data intrinsic dimensionality. High-dimensional
> *uniform random* synthetic data is the worst case for graph indexes; real
> embedding data behaves far better.

See [validation/README.md](validation/README.md) for the harness details.

## Project Layout

```
StarryVector/
├── include/starry/   # Public headers (types, distance, flat_index, hnsw_index, db)
├── src/              # Library implementation (zero dependencies)
├── apps/             # Benchmark driver (starry_bench)
├── examples/         # RAG retrieval demo (starry_rag_demo)
├── tests/            # Unit tests (doctest, vendored)
├── validation/       # Python harness -> HTML performance report
└── third_party/      # Vendored dev-only dependencies (doctest)
```

## Contributing

Issues and pull requests are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
