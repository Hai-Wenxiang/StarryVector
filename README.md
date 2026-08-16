# StarryVector

A high-performance **vector database for RAG (Retrieval-Augmented Generation)** for **Linux**, written in portable **C++17** (GCC 7+, Clang 5+). Windows is not a supported platform.

StarryVector provides efficient storage, indexing, and similarity search for high-dimensional embedding vectors, designed to serve as the retrieval backbone of RAG pipelines.

## Features

- **Exact Search (available now)** — brute-force kNN with scalar distance kernels (L2, inner product, cosine), 100% recall by construction. Cosine vectors are L2-normalised on insert so searches reduce to inner products (~2x faster).
- **Persistence** — single-file `save()` / `load()` with tombstone-aware soft deletion
- **ANN Indexes** (planned) — HNSW graph index and IVF with quantization, validated against the exact core
- **Metadata Filtering** (planned) — combine vector search with scalar attribute filters
- **Simple API** — clean C++17 interface; status codes instead of exceptions

## Roadmap

- [x] Core vector storage & exact kNN search
- [ ] SIMD distance kernels
- [ ] HNSW index
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

## Quick Start

```cpp
#include <starry/db.hpp>

int main() {
    starry::VectorDB db(768);  // dimension of embeddings

    db.insert(1, {0.1f, 0.2f, /* ... */});   // returns starry::kOk
    db.insert(2, {0.3f, 0.4f, /* ... */});

    auto results = db.search({0.1f, 0.2f, /* ... */}, /*k=*/10);
    for (const auto& hit : results) {
        // hit.id, hit.distance — top-k nearest neighbors
    }

    db.save("myvectors.bin");                // single-file persistence
}
```

## Benchmarks & Validation

```bash
python3 validation/run_validation.py --open
```

Runs a benchmark matrix (dataset size / threads / dimensions / metrics)
and opens a self-contained HTML report. Requires `python3-matplotlib`.
See [validation/README.md](validation/README.md).

## Project Layout

```
StarryVector/
├── include/starry/   # Public headers (types, distance, flat_index, db)
├── src/              # Library implementation (zero dependencies)
├── apps/             # Benchmark driver (starry_bench)
├── examples/         # RAG retrieval demo (starry_rag_demo)
├── tests/            # Unit tests (doctest, vendored)
├── validation/       # Python harness -> HTML performance report
└── third_party/      # Vendored dev-only dependencies (doctest)
```

## License

TBD
