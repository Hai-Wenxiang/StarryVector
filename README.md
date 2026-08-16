# StarryVector

A high-performance **vector database for RAG (Retrieval-Augmented Generation)**, written in modern C++.

StarryVector provides efficient storage, indexing, and similarity search for high-dimensional embedding vectors, designed to serve as the retrieval backbone of RAG pipelines.

## Features

- **ANN Indexes** — HNSW graph index and IVF (inverted file) index for approximate nearest neighbor search
- **Exact Search** — brute-force kNN with SIMD-accelerated (AVX2/AVX-512/NEON) distance computations (L2, inner product, cosine)
- **Metadata Filtering** — combine vector search with scalar attribute filters
- **Persistence** — memory-mapped (mmap) storage format with incremental snapshots
- **Simple API** — clean C++ interface plus a lightweight HTTP server mode

## Roadmap

- [ ] Core vector storage & exact kNN search
- [ ] SIMD distance kernels
- [ ] HNSW index
- [ ] IVF index with k-means clustering
- [ ] Metadata filtering
- [ ] Persistence & snapshots
- [ ] HTTP server (REST) interface
- [ ] Python client bindings

## Building

Requirements:

- C++20 compliant compiler (GCC 11+, Clang 14+, or MSVC 19.30+)
- CMake 3.20+

```bash
git clone https://github.com/Hai-Wenxiang/StarryVector.git
cd StarryVector
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Quick Start

```cpp
#include <starryvector/db.hpp>

int main() {
    starry::VectorDB db(768);  // dimension of embeddings

    db.insert(1, {0.1f, 0.2f, /* ... */});
    db.insert(2, {0.3f, 0.4f, /* ... */});

    auto results = db.search({0.1f, 0.2f, /* ... */}, /*k=*/10);
    for (auto& [id, distance] : results) {
        // top-k nearest neighbors
    }
}
```

## Project Layout

```
StarryVector/
├── include/    # Public headers
├── src/        # Library implementation
├── tests/      # Unit tests
├── benchmarks/ # Performance benchmarks
└── examples/   # Usage examples
```

## License

TBD
