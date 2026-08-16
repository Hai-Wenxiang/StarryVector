# Contributing to StarryVector

Thanks for your interest in improving StarryVector! Issues and pull
requests are both welcome.

## Getting started

1. Fork the repository and create your branch from `master`:

   ```bash
   git clone https://github.com/<your-fork>/StarryVector.git
   cd StarryVector
   git checkout -b feat/my-feature
   ```

2. Build and run the tests:

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```

3. Make your change. Please keep the code C++17 and Linux-only.

## What we look for in pull requests

- **Tests first.** New kernels/indexes/features must come with unit
  tests. The strongest pattern in this codebase is the *oracle
  comparison*: validate results against the naive reference
  implementation (see `tests/test_core.cpp`).
- **No regressions.** `ctest` must pass. If you touch a hot path,
  include before/after numbers from `starry_bench` (single thread,
  pinned CPU) in the PR description.
- **Zero dependencies in the core.** `src/` and `include/starry/` must
  only use the C++ standard library.
- **Commit style.** Conventional-commit-ish one-liners:
  `feat: ...`, `fix: ...`, `perf: ...`, `docs: ...`, `chore: ...`.

## Reporting bugs

Open an issue with:

- StarryVector version / commit
- Compiler and flags
- Minimal reproduction (code + data shape: n, dim, metric, index kind)
- Expected vs actual behaviour

## Reporting performance claims

Performance numbers are only meaningful with the recall attached. When
reporting benchmarks, include the full command line, hardware info and
the recall@k at which QPS was measured.

## Branch model

`master` is protected: all changes land through pull requests. The repo
owner reviews and merges.
