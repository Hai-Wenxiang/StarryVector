// starry_bench - synthetic-data benchmark driver for the StarryVector core.
//
// Generates a reproducible random dataset (fixed-seed mt19937), builds a
// VectorDB, then measures:
//
//   * build throughput      (inserts per second)
//   * search throughput     (queries per second, wall clock over all
//                            worker threads together)
//   * per-query latency     (p50 / p95 / p99 / min / max, microseconds)
//   * scan bandwidth        (GB/s of vector bytes touched per second)
//
// Search is brute force and therefore exact: recall is 100% by
// construction; the JSON output states "exact": true so downstream
// tooling never mistakes this for an approximate index.
//
// The single JSON object with all results is printed to STDOUT (so it can
// be piped/parsed by the validation harness); human progress messages go
// to STDERR.
//
// Usage:
//   starry_bench [--dim N] [--n N] [--k N] [--metric l2|ip|cosine]
//                [--index flat|hnsw] [--build serial|bulk]
//                [--queries N] [--threads N] [--seed N] [--ef N]
//
// Defaults: dim=128 n=100000 k=10 metric=l2 queries=200 threads=1
//           seed=42 build=serial
//
// --build bulk feeds the dataset through VectorDB::insert_bulk() with
// --threads workers (deterministic parallel HNSW construction); serial
// keeps the classic row-by-row insert() path for comparability.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "starry/db.hpp"

namespace {

// ---- tiny helpers ---------------------------------------------------------

typedef std::chrono::steady_clock Clock;

double seconds_between(Clock::time_point t0, Clock::time_point t1) {
  return std::chrono::duration<double>(t1 - t0).count();
}

// Percentile of an ALREADY SORTED sample.  p is in [0, 100].
double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) {
    return 0.0;
  }
  std::size_t idx = static_cast<std::size_t>(
      p / 100.0 * static_cast<double>(sorted.size() - 1) + 0.5);
  if (idx >= sorted.size()) {
    idx = sorted.size() - 1;
  }
  return sorted[idx];
}

std::string json_num(const char* key, double v) {
  std::ostringstream o;
  o << "    \"" << key << "\": " << std::fixed << std::setprecision(3) << v;
  return o.str();
}

// ---- configuration --------------------------------------------------------

struct Config {
  std::size_t dim = 128;
  std::size_t n = 100000;
  std::size_t k = 10;
  starry::Metric metric = starry::kL2;
  std::size_t queries = 200;
  std::size_t threads = 1;
  std::uint32_t seed = 42;
  std::string index = "flat";  // flat | hnsw
  std::string build = "serial";  // serial | bulk
  std::size_t ef = 0;          // 0 -> library default
};

bool parse_args(int argc, char** argv, Config* cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = (i + 1 < argc);
    if (arg == "--index" || arg == "--ef") {
      // flags with values, validated below
    } else if (!has_value) {
      std::cerr << "missing value for " << arg << "\n";
      return false;
    }
    const char* val = has_value ? argv[++i] : "";
    if (arg == "--dim") cfg->dim = std::strtoul(val, 0, 10);
    else if (arg == "--n") cfg->n = std::strtoul(val, 0, 10);
    else if (arg == "--k") cfg->k = std::strtoul(val, 0, 10);
    else if (arg == "--queries") cfg->queries = std::strtoul(val, 0, 10);
    else if (arg == "--threads") cfg->threads = std::strtoul(val, 0, 10);
    else if (arg == "--seed") cfg->seed = static_cast<std::uint32_t>(std::strtoul(val, 0, 10));
    else if (arg == "--index") cfg->index = val;
    else if (arg == "--build") cfg->build = val;
    else if (arg == "--ef") cfg->ef = std::strtoul(val, 0, 10);
    else if (arg == "--metric") { if (!starry::parse_metric(val, &cfg->metric)) { std::cerr << "unknown metric: " << val << "\n"; return false; } }
    else { std::cerr << "unknown flag: " << arg << "\n"; return false; }
  }
  return cfg->dim > 0 && cfg->n > 0 && cfg->k > 0 && cfg->queries > 0 &&
         cfg->threads > 0 && (cfg->index == "flat" || cfg->index == "hnsw") &&
         (cfg->build == "serial" || cfg->build == "bulk");
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  if (!parse_args(argc, argv, &cfg)) {
    std::cerr << "usage: starry_bench [--dim N] [--n N] [--k N] "
                 "[--metric l2|ip|cosine] [--index flat|hnsw] "
                 "[--build serial|bulk] [--ef N] [--queries N] "
                 "[--threads N] [--seed N]\n";
    return 1;
  }

  // Reproducible synthetic data: components uniform in [-1, 1).  The
  // exact distribution matters little for a brute-force index; what
  // matters is that the same seed yields byte-identical datasets across
  // runs, machines and compilers.
  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

  const starry::IndexKind kind = cfg.index == "hnsw"
      ? starry::kHnswIndex : starry::kFlatIndex;
  starry::VectorDB db(cfg.dim, cfg.metric, kind);
  db.set_search_ef(cfg.ef);

  // -- build phase ---------------------------------------------------------
  std::cerr << "[bench] building " << cfg.index << " index (" << cfg.build
            << "): n=" << cfg.n << " dim=" << cfg.dim
            << " metric=" << starry::metric_name(cfg.metric) << "\n";
  const Clock::time_point build_t0 = Clock::now();
  if (cfg.build == "bulk") {
    // Pregenerate the whole dataset (same generator order as the serial
    // path, so both modes compare identical data), then one bulk call.
    std::vector<float> data(cfg.n * cfg.dim);
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i] = uniform(rng);
    }
    std::vector<starry::id_t> ids(cfg.n);
    for (std::size_t i = 0; i < cfg.n; ++i) {
      ids[i] = static_cast<starry::id_t>(i);
    }
    if (db.insert_bulk(ids, data, cfg.threads) != starry::kOk) {
      std::cerr << "insert_bulk failed\n";
      return 1;
    }
  } else {
    std::vector<float> buf(cfg.dim);
    for (std::size_t i = 0; i < cfg.n; ++i) {
      for (std::size_t j = 0; j < cfg.dim; ++j) {
        buf[j] = uniform(rng);
      }
      if (db.insert(static_cast<starry::id_t>(i), &buf[0]) != starry::kOk) {
        std::cerr << "insert failed at " << i << "\n";
        return 1;
      }
    }
  }
  const Clock::time_point build_t1 = Clock::now();
  const double build_s = seconds_between(build_t0, build_t1);

  // -- query generation ----------------------------------------------------
  std::vector<float> queries(cfg.queries * cfg.dim);
  for (std::size_t i = 0; i < queries.size(); ++i) {
    queries[i] = uniform(rng);
  }

  // Ground truth for recall: an independent exact FlatIndex scan (also
  // the fairness oracle - hnsw numbers below are meaningless without it).
  double recall = 1.0;
  if (kind == starry::kHnswIndex) {
    std::cerr << "[bench] computing exact ground truth for recall...\n";
    starry::FlatIndex truth(cfg.dim, cfg.metric);
    std::vector<float> data(cfg.n * cfg.dim);
    // Re-generate the dataset deterministically (same seed, same order).
    std::mt19937 rng2(cfg.seed);
    for (std::size_t i = 0; i < cfg.n * cfg.dim; ++i) {
      data[i] = uniform(rng2);
    }
    for (std::size_t i = 0; i < cfg.n; ++i) {
      truth.add(static_cast<starry::id_t>(i), &data[i * cfg.dim]);
    }
    double hits = 0.0;
    double total = 0.0;
    for (std::size_t q = 0; q < cfg.queries; ++q) {
      const std::vector<starry::SearchResult> want =
          truth.search(&queries[q * cfg.dim], cfg.k, 0);
      const std::vector<starry::SearchResult> got =
          db.search(&queries[q * cfg.dim], cfg.k);
      std::unordered_set<starry::id_t> want_ids;
      for (std::size_t i = 0; i < want.size(); ++i) {
        want_ids.insert(want[i].id);
      }
      for (std::size_t i = 0; i < got.size(); ++i) {
        if (want_ids.count(got[i].id) != 0) {
          hits += 1.0;
        }
      }
      total += static_cast<double>(cfg.k);
    }
    recall = total > 0.0 ? hits / total : 0.0;
  }

  // -- warm up caches/JIT-free code paths with a few discarded queries ----
  for (std::size_t i = 0; i < 5 && i < cfg.queries; ++i) {
    db.search(&queries[i * cfg.dim], cfg.k);
  }

  // -- measured search phase ------------------------------------------------
  // Workers pull query indices from an atomic counter (self-scheduling);
  // each records its own latencies locally, merged after the join, so no
  // lock is taken on the hot path.
  std::cerr << "[bench] searching: queries=" << cfg.queries
            << " threads=" << cfg.threads << "\n";
  std::atomic<std::size_t> next_query(0);
  std::vector<std::vector<double> > latency_us(cfg.threads);

  const Clock::time_point search_t0 = Clock::now();
  std::vector<std::thread> workers;
  for (std::size_t w = 0; w < cfg.threads; ++w) {
    workers.push_back(std::thread([&cfg, &queries, &db, &next_query,
                                   &latency_us, w]() {
      std::vector<double>& mine = latency_us[w];
      for (;;) {
        const std::size_t q = next_query.fetch_add(1);
        if (q >= cfg.queries) {
          break;
        }
        const Clock::time_point t0 = Clock::now();
        db.search(&queries[q * cfg.dim], cfg.k);
        const Clock::time_point t1 = Clock::now();
        mine.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
    }));
  }
  for (std::size_t w = 0; w < workers.size(); ++w) {
    workers[w].join();
  }
  const Clock::time_point search_t1 = Clock::now();
  const double search_s = seconds_between(search_t0, search_t1);

  // -- aggregate ------------------------------------------------------------
  std::vector<double> all_us;
  for (std::size_t w = 0; w < latency_us.size(); ++w) {
    all_us.insert(all_us.end(), latency_us[w].begin(), latency_us[w].end());
  }
  std::sort(all_us.begin(), all_us.end());

  double sum_us = 0.0;
  for (std::size_t i = 0; i < all_us.size(); ++i) sum_us += all_us[i];
  const double avg_us = all_us.empty() ? 0.0 : sum_us / all_us.size();
  const double qps = search_s > 0.0
      ? static_cast<double>(cfg.queries) / search_s : 0.0;
  // One query linearly scans n * dim floats + n * 8 bytes of ids.
  const double bytes_per_query =
      static_cast<double>(cfg.n) * cfg.dim * 4.0 +
      static_cast<double>(cfg.n) * 8.0;
  const double gbps = qps * bytes_per_query / 1e9;

  // -- JSON report (stdout) --------------------------------------------------
  std::cout << "{\n";
  std::cout << "  \"config\": {\n";
  std::cout << "    \"dim\": " << cfg.dim << ",\n";
  std::cout << "    \"n\": " << cfg.n << ",\n";
  std::cout << "    \"k\": " << cfg.k << ",\n";
  std::cout << "    \"metric\": \"" << starry::metric_name(cfg.metric) << "\",\n";
  std::cout << "    \"queries\": " << cfg.queries << ",\n";
  std::cout << "    \"threads\": " << cfg.threads << ",\n";
  std::cout << "    \"seed\": " << cfg.seed << ",\n";
  std::cout << "    \"index\": \"" << cfg.index << "\",\n";
  std::cout << "    \"build\": \"" << cfg.build << "\",\n";
  std::cout << "    \"ef\": " << db.search_ef() << "\n";
  std::cout << "  },\n";
  std::cout << "  \"build\": {\n";
  std::cout << json_num("seconds", build_s) << ",\n";
  std::cout << json_num("vectors_per_second",
                        cfg.n / (build_s > 0 ? build_s : 1)) << "\n";
  std::cout << "  },\n";
  std::cout << "  \"search\": {\n";
  std::cout << json_num("wall_seconds", search_s) << ",\n";
  std::cout << json_num("qps", qps) << ",\n";
  std::cout << json_num("avg_us", avg_us) << ",\n";
  std::cout << json_num("min_us", all_us.empty() ? 0 : all_us.front()) << ",\n";
  std::cout << json_num("p50_us", percentile(all_us, 50)) << ",\n";
  std::cout << json_num("p95_us", percentile(all_us, 95)) << ",\n";
  std::cout << json_num("p99_us", percentile(all_us, 99)) << ",\n";
  std::cout << json_num("max_us", all_us.empty() ? 0 : all_us.back()) << "\n";
  std::cout << "  },\n";
  std::cout << json_num("scan_gbps", gbps) << ",\n";
  std::cout << "  \"index\": {\n";
  std::cout << "    \"vectors\": " << db.size() << ",\n";
  std::cout << "    \"row_bytes\": " << (cfg.dim * 4 + 8) << "\n";
  std::cout << "  },\n";
  if (kind == starry::kHnswIndex) {
    std::cout << json_num("recall_at_k", recall) << ",\n";
    std::cout << "  \"exact\": false\n";
  } else {
    std::cout << "  \"exact\": true\n";
  }
  std::cout << "}\n";
  return 0;
}
