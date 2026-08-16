// Implementation of the HNSW graph index.  See hnsw_index.hpp for the
// design overview; the algorithm follows Malkov & Yashunin (2016):
//   INSERT: draw level l, greedy-descend to layer l+1, then on every
//   layer from min(l, max_level) down to 0 run an ef_construction-bounded
//   search, pick M diverse neighbours, add bidirectional links, shrink
//   overfull neighbour lists with the same heuristic.
//   SEARCH: greedy-descend with ef=1 to layer 1, ef-bounded best-first
//   walk on layer 0, return the k nearest non-skipped nodes.
#include "starry/hnsw_index.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <queue>
#include <thread>
#include <utility>

#include "starry/flat_index.hpp"

namespace starry {

namespace {

// Distance-tagged row used inside the beam searches.
typedef std::pair<float, std::int32_t> Hit;  // (distance, row)

// Wave size of the deterministic parallel bulk build.  A compile-time
// constant on purpose: the constructed graph must depend only on
// (data, seed, call sequence), never on the number of worker threads,
// so the wave boundaries have to be fixed as well.
const std::size_t kBulkWave = 32;

// std::priority_queue puts the "largest" element (per comparator) on top:
//   Farther  -> top is the entry with the LARGEST distance (use: beam
//               frontier, evict the worst)
//   Nearer   -> top is the entry with the SMALLEST distance (use: expand
//               the closest candidate next)
struct Farther {
  bool operator()(const Hit& a, const Hit& b) const {
    return a.first < b.first;
  }
};
struct Nearer {
  bool operator()(const Hit& a, const Hit& b) const {
    return a.first > b.first;
  }
};

}  // namespace

HnswIndex::HnswIndex(const FlatIndex& base, std::size_t M,
                     std::size_t ef_construction, std::uint64_t seed)
    : base_(base),
      dim_(base.dim()),
      // Same reduction as FlatIndex: cosine rows arrive pre-normalised,
      // so the graph walks with the inner-product kernel and search()
      // adds the +1 constant when reporting distances.
      dist_(base.metric() == kCosine ? resolve_distance_fn(kInnerProduct)
                                     : resolve_distance_fn(base.metric())),
      M_(M),
      M0_(M * 2),
      ef_construction_(ef_construction),
      level_mult_(1.0 / std::log(static_cast<double>(M))),
      level_rng_(seed) {}

float HnswIndex::row_dist(const float* query, std::size_t row) const {
  // Re-fetch on every call: the base buffer may reallocate (vector
  // growth) or materialise (copy-on-write off an mmap view) while the
  // index is being built, so no pointer may be cached across calls.
  return dist_(query, base_.data() + row * dim_, dim_);
}

float HnswIndex::row_row_dist(std::size_t a, std::size_t b) const {
  const float* rows = base_.data();
  return dist_(rows + a * dim_, rows + b * dim_, dim_);
}

std::int32_t HnswIndex::draw_level() {
  std::uniform_real_distribution<double> unif01(0.0, 1.0);
  std::int32_t level = static_cast<std::int32_t>(
      -std::log(unif01(level_rng_)) * level_mult_);
  return level < 0 ? 0 : level;
}

void HnswIndex::select_heuristic(const float* q,
                                 const std::vector<std::int32_t>& cands,
                                 std::size_t M,
                                 std::vector<std::int32_t>* out) const {
  // Candidates arrive sorted nearest-first.
  out->clear();
  for (std::size_t ci = 0; ci < cands.size() && out->size() < M; ++ci) {
    const std::int32_t c = cands[ci];
    const float* cv = base_.data() + static_cast<std::size_t>(c) * dim_;
    bool keep = true;
    for (std::size_t oi = 0; oi < out->size(); ++oi) {
      const std::int32_t o = (*out)[oi];
      const float* ov =
          base_.data() + static_cast<std::size_t>(o) * dim_;
      // If c is closer to an already-kept neighbour than to q, adding it
      // would not diversify the neighbourhood - drop it.
      if (dist_(cv, ov, dim_) < dist_(cv, q, dim_)) {
        keep = false;
        break;
      }
    }
    if (keep) {
      out->push_back(c);
    }
  }
}

void HnswIndex::search_layer(
    const float* query, std::size_t ef,
    const std::vector<std::int32_t>& entries,
    const std::unordered_set<id_t>* skip,
    std::vector<Hit>* results, std::size_t layer) const {
  // results is reused as the output; caller passes an empty vector.
  results->clear();

  std::priority_queue<Hit, std::vector<Hit>, Nearer> candidates;  // top: nearest
  std::priority_queue<Hit, std::vector<Hit>, Farther> frontier;   // top: farthest

  const id_t* ids = base_.ids_data();

  for (std::size_t i = 0; i < entries.size(); ++i) {
    const std::int32_t e = entries[i];
    if (e < 0 || static_cast<std::size_t>(e) >= node_count_) {
      continue;
    }
    const float d = row_dist(query, static_cast<std::size_t>(e));
    // Skip tombstoned nodes in the beam itself (in-filter search).
    if (skip != 0 && skip->find(ids[static_cast<std::size_t>(e)]) != skip->end()) {
      continue;
    }
    candidates.push(Hit(d, e));
    frontier.push(Hit(d, e));
  }

  std::vector<char> visited;  // allocated lazily per call
  visited.assign(node_count_, 0);
  auto mark = [&visited](std::int32_t r) {
    visited[static_cast<std::size_t>(r)] = 1;
  };

  while (!candidates.empty()) {
    const Hit c = candidates.top();
    candidates.pop();
    // Stop when the nearest remaining candidate is worse than the
    // ef-th best found so far (classic HNSW termination).
    if (!frontier.empty() && c.first > frontier.top().first &&
        frontier.size() >= ef) {
      break;
    }
    const Node& node = nodes_[static_cast<std::size_t>(c.second)];
    if (static_cast<std::size_t>(node.level) < layer) {
      continue;  // node does not participate on this layer
    }
    const std::vector<std::int32_t>& nbrs = node.links[layer];
    for (std::size_t ni = 0; ni < nbrs.size(); ++ni) {
      const std::int32_t n = nbrs[ni];
      if (n < 0 || static_cast<std::size_t>(n) >= node_count_ ||
          visited[static_cast<std::size_t>(n)]) {
        continue;
      }
      mark(n);
      const float d = row_dist(query, static_cast<std::size_t>(n));
      const id_t nid = ids[static_cast<std::size_t>(n)];
      if (skip != 0 && skip->find(nid) != skip->end()) {
        continue;  // deleted: do not expand through it either
      }
      const bool beam_not_full = frontier.size() < ef;
      if (beam_not_full || d < frontier.top().first) {
        candidates.push(Hit(d, n));
        frontier.push(Hit(d, n));
        if (frontier.size() > ef) {
          frontier.pop();
        }
      }
    }
  }

  results->clear();
  while (!frontier.empty()) {
    results->push_back(frontier.top());
    frontier.pop();
  }
  std::sort(results->begin(), results->end(),
            [](const Hit& a, const Hit& b) { return a.first < b.first; });
}

void HnswIndex::add_row(std::size_t row) {
  // Rows must arrive densely: 0, 1, 2, ...
  if (row != node_count_) {
    return;  // defensive; VectorDB always satisfies this
  }

  // Exponential level assignment.
  const std::int32_t level = draw_level();

  Node node;
  node.level = level;
  node.links.resize(static_cast<std::size_t>(level) + 1);

  const float* q = base_.data() + row * dim_;

  if (entry_ < 0) {
    // First node: becomes the entry point of every layer up to `level`.
    nodes_.push_back(node);
    ++node_count_;
    entry_ = static_cast<std::int64_t>(row);
    max_level_ = level;
    return;
  }

  // 1) Greedy descent with ef = 1 from the top layer to level+1.
  std::int32_t cur = static_cast<std::int32_t>(entry_);
  float cur_d = row_dist(q, static_cast<std::size_t>(cur));
  for (std::int32_t l = max_level_; l > level; --l) {
    bool improved = true;
    while (improved) {
      improved = false;
      const std::vector<std::int32_t>& nbrs =
          nodes_[static_cast<std::size_t>(cur)].links[static_cast<std::size_t>(l)];
      for (std::size_t ni = 0; ni < nbrs.size(); ++ni) {
        const std::int32_t n = nbrs[ni];
        if (n < 0 || static_cast<std::size_t>(n) >= node_count_) {
          continue;
        }
        const float d = row_dist(q, static_cast<std::size_t>(n));
        if (d < cur_d) {
          cur_d = d;
          cur = n;
          improved = true;
        }
      }
    }
  }

  // 2) ef_construction-bounded insert on every layer from min(level,
  //    max_level) down to 0.
  const std::size_t efc = ef_construction_ > M0_ ? ef_construction_ : M0_;
  std::vector<Hit> layer_hits;
  std::vector<std::int32_t> entries(1, cur);
  std::vector<std::int32_t> selected;

  nodes_.push_back(node);  // row's slot exists now (links still empty)
  ++node_count_;

  for (std::int32_t l = std::min(level, max_level_); l >= 0; --l) {
    search_layer(q, efc, entries, 0, &layer_hits,
                 static_cast<std::size_t>(l));
    // Candidates nearest-first into a plain row list.
    std::vector<std::int32_t> cands;
    cands.reserve(layer_hits.size());
    for (std::size_t i = 0; i < layer_hits.size(); ++i) {
      cands.push_back(layer_hits[i].second);
    }
    select_heuristic(q, cands, M_, &selected);
    nodes_[row].links[static_cast<std::size_t>(l)] = selected;

    // Back-link and shrink overfull lists.
    const std::size_t cap = l == 0 ? M0_ : M_;
    for (std::size_t i = 0; i < selected.size(); ++i) {
      const std::int32_t n = selected[i];
      std::vector<std::int32_t>& nl =
          nodes_[static_cast<std::size_t>(n)].links[static_cast<std::size_t>(l)];
      nl.push_back(static_cast<std::int32_t>(row));
      if (nl.size() > cap) {
        // Re-pick the best `cap` of the merged list around node n.
        std::vector<std::int32_t> merged = nl;
        const float* nv =
            base_.data() + static_cast<std::size_t>(n) * dim_;
        std::sort(merged.begin(), merged.end(),
                  [&](std::int32_t a, std::int32_t b) {
                    return row_row_dist(static_cast<std::size_t>(a),
                                        static_cast<std::size_t>(n)) <
                           row_row_dist(static_cast<std::size_t>(b),
                                        static_cast<std::size_t>(n));
                  });
        select_heuristic(nv, merged, cap, &nl);
      }
    }

    // Entry set for the next lower layer: this layer's results.
    entries.clear();
    for (std::size_t i = 0;
         i < layer_hits.size() && i < efc; ++i) {
      entries.push_back(layer_hits[i].second);
    }
    if (entries.empty()) {
      entries.push_back(static_cast<std::int32_t>(entry_));
    }
  }

  if (level > max_level_) {
    max_level_ = level;
    entry_ = static_cast<std::int64_t>(row);
  }
}

// ---- deterministic parallel bulk construction ------------------------------
//
// Wave algorithm (see add_rows_bulk in the header): rows are appended in
// waves of kBulkWave.  Within a wave every node's candidate search runs
// in parallel against the frozen graph prefix (nodes below the wave
// start); linking then happens sequentially in row order.  Both phases
// are pure functions of the frozen state, so thread count and scheduling
// cannot influence the resulting graph.

void HnswIndex::plan_row(std::size_t row, std::size_t efc,
                         BulkPlan* plan) const {
  plan->frozen_max_level = max_level_;
  plan->layers.assign(static_cast<std::size_t>(plan->level) + 1,
                      std::vector<std::int32_t>());
  if (entry_ < 0) {
    return;  // empty graph: link_row() handles the entry bootstrap
  }

  const float* q = base_.data() + row * dim_;

  // Greedy descent with ef = 1 from the top layer to level+1 (the
  // graph state here is frozen, so this is read-only).
  std::int32_t cur = static_cast<std::int32_t>(entry_);
  float cur_d = row_dist(q, static_cast<std::size_t>(cur));
  for (std::int32_t l = max_level_; l > plan->level; --l) {
    bool improved = true;
    while (improved) {
      improved = false;
      const std::vector<std::int32_t>& nbrs =
          nodes_[static_cast<std::size_t>(cur)].links[static_cast<std::size_t>(l)];
      for (std::size_t ni = 0; ni < nbrs.size(); ++ni) {
        const std::int32_t n = nbrs[ni];
        if (n < 0 || static_cast<std::size_t>(n) >= node_count_) {
          continue;
        }
        const float d = row_dist(q, static_cast<std::size_t>(n));
        if (d < cur_d) {
          cur_d = d;
          cur = n;
          improved = true;
        }
      }
    }
  }

  // ef_construction-bounded candidate search on every layer from
  // min(level, max_level) down to 0, chaining the entry sets exactly
  // like add_row() does.
  std::vector<Hit> layer_hits;
  std::vector<std::int32_t> entries(1, cur);
  for (std::int32_t l = std::min(plan->level, max_level_); l >= 0; --l) {
    search_layer(q, efc, entries, 0, &layer_hits,
                 static_cast<std::size_t>(l));
    std::vector<std::int32_t>& cands =
        plan->layers[static_cast<std::size_t>(l)];
    cands.reserve(layer_hits.size());
    for (std::size_t i = 0; i < layer_hits.size(); ++i) {
      cands.push_back(layer_hits[i].second);
    }
    entries.clear();
    for (std::size_t i = 0; i < layer_hits.size() && i < efc; ++i) {
      entries.push_back(layer_hits[i].second);
    }
    if (entries.empty()) {
      entries.push_back(static_cast<std::int32_t>(entry_));
    }
  }
}

void HnswIndex::link_row(std::size_t row, const BulkPlan& plan) {
  Node node;
  node.level = plan.level;
  node.links.resize(static_cast<std::size_t>(plan.level) + 1);
  nodes_.push_back(node);
  ++node_count_;

  if (entry_ < 0) {
    // Bootstrap of an empty graph: this row is the entry point of every
    // layer up to its level (mirrors the first-node case of add_row).
    entry_ = static_cast<std::int64_t>(row);
    max_level_ = plan.level;
    return;
  }

  const float* q = base_.data() + row * dim_;

  // Layers above frozen_max_level have no candidates (the frozen prefix
  // did not reach them); their link lists stay empty and the node is
  // still reachable through lower layers.
  std::vector<std::int32_t> selected;
  for (std::int32_t l = std::min(plan.level, plan.frozen_max_level);
       l >= 0; --l) {
    const std::vector<std::int32_t>& cands =
        plan.layers[static_cast<std::size_t>(l)];
    select_heuristic(q, cands, M_, &selected);
    nodes_[row].links[static_cast<std::size_t>(l)] = selected;

    // Back-link and shrink overfull lists (identical to add_row).
    const std::size_t cap = l == 0 ? M0_ : M_;
    for (std::size_t i = 0; i < selected.size(); ++i) {
      const std::int32_t n = selected[i];
      std::vector<std::int32_t>& nl =
          nodes_[static_cast<std::size_t>(n)].links[static_cast<std::size_t>(l)];
      nl.push_back(static_cast<std::int32_t>(row));
      if (nl.size() > cap) {
        std::vector<std::int32_t> merged = nl;
        const float* nv =
            base_.data() + static_cast<std::size_t>(n) * dim_;
        std::sort(merged.begin(), merged.end(),
                  [&](std::int32_t a, std::int32_t b) {
                    return row_row_dist(static_cast<std::size_t>(a),
                                        static_cast<std::size_t>(n)) <
                           row_row_dist(static_cast<std::size_t>(b),
                                        static_cast<std::size_t>(n));
                  });
        select_heuristic(nv, merged, cap, &nl);
      }
    }
  }

  if (plan.level > max_level_) {
    max_level_ = plan.level;
    entry_ = static_cast<std::int64_t>(row);
  }
}

void HnswIndex::add_rows_bulk(std::size_t n, std::size_t threads) {
  if (n == 0) {
    return;
  }
  if (threads == 0) {
    threads = std::thread::hardware_concurrency();
    if (threads == 0) {
      threads = 1;
    }
  }
  const std::size_t efc = ef_construction_ > M0_ ? ef_construction_ : M0_;
  const std::size_t end_row = node_count_ + n;

  // Bootstrap: an empty graph gets its entry point serially before any
  // wave runs (planning needs a graph to descend from).
  if (entry_ < 0) {
    BulkPlan plan;
    plan.level = draw_level();
    plan.frozen_max_level = -1;
    link_row(node_count_, plan);
  }

  std::size_t w = node_count_;
  while (w < end_row) {
    const std::size_t wend = w + kBulkWave < end_row ? w + kBulkWave : end_row;
    const std::size_t wn = wend - w;

    // Levels for the whole wave, drawn in row order (same generator as
    // add_row, same consumption order regardless of thread count).
    std::vector<std::int32_t> levels(wn);
    for (std::size_t i = 0; i < wn; ++i) {
      levels[i] = draw_level();
    }

    // Parallel phase: read-only planning against the frozen prefix.
    // Workers grab node indices from an atomic counter; every write goes
    // to the worker's own plan slot, so scheduling cannot be observed.
    std::vector<BulkPlan> plans(wn);
    std::atomic<std::size_t> next(0);
    auto worker = [&]() {
      for (;;) {
        const std::size_t i = next.fetch_add(1);
        if (i >= wn) {
          break;
        }
        plans[i].level = levels[i];
        plan_row(w + i, efc, &plans[i]);
      }
    };
    if (threads > 1 && wn > 1) {
      std::vector<std::thread> pool;
      pool.reserve(threads - 1);
      for (std::size_t t = 1; t < threads; ++t) {
        pool.push_back(std::thread(worker));
      }
      worker();
      for (std::size_t t = 0; t < pool.size(); ++t) {
        pool[t].join();
      }
    } else {
      worker();
    }

    // Sequential phase: link the wave in row order.
    for (std::size_t i = 0; i < wn; ++i) {
      link_row(w + i, plans[i]);
    }
    w = wend;
  }
}

std::vector<SearchResult> HnswIndex::search(
    const float* query, std::size_t k, std::size_t ef,
    const std::unordered_set<id_t>* skip) const {
  std::vector<SearchResult> out;
  if (k == 0 || node_count_ == 0 || entry_ < 0) {
    return out;
  }
  if (ef < k) {
    ef = k;
  }

  // Greedy descent, ef = 1, from the top layer down to layer 1.
  std::int32_t cur = static_cast<std::int32_t>(entry_);
  float cur_d = row_dist(query, static_cast<std::size_t>(cur));
  for (std::int32_t l = max_level_; l >= 1; --l) {
    bool improved = true;
    while (improved) {
      improved = false;
      const std::vector<std::int32_t>& nbrs =
          nodes_[static_cast<std::size_t>(cur)].links[static_cast<std::size_t>(l)];
      for (std::size_t ni = 0; ni < nbrs.size(); ++ni) {
        const std::int32_t n = nbrs[ni];
        if (n < 0 || static_cast<std::size_t>(n) >= node_count_) {
          continue;
        }
        const float d = row_dist(query, static_cast<std::size_t>(n));
        if (d < cur_d) {
          cur_d = d;
          cur = n;
          improved = true;
        }
      }
    }
  }

  std::vector<Hit> hits;
  std::vector<std::int32_t> entries(1, cur);
  search_layer(query, ef, entries, skip, &hits, 0);

  const id_t* ids = base_.ids_data();
  const bool cosine = base_.metric() == kCosine;
  const std::size_t want = std::min(k, hits.size());
  for (std::size_t i = 0; i < want; ++i) {
    SearchResult r;
    r.id = ids[static_cast<std::size_t>(hits[i].second)];
    // IP kernel returned -dot; unit vectors: cosine distance = 1 - dot.
    r.distance = cosine ? hits[i].first + 1.0f : hits[i].first;
    out.push_back(r);
  }
  // hits are nearest-first already; ids mapped through the parallel array.
  return out;
}

void HnswIndex::serialize(std::vector<std::uint64_t>* out) const {
  out->push_back(static_cast<std::uint64_t>(node_count_));
  out->push_back(static_cast<std::uint64_t>(max_level_));
  out->push_back(static_cast<std::uint64_t>(entry_));
  for (std::size_t r = 0; r < node_count_; ++r) {
    const Node& n = nodes_[r];
    out->push_back(static_cast<std::uint64_t>(n.level));
    for (std::int32_t l = 0; l <= n.level; ++l) {
      const std::vector<std::int32_t>& links =
          n.links[static_cast<std::size_t>(l)];
      out->push_back(static_cast<std::uint64_t>(links.size()));
      for (std::size_t i = 0; i < links.size(); ++i) {
        out->push_back(static_cast<std::uint64_t>(links[i]));
      }
    }
  }
}

bool HnswIndex::deserialize(const std::uint64_t* in, std::size_t n,
                            std::size_t node_count) {
  std::size_t pos = 0;
  auto u64 = [&]() -> bool {
    return pos < n;
  };
  if (!u64() || in[pos] != node_count) {
    return false;
  }
  ++pos;
  if (!u64()) return false;
  max_level_ = static_cast<std::int32_t>(in[pos++]);
  if (!u64()) return false;
  entry_ = static_cast<std::int64_t>(in[pos++]);
  if (entry_ < 0 || entry_ >= static_cast<std::int64_t>(node_count)) {
    return false;
  }

  nodes_.clear();
  nodes_.resize(node_count);
  node_count_ = node_count;
  for (std::size_t r = 0; r < node_count; ++r) {
    if (!u64()) return false;
    const std::int32_t level = static_cast<std::int32_t>(in[pos++]);
    if (level < 0 || level > max_level_) {
      return false;
    }
    Node& nd = nodes_[r];
    nd.level = level;
    nd.links.resize(static_cast<std::size_t>(level) + 1);
    for (std::int32_t l = 0; l <= level; ++l) {
      if (!u64()) return false;
      const std::uint64_t deg = in[pos++];
      if (deg > node_count) {
        return false;
      }
      nd.links[static_cast<std::size_t>(l)].resize(static_cast<std::size_t>(deg));
      for (std::uint64_t i = 0; i < deg; ++i) {
        if (!u64()) return false;
        const std::int64_t nb = static_cast<std::int64_t>(in[pos++]);
        if (nb < 0 || nb >= static_cast<std::int64_t>(node_count)) {
          return false;
        }
        nd.links[static_cast<std::size_t>(l)][static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(nb);
      }
    }
  }
  return true;
}

}  // namespace starry
