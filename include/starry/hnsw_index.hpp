// StarryVector - HNSW approximate nearest neighbour index (M2).
//
// Implementation of the Hierarchical Navigable Small World graph
// (Malkov & Yashunin, 2016) with the standard parameters:
//
//   * node level drawn from an exponential distribution, mL = 1/ln(M)
//   * greedy descent from the top layer to layer 1 (ef = 1)
//   * ef-bounded best-first search on layer 0
//   * neighbour selection with the diversity heuristic (Algorithm 4)
//
// The index does NOT own the vectors: it navigates rows of an external,
// immutable-once-added base storage (the FlatIndex buffer inside
// VectorDB).  Row pointers are re-fetched through the base on every
// distance evaluation, so reallocation of the underlying std::vector
// during the build phase is harmless.
//
// Determinism: the level generator is a seeded mt19937, so the same
// insert sequence always yields the identical graph (reproducible
// builds, see notes/01 §3.3).  add_rows_bulk() extends this guarantee
// to parallel construction: the wave size is a fixed constant, planning
// reads only a frozen graph prefix and linking is sequential, so the
// resulting graph is a pure function of (data, seed, call sequence) -
// independent of the number of worker threads and their scheduling.
// Note the bulk-built graph is deterministic but NOT identical to the
// row-by-row add_row() graph: within one wave, nodes link only to rows
// that existed before the wave (recall is equivalent, see tests).
//
// Deletion: callers pass a skip set of ids; deleted nodes are ignored
// during the walk.  Graph repair after deletion is future work (M5).
#ifndef STARRY_HNSW_INDEX_HPP
#define STARRY_HNSW_INDEX_HPP

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include "starry/distance.hpp"
#include "starry/types.hpp"

namespace starry {

class FlatIndex;

class HnswIndex {
 public:
  // `base` supplies row vectors for distance evaluation; it must outlive
  // this index and keep rows 0..r-1 valid once row r has been added.
  HnswIndex(const FlatIndex& base, std::size_t M = 16,
            std::size_t ef_construction = 200, std::uint64_t seed = 100);

  // Adds one node for base row `row` (must equal current node count,
  // i.e. rows are added densely in order 0, 1, 2, ...).
  void add_row(std::size_t row);

  // Deterministic parallel bulk construction of `n` nodes for base rows
  // [size(), size()+n); the rows must already exist in the base index
  // (VectorDB::insert_bulk appends them first).  `threads` workers
  // cooperate; 0 selects hardware_concurrency (clamped to >= 1).
  //
  // Algorithm: rows are processed in waves of a fixed compile-time size
  // (kBulkWave).  For each wave, every node's ef_construction-bounded
  // candidate search runs in parallel against the frozen graph prefix
  // (rows below the wave start - read-only, lock-free); the nodes are
  // then linked into the graph sequentially in row order.  The wave
  // size never depends on `threads`, hence the determinism above.
  void add_rows_bulk(std::size_t n, std::size_t threads = 0);

  // Approximate k-NN over rows, skipping ids in `skip` (may be null).
  // `ef` is the search-time beam width; larger = better recall, slower.
  // Returns at most k hits sorted by ascending distance; ids come from
  // the base index.
  std::vector<SearchResult> search(const float* query, std::size_t k,
                                   std::size_t ef,
                                   const std::unordered_set<id_t>* skip) const;

  // Default beam width used when the caller passes ef = 0.
  static std::size_t default_ef() { return 64; }

  std::size_t size() const { return node_count_; }
  std::size_t M() const { return M_; }
  std::size_t ef_construction() const { return ef_construction_; }

  // ---- persistence --------------------------------------------------------
  // Appends the graph structure (levels + adjacency) to a flat uint64
  // stream; VectorDB stitches this into its on-disk format.  The vector
  // payload itself is NOT duplicated here - it lives in the base index.
  void serialize(std::vector<std::uint64_t>* out) const;

  // Restores a graph written by serialize().  `node_count` rows must
  // already exist in the base index.  Returns false on a malformed stream.
  bool deserialize(const std::uint64_t* in, std::size_t n, std::size_t node_count);

  // Entry row for the top layer (-1 when the graph is empty).
  std::int64_t entry_point() const { return entry_; }
  std::int32_t max_level() const { return max_level_; }

 private:
  struct Node {
    std::int32_t level = -1;                    // top layer of this node
    std::vector<std::vector<std::int32_t>> links;  // [level] -> neighbour rows
  };

  // Distance from `query` to base row `row` (re-fetched every call).
  float row_dist(const float* query, std::size_t row) const;
  float row_row_dist(std::size_t a, std::size_t b) const;

  // Draws the next node level from the seeded generator (exponential
  // distribution, mL = 1/ln(M)).  Shared by add_row and the bulk path
  // so both consume the generator identically.
  std::int32_t draw_level();

  // Diversity heuristic (paper Alg. 4, keepPrunedConnections disabled):
  // walk candidates nearest-first, keep c only when it is closer to the
  // base point than to every neighbour already kept.  `cands` is only
  // read (sorted nearest-first by the caller).
  void select_heuristic(const float* q,
                        const std::vector<std::int32_t>& cands,
                        std::size_t M, std::vector<std::int32_t>* out) const;

  // Best-first search on one layer, beam width ef.  Fills `results` with
  // up to ef (row, dist) pairs, nearest first.
  void search_layer(const float* query, std::size_t ef,
                    const std::vector<std::int32_t>& entries,
                    const std::unordered_set<id_t>* skip,
                    std::vector<std::pair<float, std::int32_t>>* results,
                    std::size_t layer) const;

  // ---- deterministic parallel bulk construction (add_rows_bulk) ----------
  // Per-node plan produced by the parallel phase: the level and, for
  // every layer that exists in the frozen prefix, the candidate rows
  // (nearest first) found by searching the frozen graph.
  struct BulkPlan {
    std::int32_t level = 0;
    std::int32_t frozen_max_level = 0;  // max_level_ when planned
    // layer -> candidate rows; sized level+1, layers above
    // frozen_max_level stay empty (linked lists stay empty likewise).
    std::vector<std::vector<std::int32_t>> layers;
  };

  // Read-only worker of the parallel phase: computes the plan for one
  // row against the current (frozen) graph.  Must not mutate state.
  void plan_row(std::size_t row, std::size_t efc, BulkPlan* plan) const;

  // Sequential counterpart of add_row()'s linking half: materialises the
  // planned node, back-links neighbours and shrinks overfull lists.
  void link_row(std::size_t row, const BulkPlan& plan);

  const FlatIndex& base_;
  const std::size_t dim_;
  const DistanceFn dist_;
  const std::size_t M_;
  const std::size_t M0_;                // max degree on layer 0 (2*M)
  const std::size_t ef_construction_;
  const double level_mult_;

  std::vector<Node> nodes_;
  std::size_t node_count_ = 0;
  std::int32_t max_level_ = -1;
  std::int64_t entry_ = -1;

  // Seeded generator for level assignment; same seed => identical graph.
  std::mt19937_64 level_rng_;

  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;
};

}  // namespace starry

#endif  // STARRY_HNSW_INDEX_HPP
