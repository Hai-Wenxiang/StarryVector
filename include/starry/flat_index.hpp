// StarryVector - brute-force ("Flat") exact nearest neighbour index.
//
// This is the simplest possible index: all vectors live in one contiguous
// growable buffer and a search linearly scans every stored vector, keeping
// the k best candidates in a small sorted buffer (insertion into position).
// It is exact - recall is 100% by construction - and doubles as the ground
// truth generator for future approximate indexes (HNSW, IVF).
//
// Cosine metric: vectors are L2-normalised once on add() and the query is
// normalised once per search(); the scan then reduces to plain inner
// products (cosine distance = 1 - dot of unit vectors), which halves the
// per-row arithmetic compared to computing both norms every time.  Zero
// vectors are stored unchanged and naturally keep the "distance 1"
// convention of the reference kernel.
//
// Memory layout: row-major, vector i occupies [i * dim, (i+1) * dim).
// Contiguity is kept so that a later SIMD kernel or an mmap'd file can be
// handed the raw buffer pointer directly, with zero deserialisation.
#ifndef STARRY_FLAT_INDEX_HPP
#define STARRY_FLAT_INDEX_HPP

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "starry/distance.hpp"
#include "starry/types.hpp"

namespace starry {

class FlatIndex {
 public:
  // Builds an empty index for vectors of `dim` dimensions compared with
  // `metric`.  Both values are fixed for the lifetime of the index.
  FlatIndex(std::size_t dim, Metric metric);

  // Appends one vector.  The pointed-to memory is COPIED into internal
  // storage (L2-normalised first when the metric is cosine), so the caller
  // may reuse or free its buffer afterwards.  The caller is responsible
  // for id uniqueness (VectorDB enforces it); the index itself accepts
  // anything to stay a dumb, fast container.
  void add(id_t id, const float* vec);

  // Number of stored vectors (including ones the caller may consider
  // deleted - deletion is the caller's concern, see the skip parameter
  // of search()).
  std::size_t size() const { return ids_.size(); }

  // Vector dimensionality.
  std::size_t dim() const { return dim_; }

  // Comparison metric of this index (needed by indexes layered on top,
  // e.g. HnswIndex, to resolve their own distance kernel).
  Metric metric() const { return metric_; }

  // Read-only access to the parallel id array (ids()[i] belongs to the
  // vector at row i).  Used by VectorDB for persistence.
  const std::vector<id_t>& ids() const { return ids_; }

  // Read-only access to the raw row-major vector storage.  The pointer
  // stays valid until the next add() / destruction.  With the cosine
  // metric the rows are stored normalised.
  const std::vector<float>& storage() const { return storage_; }

  // Exact k-nearest-neighbour search.
  //
  //   query - points to dim() floats
  //   k     - number of neighbours wanted (0 returns an empty result)
  //   skip  - optional set of ids to pretend they do not exist (this is
  //           how VectorDB implements deletion without rewriting the
  //           buffer).  May be null.
  //
  // Returns at most k hits sorted by ascending distance.  If fewer than k
  // non-skipped vectors exist, fewer hits are returned.
  std::vector<SearchResult> search(const float* query,
                                   std::size_t k,
                                   const std::unordered_set<id_t>* skip) const;

 private:
  const std::size_t dim_;        // fixed dimensionality
  const Metric metric_;          // metric handed out by metric()
  const DistanceFn dist_;        // resolved scan kernel (never null)
  const bool cosine_pre_;        // metric == kCosine: pre-normalised rows
  std::vector<float> storage_;   // size = size() * dim_, row-major
  std::vector<id_t> ids_;        // storage row -> caller id
};

}  // namespace starry

#endif  // STARRY_FLAT_INDEX_HPP
