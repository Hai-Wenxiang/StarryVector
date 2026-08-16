// StarryVector - VectorDB, the top-level engine facade.
//
// VectorDB wraps a FlatIndex and adds the "database" behaviours that a raw
// index does not care about:
//
//   * id uniqueness enforcement      (insert returns kDuplicateId)
//   * soft deletion via tombstones   (remove; the vector stays in the
//                                     index buffer but is skipped during
//                                     search until a future compaction
//                                     rewrites the segment)
//   * point lookup by id             (get)
//   * whole-database persistence     (save / load, one portable file)
//
// Error handling: no exceptions for control flow.  Every operation that
// can fail returns a Status code, in the spirit of LevelDB / SQLite.
//
// Concurrency: NOT thread-safe yet.  The benchmark tool serialises writes
// and only parallelises read-only searches (which are safe concurrently:
// search never mutates state).  A read-write lock is planned together
// with the WAL storage layer.
#ifndef STARRY_DB_HPP
#define STARRY_DB_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "starry/flat_index.hpp"
#include "starry/hnsw_index.hpp"
#include "starry/types.hpp"

namespace starry {

// Result codes for database operations.  kOk means success.
enum Status {
  kOk = 0,
  kInvalidArgument = 1,  // e.g. wrong vector dimension
  kDuplicateId = 2,      // insert() with an id that already exists
  kNotFound = 3,         // remove()/get() of an unknown id
  kIoError = 4,          // file could not be opened / read / written
};

// Which index backs search().  kFlat is exact brute force; kHnsw keeps a
// FlatIndex for storage/ground truth plus an HNSW graph for approximate
// search (recall tunable via the ef search parameter).
enum IndexKind {
  kFlatIndex = 0,
  kHnswIndex = 1,
};

class VectorDB {
 public:
  // Creates an empty database for `dim`-dimensional vectors compared
  // with `metric` (default: squared Euclidean).  With kind == kHnswIndex
  // the database maintains an HNSW graph (M=16, ef_construction=200,
  // fixed seed => reproducible builds) on top of the flat storage.
  explicit VectorDB(std::size_t dim, Metric metric = kL2,
                    IndexKind kind = kFlatIndex);

  // ---- write path ----------------------------------------------------

  // Inserts one vector.  The buffer is copied; it must hold dim() floats.
  Status insert(id_t id, const float* vec);

  // Convenience overload for std::vector storage; validates the size and
  // forwards to the pointer version.
  Status insert(id_t id, const std::vector<float>& vec);

  // Soft-deletes an id: it disappears from search() and get() but the
  // bytes remain in the index buffer until compaction (future work).
  // Returns kNotFound for unknown or already deleted ids.
  Status remove(id_t id);

  // ---- read path -----------------------------------------------------

  // k-NN search over all live (non-deleted) vectors.  With kFlatIndex
  // this is exact; with kHnswIndex it is approximate (see set_search_ef).
  // Returns at most k hits sorted by ascending distance.
  std::vector<SearchResult> search(const float* query, std::size_t k) const;
  std::vector<SearchResult> search(const std::vector<float>& query,
                                   std::size_t k) const;

  // Beam width of HNSW searches (ignored by kFlatIndex).  0 selects the
  // default (64).  Larger values trade speed for recall.
  void set_search_ef(std::size_t ef);
  std::size_t search_ef() const;

  // Index kind this database was created with.
  IndexKind kind() const { return kind_; }

  // Copies the stored vector for `id` into *out.  Returns false when the
  // id is unknown or deleted.  Note: with the cosine metric the stored
  // (and returned) vector is L2-normalised, by design of FlatIndex.
  bool get(id_t id, std::vector<float>* out) const;

  // Number of live (non-deleted) vectors.
  std::size_t size() const { return row_of_.size(); }

  // Total number of rows in the underlying buffer (live + deleted).
  // size() <= capacity() always; the gap is reclaimed by compaction.
  std::size_t capacity() const { return index_.size(); }

  // Vector dimensionality of this database.
  std::size_t dim() const { return dim_; }

  // ---- persistence ---------------------------------------------------

  // Writes the whole database (including tombstones) to a single file.
  // See the top of db.cpp for the exact on-disk layout.  Overwrites any
  // previous file at `path`.
  Status save(const std::string& path) const;

  // Restores a database previously written by save().  On success returns
  // an owning pointer; on failure returns null and, if `status` is
  // non-null, stores the reason.  The caller owns the returned object.
  static std::unique_ptr<VectorDB> load(const std::string& path,
                                        Status* status);

 private:
  const std::size_t dim_;
  const Metric metric_;
  IndexKind kind_;  // logically const after construction; load() flips it
                    // from flat to hnsw after restoring the saved graph
  FlatIndex index_;  // raw storage + brute-force search (ground truth)

  // Present only when kind_ == kHnswIndex.  Null during load(): rows are
  // restored into the flat storage first, then the saved graph is
  // deserialized on top (avoids rebuilding it row by row).
  std::unique_ptr<HnswIndex> hnsw_;
  std::size_t ef_ = 0;  // 0 -> HnswIndex::default_ef()

  // Live id -> row in index_.  Deleted ids are erased from here and put
  // into deleted_ instead; the row stays in the index buffer.
  std::unordered_map<id_t, std::size_t> row_of_;
  std::unordered_set<id_t> deleted_;
};

}  // namespace starry

#endif  // STARRY_DB_HPP
