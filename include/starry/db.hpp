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
// Concurrency: readers (search/get/size) may run concurrently with each
// other; every mutation (insert/remove/insert_bulk/checkpoint/close)
// takes the exclusive side of a shared_mutex.  Concurrent read-only
// searches during writes are safe and lock-based (no lock-free paths
// yet).
#ifndef STARRY_DB_HPP
#define STARRY_DB_HPP

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "starry/flat_index.hpp"
#include "starry/hnsw_index.hpp"
#include "starry/types.hpp"
#include "starry/wal.hpp"

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

  // Bulk insert of ids.size() vectors (packed row-major in `vecs`,
  // exactly ids.size() * dim() floats; row i belongs to ids[i]).
  //
  // With the HNSW kind the graph is built by the deterministic parallel
  // bulk constructor: the resulting graph depends only on (vectors, id
  // order, seed), never on `threads` or thread scheduling - replaying
  // the same batch sequence reproduces a bit-identical graph (see
  // HnswIndex::add_rows_bulk).  For kFlatIndex this is simply a batched
  // append.  `threads` = 0 selects hardware_concurrency (>= 1).
  //
  // Validation is atomic: a size mismatch returns kInvalidArgument and
  // a duplicated id (inside the batch or live in the database) returns
  // kDuplicateId with NOTHING inserted - unlike insert(), which leaves
  // earlier rows in place when it fails midway.  Soft-deleted ids are
  // revived exactly as with insert().
  Status insert_bulk(const std::vector<id_t>& ids,
                     const std::vector<float>& vecs,
                     std::size_t threads = 0);

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
  std::size_t size() const {
    std::shared_lock<std::shared_mutex> guard(rw_);
    return row_of_.size();
  }

  // Total number of rows in the underlying buffer (live + deleted).
  // size() <= capacity() always; the gap is reclaimed by compaction.
  std::size_t capacity() const {
    std::shared_lock<std::shared_mutex> guard(rw_);
    return index_.size();
  }

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

  // ---- durable storage: directory-backed databases ------------------------
  //
  // open() attaches this process to a database DIRECTORY:
  //
  //   <dir>/snapshot.bin   last checkpoint (the save() format, incl. the
  //                        HNSW graph when kind == kHnswIndex)
  //   <dir>/wal.log        write-ahead log of post-checkpoint mutations
  //   <dir>/starry.meta    schema (dim/metric/kind), written at creation
  //
  // Every mutation is logged to the WAL before it is applied in memory
  // (kSyncAlways fsyncs each record before acknowledging it).  Recovery
  // = snapshot + prefix replay of the WAL: torn or corrupted log tails
  // are dropped, so the reopened state is always a consistent prefix of
  // the acknowledged write sequence.
  //
  // A FRESH directory needs an explicit schema (dim > 0; metric/kind
  // default to kL2/kFlatIndex).  Existing directories derive the schema
  // from starry.meta (or the snapshot header, for pre-meta layouts) and
  // reject a conflicting explicit schema with kInvalidArgument.
  static std::unique_ptr<VectorDB> open(const std::string& dir,
                                        Status* status, std::size_t dim = 0,
                                        Metric metric = kL2,
                                        IndexKind kind = kFlatIndex);

  // Atomically rewrites snapshot.bin (tmp file + rename + fsync) and
  // truncates the WAL.  Only meaningful for directory-backed databases;
  // a no-op kOk otherwise.
  Status checkpoint();

  // Flushes the WAL to disk and closes it.  Post-close mutations return
  // kIoError; reads keep working on the in-memory state.  Idempotent.
  Status close();

  // Durability policy for acknowledged writes (see WalSync in wal.hpp):
  //   kSyncAlways       fsync per record before ack - a crash loses
  //                     nothing that was acknowledged
  //   kSyncOnCheckpoint OS buffers; ack'd writes may be lost to a machine
  //                     crash, but recovery still yields a clean prefix
  //   kSyncOnClose      buffers flushed at close (benchmark mode)
  void set_wal_sync(WalSync policy);
  WalSync wal_sync() const { return sync_; }

  // True when this database is backed by a directory (created by open()).
  bool attached() const { return attached_; }

 private:
  std::size_t dim_;  // fixed after construction (load/open set it once)
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

  // ---- storage-engine state (attached databases only) ----------------
  std::string dir_;              // database directory ("" when detached)
  WalWriter wal_;                // open iff attached_ && !closed_
  WalSync sync_ = kSyncOnCheckpoint;
  bool attached_ = false;
  bool closed_ = false;

  // One RW lock for everything: readers take the shared side, mutators
  // the exclusive side.  dim_/metric_ never change after open and are
  // read without the lock.
  mutable std::shared_mutex rw_;

  // Lock-free serializer body; save() wraps it with the read lock,
  // checkpoint() calls it under its already-held write lock.
  Status save_body(const std::string& path) const;

  // Restores rows/tombstones/graph from a save() file into THIS (fresh)
  // object; shared by load() and open().  Returns the failure status.
  Status restore_from_file(const std::string& path);

  // Applies a WAL record stream (prefix semantics; duplicate inserts and
  // unknown removes are tolerated - idempotent replay across a crash in
  // the middle of checkpoint).
  Status replay_wal();
};

}  // namespace starry

#endif  // STARRY_DB_HPP
