// Implementation of VectorDB: id management, soft deletion, index-kind
// routing (flat / HNSW), the single-file persistence format and the
// directory-backed storage engine (WAL + checkpoint, see wal.hpp).
//
// ---------------------------------------------------------------------------
// On-disk format (save), all integers little-endian, floats in the native
// IEEE-754 bit pattern (every supported host so far is little-endian x86
// or ARM; a byte-swap layer can be added when that changes):
//
//   offset  size  field
//        0     8  magic "STARRYV1" (flat) or "STARRYV2" (HNSW-capable)
//        8     4  metric          (uint32, see the Metric enum values)
//       12     4  index kind      (uint32; STARRYV1 files predate this
//                                   field, the reserved zero there reads
//                                   back as kFlatIndex)
//       16     8  dim             (uint64)
//       24     8  count           (uint64) rows stored, live + deleted
//       32   n*d  vector payload  (float32, row-major, row i is id i)
//       +    8*n  ids             (uint64, parallel to the rows above)
//       +     8  deleted_count    (uint64)
//       +  8*m  deleted ids       (uint64)
//       +     8  graph_words      (uint64; V2 with kind=kHnswIndex only)
//       +  8*g  HNSW graph stream (uint64 words, see HnswIndex::serialize)
//
// The layout stays intentionally dumb and append-only: it can be parsed
// by hand or from the Python validation tooling, and the payload block
// is exactly the FlatIndex storage (mmap-ready in a future milestone).
//
// Directory layout (open()/checkpoint(), see db.hpp for the contracts):
//
//   <dir>/starry.meta     "STMETA01" + metric/kind u32s + dim u64
//   <dir>/snapshot.bin    save() format, replaced atomically on
//                         checkpoint (tmp + fsync + rename + dir fsync)
//   <dir>/wal.log         framed records (wal.hpp); truncated after each
//                         successful checkpoint; replayed as a prefix on
//                         recovery with idempotent semantics
// ---------------------------------------------------------------------------
#include "starry/db.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace starry {

namespace {

const char kMagicV1[8] = {'S', 'T', 'A', 'R', 'R', 'Y', 'V', '1'};
const char kMagicV2[8] = {'S', 'T', 'A', 'R', 'R', 'Y', 'V', '2'};
const char kMetaMagic[8] = {'S', 'T', 'M', 'E', 'T', 'A', '0', '1'};

// -- little-endian integer encode/decode helpers ---------------------------
// memcpy keeps this free of undefined behaviour (no aliasing violations,
// no unaligned reads the compiler cannot see).

void put_u32(char* p, std::uint32_t v) { std::memcpy(p, &v, 4); }
void put_u64(char* p, std::uint64_t v) { std::memcpy(p, &v, 8); }

std::uint32_t get_u32(const char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
std::uint64_t get_u64(const char* p) {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

// Reads exactly n bytes; false on short read or stream error.
bool read_exact(std::ifstream& in, char* dst, std::streamsize n) {
  in.read(dst, n);
  return in.gcount() == n;
}

// -- POSIX plumbing for the directory-backed mode ---------------------------

// mkdir -p (best effort, only the obvious failure paths are reported).
bool mkdir_p(const std::string& path) {
  std::string cur;
  for (std::size_t i = 0; i < path.size(); ++i) {
    cur.push_back(path[i]);
    const bool boundary = path[i] == '/' || i + 1 == path.size();
    if (!boundary) {
      continue;
    }
    if (cur == "/" || cur == ".") {
      continue;
    }
    if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

bool path_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

// fsyncs a regular file or a directory (dir=true).
bool fsync_path(const std::string& path, bool dir) {
  const int flags = O_RDONLY | (dir ? O_DIRECTORY : 0);
  const int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    return false;
  }
  const bool ok = ::fsync(fd) == 0;
  ::close(fd);
  return ok;
}

bool write_meta(const std::string& dir, std::size_t dim, Metric metric,
                IndexKind kind) {
  char buf[32];
  std::memcpy(buf, kMetaMagic, 8);
  put_u32(buf + 8, static_cast<std::uint32_t>(metric));
  put_u32(buf + 12, static_cast<std::uint32_t>(kind));
  put_u64(buf + 16, static_cast<std::uint64_t>(dim));
  put_u64(buf + 24, 0);  // reserved
  const std::string path = dir + "/starry.meta";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == 0) {
    return false;
  }
  const bool ok = std::fwrite(buf, 1, sizeof(buf), f) == sizeof(buf);
  const bool flushed = std::fflush(f) == 0;
  const bool synced = flushed && ::fsync(::fileno(f)) == 0;
  std::fclose(f);
  return ok && synced;
}

bool read_meta(const std::string& dir, std::size_t* dim, Metric* metric,
               IndexKind* kind) {
  std::FILE* f = std::fopen((dir + "/starry.meta").c_str(), "rb");
  if (f == 0) {
    return false;
  }
  char buf[32];
  const bool ok = std::fread(buf, 1, sizeof(buf), f) == sizeof(buf);
  std::fclose(f);
  if (!ok || std::memcmp(buf, kMetaMagic, 8) != 0) {
    return false;
  }
  const std::uint32_t m = get_u32(buf + 8);
  const std::uint32_t k = get_u32(buf + 12);
  const std::uint64_t d = get_u64(buf + 16);
  if (m > kCosine || k > kHnswIndex || d == 0 || d > (1ull << 24)) {
    return false;
  }
  *metric = static_cast<Metric>(m);
  *kind = static_cast<IndexKind>(k);
  *dim = static_cast<std::size_t>(d);
  return true;
}

// Peeks the 32-byte save() header of a snapshot to recover the schema
// (used when a snapshot exists but starry.meta does not).
bool peek_snapshot_schema(const std::string& path, std::size_t* dim,
                          Metric* metric, IndexKind* kind) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == 0) {
    return false;
  }
  char header[32];
  const bool ok = std::fread(header, 1, sizeof(header), f) == sizeof(header);
  std::fclose(f);
  if (!ok) {
    return false;
  }
  const bool is_v2 = std::memcmp(header, kMagicV2, 8) == 0;
  if (!is_v2 && std::memcmp(header, kMagicV1, 8) != 0) {
    return false;
  }
  const std::uint32_t m = get_u32(header + 8);
  const std::uint32_t k = get_u32(header + 12);
  const std::uint64_t d = get_u64(header + 16);
  if (m > kCosine || k > kHnswIndex || d == 0 || d > (1ull << 24)) {
    return false;
  }
  *metric = static_cast<Metric>(m);
  *kind = is_v2 ? static_cast<IndexKind>(k) : kFlatIndex;
  *dim = static_cast<std::size_t>(d);
  return true;
}

}  // namespace

// Read-only whole-file mmap with RAII ( PROT_READ / MAP_PRIVATE ).
// Declared in db.hpp as an opaque type; the mapping must outlive any
// FlatIndex view adopted off its bytes.
class MappedFile {
 public:
  ~MappedFile() { close(); }

  bool open_read(const std::string& path) {
    close();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
      ::close(fd);
      return false;
    }
    size_ = static_cast<std::size_t>(st.st_size);
    map_ = ::mmap(0, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) {
      map_ = 0;
      return false;
    }
    return true;
  }

  const char* data() const { return static_cast<const char*>(map_); }
  std::size_t size() const { return size_; }

  void close() {
    if (map_ != 0) {
      ::munmap(map_, size_);
      map_ = 0;
    }
    size_ = 0;
  }

 private:
  void* map_ = 0;
  std::size_t size_ = 0;
};


// ---- construction ---------------------------------------------------------

VectorDB::VectorDB(std::size_t dim, Metric metric, IndexKind kind)
    : dim_(dim), metric_(metric), kind_(kind), index_(dim, metric) {
  if (kind_ == kHnswIndex) {
    hnsw_.reset(new HnswIndex(index_));
  }
}

VectorDB::~VectorDB() {}  // members (incl. the MappedFile) unwind here

// ---- write path -----------------------------------------------------------

Status VectorDB::insert(id_t id, const float* vec) {
  std::unique_lock<std::shared_mutex> guard(rw_);
  if (closed_) {
    return kIoError;
  }
  if (row_of_.find(id) != row_of_.end()) {
    return kDuplicateId;
  }
  // WAL first: the record is durable (per the sync policy) before it
  // becomes visible in memory.
  if (attached_) {
    if (!wal_.append(kWalInsert, &id, 1, vec, dim_) ||
        !wal_.sync(sync_)) {
      return kIoError;
    }
  }
  // Deleted ids may be re-inserted: the old row stays behind as garbage
  // until compaction, the fresh row becomes the live one.
  const std::size_t row = index_.size();
  row_of_[id] = row;
  index_.add(id, vec);
  if (hnsw_ != 0) {
    hnsw_->add_row(row);  // rows arrive densely 0, 1, 2, ...
  }
  deleted_.erase(id);  // an id cannot be both live and deleted
  return kOk;
}

Status VectorDB::insert(id_t id, const std::vector<float>& vec) {
  if (vec.size() != dim_) {
    return kInvalidArgument;
  }
  return insert(id, &vec[0]);
}

Status VectorDB::insert_bulk(const std::vector<id_t>& ids,
                             const std::vector<float>& vecs,
                             std::size_t threads) {
  if (vecs.size() != ids.size() * dim_) {
    return kInvalidArgument;
  }
  if (ids.empty()) {
    return kOk;
  }
  std::unique_lock<std::shared_mutex> guard(rw_);
  if (closed_) {
    return kIoError;
  }
  // Atomic validation: no duplicates inside the batch, none against the
  // live ids.  Only after the whole batch passes are rows appended.
  std::unordered_set<id_t> batch(ids.begin(), ids.end());
  if (batch.size() != ids.size()) {
    return kDuplicateId;
  }
  for (std::unordered_set<id_t>::const_iterator it = batch.begin();
       it != batch.end(); ++it) {
    if (row_of_.find(*it) != row_of_.end()) {
      return kDuplicateId;
    }
  }
  if (attached_) {
    // One batch = one WAL record: replay is all-or-nothing.
    if (!wal_.append(kWalBulk, ids.empty() ? 0 : &ids[0], ids.size(),
                     vecs.empty() ? 0 : &vecs[0], vecs.size()) ||
        !wal_.sync(sync_)) {
      return kIoError;
    }
  }
  const std::size_t start = index_.size();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    row_of_[ids[i]] = start + i;
    index_.add(ids[i], &vecs[i * dim_]);
    deleted_.erase(ids[i]);  // revived, as with insert()
  }
  if (hnsw_ != 0) {
    hnsw_->add_rows_bulk(ids.size(), threads);
  }
  return kOk;
}

Status VectorDB::remove(id_t id) {
  std::unique_lock<std::shared_mutex> guard(rw_);
  if (closed_) {
    return kIoError;
  }
  std::unordered_map<id_t, std::size_t>::iterator it = row_of_.find(id);
  if (it == row_of_.end()) {
    return kNotFound;
  }
  if (attached_) {
    if (!wal_.append(kWalRemove, &id, 1, 0, 0) || !wal_.sync(sync_)) {
      return kIoError;
    }
  }
  row_of_.erase(it);
  deleted_.insert(id);
  return kOk;
}

// ---- read path ------------------------------------------------------------

std::vector<SearchResult> VectorDB::search(const float* query,
                                           std::size_t k) const {
  std::shared_lock<std::shared_mutex> guard(rw_);
  if (hnsw_ != 0) {
    const std::size_t ef = ef_ != 0 ? ef_ : HnswIndex::default_ef();
    return hnsw_->search(query, k, ef, &deleted_);
  }
  // Tombstones are passed down as the skip set; the flat index itself
  // stays a dumb container that does not know about deletion semantics.
  return index_.search(query, k, &deleted_);
}

void VectorDB::set_search_ef(std::size_t ef) {
  std::unique_lock<std::shared_mutex> guard(rw_);
  ef_ = ef;
}

std::size_t VectorDB::search_ef() const {
  std::shared_lock<std::shared_mutex> guard(rw_);
  return ef_ != 0 ? ef_ : HnswIndex::default_ef();
}

std::vector<SearchResult> VectorDB::search(
    const std::vector<float>& query, std::size_t k) const {
  if (query.size() != dim_) {
    return std::vector<SearchResult>();
  }
  return search(&query[0], k);
}

bool VectorDB::get(id_t id, std::vector<float>* out) const {
  std::shared_lock<std::shared_mutex> guard(rw_);
  std::unordered_map<id_t, std::size_t>::const_iterator it = row_of_.find(id);
  if (it == row_of_.end()) {
    return false;
  }
  const float* rows = index_.data();
  const std::size_t row = it->second;
  out->assign(rows + row * dim_, rows + (row + 1) * dim_);
  return true;
}

// ---- single-file persistence (save / load) ---------------------------------

Status VectorDB::save(const std::string& path) const {
  std::shared_lock<std::shared_mutex> guard(rw_);
  return save_body(path);
}

// Lock-free serializer body; save() wraps it with the read lock and
// checkpoint() calls it under its already-held write lock (never both).
Status VectorDB::save_body(const std::string& path) const {
  std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!out) {
    return kIoError;
  }

  const std::uint64_t n = static_cast<std::uint64_t>(index_.size());
  const std::uint64_t d = static_cast<std::uint64_t>(dim_);
  const std::uint64_t del = static_cast<std::uint64_t>(deleted_.size());

  char header[32];
  std::memcpy(header, kind_ == kHnswIndex ? kMagicV2 : kMagicV1, 8);
  put_u32(header + 8, static_cast<std::uint32_t>(metric_));
  put_u32(header + 12, static_cast<std::uint32_t>(kind_));
  put_u64(header + 16, d);
  put_u64(header + 24, n);
  out.write(header, sizeof(header));

  if (n > 0) {
    out.write(reinterpret_cast<const char*>(index_.data()),
              static_cast<std::streamsize>(n * d * sizeof(float)));
    out.write(reinterpret_cast<const char*>(index_.ids_data()),
              static_cast<std::streamsize>(n * sizeof(id_t)));
  }

  char count8[8];
  put_u64(count8, del);
  out.write(count8, sizeof(count8));
  for (std::unordered_set<id_t>::const_iterator it = deleted_.begin();
       it != deleted_.end(); ++it) {
    put_u64(count8, *it);
    out.write(count8, sizeof(count8));
  }

  if (kind_ == kHnswIndex) {
    std::vector<std::uint64_t> graph;
    hnsw_->serialize(&graph);
    put_u64(count8, graph.size());
    out.write(count8, sizeof(count8));
    if (!graph.empty()) {
      out.write(reinterpret_cast<const char*>(graph.data()),
                static_cast<std::streamsize>(graph.size() * sizeof(std::uint64_t)));
    }
  }

  out.flush();
  if (!out) {
    return kIoError;
  }
  return kOk;
}

// Restores the save() file `path` into this (fresh) object.  Refactored
// out of load() so open() can reuse it for snapshot recovery; identical
// logic to the historical loader.
Status VectorDB::restore_from_file(const std::string& path) {
  // Zero-copy recovery: the snapshot is mmapped and its payload/ids
  // blocks are adopted by the FlatIndex as a read-only view (see
  // FlatIndex::adopt).  Only the HNSW graph words (tiny relative to the
  // payload) are copied.  The mapping is kept for the database's
  // lifetime - or until the first write materialises the view.
  MappedFile* mf = new MappedFile();
  if (!mf->open_read(path)) {
    delete mf;
    return kIoError;
  }
  const char* p = mf->data();
  const std::size_t sz = mf->size();
  if (sz < 32) {
    delete mf;
    return kIoError;
  }

  const bool is_v2 = std::memcmp(p, kMagicV2, 8) == 0;
  if (!is_v2 && std::memcmp(p, kMagicV1, 8) != 0) {
    delete mf;
    return kIoError;  // wrong format
  }
  const std::uint32_t metric_raw = get_u32(p + 8);
  if (metric_raw != static_cast<std::uint32_t>(metric_)) {
    delete mf;
    return kIoError;  // schema mismatch
  }
  const std::uint32_t kind_raw = get_u32(p + 12);
  if (kind_raw > kHnswIndex) {
    delete mf;
    return kIoError;  // unknown index kind
  }
  const IndexKind kind = is_v2 ? static_cast<IndexKind>(kind_raw) : kFlatIndex;
  const std::uint64_t d = get_u64(p + 16);
  const std::uint64_t n = get_u64(p + 24);
  if (d != dim_ || n > (1ull << 40)) {
    delete mf;
    return kIoError;  // dim mismatch or insane count
  }

  // Segment boundaries with overflow-safe checks: the file must hold
  // header + payload + ids + tombstone count at the very least.
  const std::uint64_t row_bytes = d * 4 + 8;  // <= 2^26 + 8, no overflow
  if (n > 0 && row_bytes > 0 &&
      (n > (sz - 40) / row_bytes)) {
    delete mf;
    return kIoError;  // truncated file
  }
  const std::size_t payload_off = 32;
  const std::size_t ids_off =
      payload_off + static_cast<std::size_t>(n) * static_cast<std::size_t>(d) * 4;
  const std::size_t del_off =
      ids_off + static_cast<std::size_t>(n) * 8;
  if (del_off + 8 > sz) {
    delete mf;
    return kIoError;
  }

  // Adopt the payload + ids blocks (already cosine-normalised where the
  // metric demands it - that is exactly what save() wrote).
  index_.adopt(reinterpret_cast<const float*>(p + payload_off),
               reinterpret_cast<const id_t*>(p + ids_off),
               static_cast<std::size_t>(n));

  // Rebuild the live-id map straight off the adopted view; duplicate
  // ids mean a corrupt snapshot.
  for (std::uint64_t i = 0; i < n; ++i) {
    const id_t id = get_u64(p + ids_off + static_cast<std::size_t>(i) * 8);
    if (!row_of_.emplace(id, static_cast<std::size_t>(i)).second) {
      delete mf;  // view dies with the mapping; index_ is reset by the
                  // caller failing the whole open/load anyway
      return kIoError;
    }
  }

  // Tombstones.
  const std::uint64_t del = get_u64(p + del_off);
  if (del_off + 8 + static_cast<std::size_t>(del) * 8 > sz) {
    delete mf;
    return kIoError;
  }
  for (std::uint64_t i = 0; i < del; ++i) {
    deleted_.insert(
        get_u64(p + del_off + 8 + static_cast<std::size_t>(i) * 8));
  }
  // Deleted ids that never existed are tolerated (corrupt file), same
  // as the historical streaming loader did via remove()'s kNotFound.
  for (std::unordered_set<id_t>::const_iterator it = deleted_.begin();
       it != deleted_.end(); ++it) {
    row_of_.erase(*it);
  }

  // HNSW graph (V2 only): the only block actually copied - word counts
  // are tiny next to the payload.
  if (kind == kHnswIndex) {
    const std::size_t graph_off = del_off + 8 + static_cast<std::size_t>(del) * 8;
    if (graph_off + 8 > sz) {
      delete mf;
      return kIoError;
    }
    const std::uint64_t graph_words = get_u64(p + graph_off);
    if (graph_words > (1ull << 32) ||
        graph_off + 8 + static_cast<std::size_t>(graph_words) * 8 > sz) {
      delete mf;
      return kIoError;
    }
    std::vector<std::uint64_t> graph(
        reinterpret_cast<const std::uint64_t*>(p + graph_off + 8),
        reinterpret_cast<const std::uint64_t*>(p + graph_off + 8) +
            static_cast<std::size_t>(graph_words));
    if (hnsw_ == 0) {
      kind_ = kHnswIndex;
      hnsw_.reset(new HnswIndex(index_));
    }
    if (!hnsw_->deserialize(graph.empty() ? 0 : &graph[0], graph.size(),
                            static_cast<std::size_t>(n))) {
      delete mf;
      return kIoError;
    }
  }

  // Success: the mapping (and the adopted view) belongs to this object.
  mapped_.reset(mf);
  return kOk;
}

std::unique_ptr<VectorDB> VectorDB::load(const std::string& path,
                                         Status* status) {
  // Peek the header to learn the schema, construct a matching (fresh)
  // database, then restore into it.
  Metric metric = kL2;
  IndexKind kind = kFlatIndex;
  std::size_t dim = 0;
  if (!peek_snapshot_schema(path, &dim, &metric, &kind)) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }
  std::unique_ptr<VectorDB> db(new VectorDB(dim, metric, kFlatIndex));
  const Status s = db->restore_from_file(path);
  if (s != kOk) {
    if (status != 0) *status = s;
    return std::unique_ptr<VectorDB>();
  }
  if (status != 0) *status = kOk;
  return db;
}

// ---- directory-backed storage engine ---------------------------------------

Status VectorDB::checkpoint() {
  std::unique_lock<std::shared_mutex> guard(rw_);
  if (!attached_ || closed_) {
    return kOk;  // in-memory database or nothing left to flush
  }
  // Snapshot to a temp file, fsync it, rename over the old snapshot,
  // fsync the directory (rename durability), then drop the log.  A crash
  // at any point leaves either the old snapshot+wal or the new snapshot
  // (+ a stale wal whose replay is idempotent - see replay_wal).
  const std::string tmp = dir_ + "/snapshot.bin.tmp";
  const std::string dst = dir_ + "/snapshot.bin";
  const Status s = save_body(tmp);
  if (s != kOk) {
    return s;
  }
  if (!fsync_path(tmp, false)) {
    return kIoError;
  }
  if (::rename(tmp.c_str(), dst.c_str()) != 0) {
    return kIoError;
  }
  if (!fsync_path(dir_, true)) {
    return kIoError;
  }
  if (!wal_.truncate()) {
    return kIoError;
  }
  // Marker: "a snapshot exists".  If it disappears (operator error,
  // partial restore) while the WAL was already truncated, refusing to
  // open beats silently serving an incomplete database.
  const std::string mark = dir_ + "/checkpoint.ok";
  std::FILE* f = std::fopen(mark.c_str(), "wb");
  if (f == 0) {
    return kIoError;
  }
  std::fclose(f);
  if (!fsync_path(mark, false) || !fsync_path(dir_, true)) {
    return kIoError;
  }
  return kOk;
}

Status VectorDB::close() {
  std::unique_lock<std::shared_mutex> guard(rw_);
  if (!attached_ || closed_) {
    return kOk;
  }
  // Flush + fsync the log (whatever the policy promised ends up on
  // disk), then close the writer.  Writes already fsynced under
  // kSyncAlways are durable regardless.
  if (!wal_.sync(kSyncAlways)) {
    return kIoError;
  }
  wal_.close();
  closed_ = true;
  return kOk;
}

void VectorDB::set_wal_sync(WalSync policy) {
  std::unique_lock<std::shared_mutex> guard(rw_);
  sync_ = policy;
}

Status VectorDB::replay_wal() {
  WalReader reader;
  if (!reader.open(dir_ + "/wal.log")) {
    return kOk;  // no log: empty database, nothing to replay
  }
  WalReader::Record rec;
  while (reader.next(&rec)) {
    if (rec.op == kWalRemove) {
      remove(rec.ids[0]);  // kNotFound tolerated (stale/idempotent)
      continue;
    }
    if (rec.op == kWalInsert) {
      if (rec.vec_floats != dim_) {
        return kIoError;  // schema mismatch: corrupt log
      }
      const std::vector<float> v(rec.vecs, rec.vecs + rec.vec_floats);
      const Status s = insert(rec.ids[0], v);
      // kDuplicateId: the record predates the snapshot (checkpoint raced
      // with a crash) - already applied, skip idempotently.
      if (s != kOk && s != kDuplicateId) {
        return kIoError;
      }
      continue;
    }
    // kWalBulk: apply row by row through insert() so the HNSW graph (if
    // any) is rebuilt serially - bit-identical to the original build.
    if (rec.count == 0 || rec.vec_floats % rec.count != 0 ||
        rec.vec_floats / rec.count != dim_) {
      return kIoError;
    }
    for (std::uint32_t i = 0; i < rec.count; ++i) {
      const std::vector<float> v(
          rec.vecs + static_cast<std::size_t>(i) * dim_,
          rec.vecs + static_cast<std::size_t>(i + 1) * dim_);
      const Status s = insert(rec.ids[i], v);
      if (s != kOk && s != kDuplicateId) {
        return kIoError;
      }
    }
  }
  return kOk;
}

std::unique_ptr<VectorDB> VectorDB::open(const std::string& dir,
                                         Status* status, std::size_t dim,
                                         Metric metric, IndexKind kind) {
  if (status != 0) {
    *status = kOk;
  }
  if (!mkdir_p(dir)) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }

  // Resolve the schema: meta file wins, else the snapshot header (meta
  // back-fill), else the caller must provide one.
  std::size_t dim_from_store = 0;
  Metric metric_from_store = kL2;
  IndexKind kind_from_store = kFlatIndex;
  const bool have_meta =
      read_meta(dir, &dim_from_store, &metric_from_store, &kind_from_store);
  if (!have_meta) {
    const std::string snap = dir + "/snapshot.bin";
    if (peek_snapshot_schema(snap, &dim_from_store, &metric_from_store,
                             &kind_from_store)) {
      if (!write_meta(dir, dim_from_store, metric_from_store,
                      kind_from_store)) {
        if (status != 0) *status = kIoError;
        return std::unique_ptr<VectorDB>();
      }
    } else if (dim == 0) {
      // Fresh directory, no schema anywhere: refuse to guess.
      if (status != 0) *status = kInvalidArgument;
      return std::unique_ptr<VectorDB>();
    } else {
      dim_from_store = dim;
      metric_from_store = metric;
      kind_from_store = kind;
      if (!write_meta(dir, dim_from_store, metric_from_store,
                      kind_from_store)) {
        if (status != 0) *status = kIoError;
        return std::unique_ptr<VectorDB>();
      }
    }
  }

  // An explicit schema must agree with the stored one.
  if (dim != 0 &&
      (dim != dim_from_store || metric != metric_from_store ||
       kind != kind_from_store)) {
    if (status != 0) *status = kInvalidArgument;
    return std::unique_ptr<VectorDB>();
  }

  // A checkpoint marker without its snapshot means the snapshot was
  // lost after the WAL had already been truncated: the recoverable
  // prefix no longer covers acknowledged writes - refuse.
  if (path_exists(dir + "/checkpoint.ok") &&
      !path_exists(dir + "/snapshot.bin")) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }

  // Restore the snapshot (when present) ...
  std::unique_ptr<VectorDB> db(
      new VectorDB(dim_from_store, metric_from_store, kind_from_store));
  db->dir_ = dir;
  if (path_exists(dir + "/snapshot.bin")) {
    const Status s = db->restore_from_file(dir + "/snapshot.bin");
    if (s != kOk) {
      if (status != 0) *status = s;
      return std::unique_ptr<VectorDB>();
    }
  }

  // ... then replay the WAL on top (prefix + idempotent semantics).
  const Status rs = db->replay_wal();
  if (rs != kOk) {
    if (status != 0) *status = rs;
    return std::unique_ptr<VectorDB>();
  }

  // Engine attached: mutations go to the log from here on.
  if (!db->wal_.open(dir + "/wal.log")) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }
  db->attached_ = true;
  return db;
}

}  // namespace starry
