// Implementation of VectorDB: id management, soft deletion, index-kind
// routing (flat / HNSW) and the single-file persistence format.
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
// ---------------------------------------------------------------------------
#include "starry/db.hpp"

#include <cstring>
#include <fstream>

namespace starry {

namespace {

const char kMagicV1[8] = {'S', 'T', 'A', 'R', 'R', 'Y', 'V', '1'};
const char kMagicV2[8] = {'S', 'T', 'A', 'R', 'R', 'Y', 'V', '2'};

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

}  // namespace

// ---- construction ---------------------------------------------------------

VectorDB::VectorDB(std::size_t dim, Metric metric, IndexKind kind)
    : dim_(dim), metric_(metric), kind_(kind), index_(dim, metric) {
  if (kind_ == kHnswIndex) {
    hnsw_.reset(new HnswIndex(index_));
  }
}

// ---- write path -----------------------------------------------------------

Status VectorDB::insert(id_t id, const float* vec) {
  if (row_of_.find(id) != row_of_.end()) {
    return kDuplicateId;
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
  std::unordered_map<id_t, std::size_t>::iterator it = row_of_.find(id);
  if (it == row_of_.end()) {
    return kNotFound;
  }
  row_of_.erase(it);
  deleted_.insert(id);
  return kOk;
}

// ---- read path ------------------------------------------------------------

std::vector<SearchResult> VectorDB::search(const float* query,
                                           std::size_t k) const {
  if (hnsw_ != 0) {
    const std::size_t ef = ef_ != 0 ? ef_ : HnswIndex::default_ef();
    return hnsw_->search(query, k, ef, &deleted_);
  }
  // Tombstones are passed down as the skip set; the flat index itself
  // stays a dumb container that does not know about deletion semantics.
  return index_.search(query, k, &deleted_);
}

void VectorDB::set_search_ef(std::size_t ef) { ef_ = ef; }

std::size_t VectorDB::search_ef() const {
  return ef_ != 0 ? ef_ : HnswIndex::default_ef();
}

std::vector<SearchResult> VectorDB::search(const std::vector<float>& query,
                                           std::size_t k) const {
  if (query.size() != dim_) {
    return std::vector<SearchResult>();
  }
  return search(&query[0], k);
}

bool VectorDB::get(id_t id, std::vector<float>* out) const {
  std::unordered_map<id_t, std::size_t>::const_iterator it = row_of_.find(id);
  if (it == row_of_.end()) {
    return false;
  }
  const std::vector<float>& storage = index_.storage();
  const std::size_t row = it->second;
  out->assign(storage.begin() + row * dim_,
              storage.begin() + (row + 1) * dim_);
  return true;
}

// ---- persistence ----------------------------------------------------------

Status VectorDB::save(const std::string& path) const {
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
    out.write(reinterpret_cast<const char*>(index_.storage().data()),
              static_cast<std::streamsize>(n * d * sizeof(float)));
    out.write(reinterpret_cast<const char*>(index_.ids().data()),
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

std::unique_ptr<VectorDB> VectorDB::load(const std::string& path,
                                         Status* status) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }

  char header[32];
  if (!read_exact(in, header, sizeof(header))) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }
  const bool is_v2 = std::memcmp(header, kMagicV2, 8) == 0;
  if (!is_v2 && std::memcmp(header, kMagicV1, 8) != 0) {
    if (status != 0) *status = kIoError;  // truncated or wrong format
    return std::unique_ptr<VectorDB>();
  }
  const std::uint32_t metric_raw = get_u32(header + 8);
  if (metric_raw > kCosine) {
    if (status != 0) *status = kIoError;  // unknown metric value
    return std::unique_ptr<VectorDB>();
  }
  const Metric metric = static_cast<Metric>(metric_raw);
  const std::uint32_t kind_raw = get_u32(header + 12);
  if (kind_raw > kHnswIndex) {
    if (status != 0) *status = kIoError;  // unknown index kind
    return std::unique_ptr<VectorDB>();
  }
  // A V1 file always wrote zero here, which reads back as kFlatIndex.
  const IndexKind kind = is_v2 ? static_cast<IndexKind>(kind_raw) : kFlatIndex;
  const std::uint64_t d = get_u64(header + 16);
  const std::uint64_t n = get_u64(header + 24);
  if (d == 0 || d > (1u << 24) || n > (1ull << 40)) {
    if (status != 0) *status = kInvalidArgument;  // sanity limits
    return std::unique_ptr<VectorDB>();
  }

  // Build the database through the public insert()/remove() path so the
  // in-memory state is guaranteed consistent with a freshly built one.
  // kind is passed as flat for now: the HNSW graph (if any) is restored
  // from its serialised form afterwards instead of being rebuilt, which
  // is both faster and bit-identical to what was saved.
  std::unique_ptr<VectorDB> db(new VectorDB(
      static_cast<std::size_t>(d), metric, kFlatIndex));
  std::vector<float> buf(static_cast<std::size_t>(d));

  // The ids block sits AFTER the payload block (see the format table at
  // the top of this file), so jump ahead, read it into memory, then seek
  // back and stream the payloads row by row.  This keeps the peak extra
  // memory at 8 bytes per row instead of duplicating the payload.
  std::vector<char> row_ids;
  row_ids.resize(static_cast<std::size_t>(n) * sizeof(id_t));
  if (n > 0) {
    in.seekg(static_cast<std::streamoff>(32 + n * d * sizeof(float)));
    if (!read_exact(in, &row_ids[0],
                    static_cast<std::streamsize>(row_ids.size()))) {
      if (status != 0) *status = kIoError;
      return std::unique_ptr<VectorDB>();
    }
    in.seekg(static_cast<std::streamoff>(32));
  }
  for (std::uint64_t i = 0; i < n; ++i) {
    if (!read_exact(in, reinterpret_cast<char*>(&buf[0]),
                    static_cast<std::streamsize>(buf.size() * sizeof(float)))) {
      if (status != 0) *status = kIoError;
      return std::unique_ptr<VectorDB>();
    }
    const id_t id = get_u64(&row_ids[static_cast<std::size_t>(i) * sizeof(id_t)]);
    if (db->insert(id, &buf[0]) != kOk) {
      if (status != 0) *status = kIoError;  // duplicate id: corrupt file
      return std::unique_ptr<VectorDB>();
    }
  }

  char count8[8];
  // The payload and id blocks were consumed above; jump past them to
  // the trailing tombstone section instead of trusting the stream
  // position (which sits at the start of the id block right now).
  if (n > 0) {
    in.seekg(static_cast<std::streamoff>(
        32 + n * d * sizeof(float) + n * sizeof(id_t)));
  }
  if (!read_exact(in, count8, sizeof(count8))) {
    if (status != 0) *status = kIoError;
    return std::unique_ptr<VectorDB>();
  }
  const std::uint64_t del = get_u64(count8);
  for (std::uint64_t i = 0; i < del; ++i) {
    if (!read_exact(in, count8, sizeof(count8))) {
      if (status != 0) *status = kIoError;
      return std::unique_ptr<VectorDB>();
    }
    db->remove(get_u64(count8));  // already-absent ids in a corrupt file
                                   // are tolerated (kNotFound ignored)
  }

  if (kind == kHnswIndex) {
    std::uint64_t graph_words = 0;
    if (!read_exact(in, reinterpret_cast<char*>(&graph_words),
                    sizeof(graph_words)) ||
        graph_words > (1ull << 32)) {
      if (status != 0) *status = kIoError;
      return std::unique_ptr<VectorDB>();
    }
    std::vector<std::uint64_t> graph(static_cast<std::size_t>(graph_words));
    if (graph_words > 0 &&
        !read_exact(in, reinterpret_cast<char*>(graph.data()),
                    static_cast<std::streamsize>(graph.size() * 8))) {
      if (status != 0) *status = kIoError;
      return std::unique_ptr<VectorDB>();
    }
    // Switch the live database over to the HNSW kind and restore the
    // saved graph on top of the freshly loaded rows.
    db->kind_ = kHnswIndex;
    db->hnsw_.reset(new HnswIndex(db->index_));
    if (!db->hnsw_->deserialize(graph.empty() ? 0 : &graph[0], graph.size(),
                                static_cast<std::size_t>(n))) {
      if (status != 0) *status = kIoError;
      return std::unique_ptr<VectorDB>();
    }
  }

  if (status != 0) *status = kOk;
  return db;
}

}  // namespace starry
