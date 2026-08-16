// Implementation of VectorDB: id management, soft deletion and the
// single-file persistence format.
//
// ---------------------------------------------------------------------------
// On-disk format (save), all integers little-endian, floats in the native
// IEEE-754 bit pattern (every supported host so far is little-endian x86
// or ARM; a byte-swap layer can be added when that changes):
//
//   offset  size  field
//        0     8  magic "STARRYV1"
//        8     4  metric          (uint32, see the Metric enum values)
//       12     4  reserved padding (zero, keeps the header 8-byte aligned)
//       16     8  dim             (uint64)
//       24     8  count           (uint64) rows stored, live + deleted
//       32   n*d  vector payload  (float32, row-major, row i is id i)
//       +    8*n  ids             (uint64, parallel to the rows above)
//       +     8  deleted_count    (uint64)
//       +  8*m  deleted ids       (uint64)
//
// The layout is intentionally dumb and append-only: it can be rewritten
// by hand with a hex editor, parsed by the Python validation tooling if
// ever needed, and later mmap'd directly (the payload block is exactly
// the FlatIndex storage).
// ---------------------------------------------------------------------------
#include "starry/db.hpp"

#include <cstring>
#include <fstream>

namespace starry {

namespace {

const char kMagic[8] = {'S', 'T', 'A', 'R', 'R', 'Y', 'V', '1'};

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

VectorDB::VectorDB(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric), index_(dim, metric) {}

// ---- write path -----------------------------------------------------------

Status VectorDB::insert(id_t id, const float* vec) {
  if (row_of_.find(id) != row_of_.end()) {
    return kDuplicateId;
  }
  // Deleted ids may be re-inserted: the old row stays behind as garbage
  // until compaction, the fresh row becomes the live one.
  row_of_[id] = index_.size();
  index_.add(id, vec);
  deleted_.erase(id);  // an id cannot be both live and deleted
  return kOk;
}

Status VectorDB::insert(id_t id, const std::vector<float>& vec) {
  if (vec.size() != dim_) {
    return kInvalidArgument;
  }
  return insert(id, &vec[0]);
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
  // Tombstones are passed down as the skip set; the index itself stays a
  // dumb container that does not know about deletion semantics.
  return index_.search(query, k, &deleted_);
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
  std::memcpy(header, kMagic, 8);
  put_u32(header + 8, static_cast<std::uint32_t>(metric_));
  put_u32(header + 12, 0);  // reserved, zero
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
  if (!read_exact(in, header, sizeof(header)) ||
      std::memcmp(header, kMagic, 8) != 0) {
    if (status != 0) *status = kIoError;  // truncated or wrong format
    return std::unique_ptr<VectorDB>();
  }
  const std::uint32_t metric_raw = get_u32(header + 8);
  if (metric_raw > kCosine) {
    if (status != 0) *status = kIoError;  // unknown metric value
    return std::unique_ptr<VectorDB>();
  }
  const Metric metric = static_cast<Metric>(metric_raw);
  const std::uint64_t d = get_u64(header + 16);
  const std::uint64_t n = get_u64(header + 24);
  if (d == 0 || d > (1u << 24) || n > (1ull << 40)) {
    if (status != 0) *status = kInvalidArgument;  // sanity limits
    return std::unique_ptr<VectorDB>();
  }

  // Build the database through the public insert()/remove() path so the
  // in-memory state is guaranteed consistent with a freshly built one.
  std::unique_ptr<VectorDB> db(new VectorDB(
      static_cast<std::size_t>(d), metric));
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

  if (status != 0) *status = kOk;
  return db;
}

}  // namespace starry
