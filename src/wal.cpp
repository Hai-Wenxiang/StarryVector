// Implementation of the write-ahead log.  See wal.hpp for the frame
// layout and the durability / prefix-replay contracts.
#include "starry/wal.hpp"

#include <cstring>
#include <unistd.h>

namespace starry {

namespace {

// Little-endian encode/decode via memcpy (alignment-safe, no UB).
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

// Sanity cap: no legitimate record carries more than this many payload
// bytes (guards against a corrupted length field trying to allocate
// gigabytes).  1 GiB is far beyond one bulk batch of any sane size.
const std::uint32_t kMaxPayload = 1u << 30;

}  // namespace

std::uint32_t wal_crc32(const void* data, std::size_t n) {
  // CRC-32/IEEE, bitwise (no table: the WAL is not hot enough to pay
  // 1 KiB of static tables; ~8 bytes/cycle is plenty here).
  static std::uint32_t table[256];
  static bool built = false;
  if (!built) {
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) != 0u ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      }
      table[i] = c;
    }
    built = true;
  }
  const unsigned char* p = static_cast<const unsigned char*>(data);
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// ---- WalWriter --------------------------------------------------------

WalWriter::WalWriter() {}

WalWriter::~WalWriter() { close(); }

bool WalWriter::open(const std::string& path) {
  close();
  path_ = path;
  f_ = std::fopen(path.c_str(), "ab");
  return f_ != 0;
}

bool WalWriter::append(std::uint8_t op, const std::uint64_t* ids,
                       std::size_t count, const float* vecs,
                       std::size_t vec_floats) {
  if (f_ == 0 || ids == 0 || count == 0) {
    return false;
  }
  const std::size_t payload =
      1 + 3 + 4 + count * 8 + vec_floats * sizeof(float);
  if (payload > kMaxPayload) {
    return false;
  }

  // Frame header + payload assembled in one buffer so a single fwrite
  // issues (torn writes only ever affect the tail of the file).
  std::vector<char> rec(8 + payload);
  put_u32(&rec[0], static_cast<std::uint32_t>(payload));
  // CRC over the payload bytes, written after they are laid out below.
  rec[8] = static_cast<char>(op);
  rec[9] = rec[10] = rec[11] = 0;
  put_u32(&rec[12], static_cast<std::uint32_t>(count));
  char* idp = &rec[16];
  for (std::size_t i = 0; i < count; ++i) {
    put_u64(idp + i * 8, ids[i]);
  }
  if (vec_floats > 0 && vecs != 0) {
    std::memcpy(&rec[16 + count * 8], vecs, vec_floats * sizeof(float));
  }
  put_u32(&rec[4], wal_crc32(&rec[8], payload));

  if (std::fwrite(&rec[0], 1, rec.size(), f_) != rec.size()) {
    return false;
  }
  return true;
}

bool WalWriter::truncate() {
  if (f_ == 0) {
    return false;
  }
  if (std::freopen(path_.c_str(), "wb", f_) == 0) {
    return false;
  }
  return std::fflush(f_) == 0 && ::fsync(::fileno(f_)) == 0;
}

bool WalWriter::sync(WalSync policy) {
  if (f_ == 0) {
    return false;
  }
  if (policy == kSyncOnClose) {
    return true;  // nothing durable promised
  }
  if (std::fflush(f_) != 0) {
    return false;
  }
  if (policy == kSyncAlways) {
    return ::fsync(::fileno(f_)) == 0;
  }
  return true;  // kSyncOnCheckpoint: OS cache is enough for now
}

void WalWriter::close() {
  if (f_ != 0) {
    std::fclose(f_);
    f_ = 0;
  }
}

// ---- WalReader --------------------------------------------------------

bool WalReader::open(const std::string& path) {
  close();
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == 0) {
    return false;  // missing WAL = empty log, caller treats as fine
  }
  std::fseek(f, 0, SEEK_END);
  const long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    std::fclose(f);
    return false;
  }
  buf_.resize(static_cast<std::size_t>(sz));
  const bool ok =
      sz == 0 || std::fread(&buf_[0], 1, buf_.size(), f) == buf_.size();
  std::fclose(f);
  pos_ = 0;
  return ok;
}

bool WalReader::next(Record* rec) {
  // Need at least a frame header.
  if (pos_ + 8 > buf_.size()) {
    return false;  // EOF or torn header
  }
  const std::uint32_t payload = get_u32(&buf_[pos_]);
  const std::uint32_t crc = get_u32(&buf_[pos_ + 4]);
  if (payload < 8 || payload > kMaxPayload) {
    return false;  // malformed length (min: op + reserved + count)
  }
  if (pos_ + 8 + payload > buf_.size()) {
    return false;  // torn tail: header promised more than the file has
  }
  const char* body = &buf_[pos_ + 8];
  if (wal_crc32(body, payload) != crc) {
    return false;  // corrupted payload
  }
  const std::uint8_t op = static_cast<std::uint8_t>(body[0]);
  if (op != kWalInsert && op != kWalRemove && op != kWalBulk) {
    return false;
  }
  // Payload layout (mirrors WalWriter::append): [op][r3][count][ids][vecs]
  // - count at offset 4, ids at offset 8, vecs right after the ids.
  const std::uint32_t count = get_u32(body + 4);
  const std::size_t ids_bytes = static_cast<std::size_t>(count) * 8;
  if (8 + ids_bytes > payload) {
    return false;  // ids run past the frame
  }
  const std::size_t vec_bytes = payload - 8 - ids_bytes;
  if (op == kWalRemove) {
    if (count != 1 || vec_bytes != 0) {
      return false;
    }
  } else if (vec_bytes % sizeof(float) != 0) {
    return false;  // vector block must be whole floats
  }

  ids_.resize(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    ids_[i] = get_u64(body + 8 + static_cast<std::size_t>(i) * 8);
  }
  vecs_.resize(vec_bytes / sizeof(float));
  if (!vecs_.empty()) {
    std::memcpy(&vecs_[0], body + 8 + ids_bytes, vec_bytes);
  }

  rec->op = op;
  rec->count = count;
  rec->ids = ids_.empty() ? 0 : &ids_[0];
  rec->vecs = vecs_.empty() ? 0 : &vecs_[0];
  rec->vec_floats = vecs_.size();
  pos_ += 8 + payload;
  return true;
}

void WalReader::close() {
  buf_.clear();
  buf_.shrink_to_fit();
  pos_ = 0;
}

}  // namespace starry
