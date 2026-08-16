// StarryVector - write-ahead log (storage engine, L1).
//
// Append-only crash-safe log backing VectorDB's durable mode.  Record
// framing (all integers little-endian):
//
//   [u32 payload_len][u32 crc32(payload)][payload ...]
//
// payload:
//   [u8 op][u8 reserved][u8 reserved][u8 reserved][u32 count]
//   [u64 id] * count[[float vec] * (count * dim, op != kRemove only)]
//
// op: 1 = insert (count == 1), 2 = remove (count == 1), 3 = bulk insert
// (count == n; one batch = one record = atomic replay unit).
//
// Durability: kSyncAlways fsyncs after every record before the write is
// acknowledged; kSyncOnCheckpoint flushes into the OS cache only (the
// data reaches the disk at checkpoint/close - a crash then loses at
// most the tail, and replay always yields a consistent prefix).
//
// Recovery contract: replay stops at the FIRST invalid frame (length
// beyond the file, CRC mismatch, malformed payload).  Everything before
// it is applied verbatim - a torn or corrupted tail is silently dropped,
// which is exactly the prefix semantics crash recovery needs.
#ifndef STARRY_WAL_HPP
#define STARRY_WAL_HPP

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace starry {

enum WalSync {
  kSyncOnCheckpoint = 0,  // default: OS buffers, fsync at checkpoint/close
  kSyncAlways = 1,        // fsync after every record (max durability)
  kSyncOnClose = 2,       // flush at close only (benchmarks)
};

enum WalOp {
  kWalInsert = 1,
  kWalRemove = 2,
  kWalBulk = 3,
};

// CRC-32/IEEE (the zlib polynomial).  Exposed for tests.
std::uint32_t wal_crc32(const void* data, std::size_t n);

// Append side.  Owns one FILE* opened lazily on the given path; every
// method returns false on I/O failure (the caller surfaces kIoError).
class WalWriter {
 public:
  WalWriter();
  ~WalWriter();

  // Opens `path` for appending (created when missing).
  bool open(const std::string& path);
  bool is_open() const { return f_ != 0; }

  // Appends one framed record.  `vecs` holds count*dim floats for
  // insert/bulk (empty for remove).  With kSyncAlways the record is
  // fsynced before returning.
  bool append(std::uint8_t op, const std::uint64_t* ids, std::size_t count,
              const float* vecs, std::size_t vec_floats);

  // Rolls the log away after a checkpoint: truncates to zero and fsyncs
  // the directory entry semantics by fsyncing the (now empty) file.
  bool truncate();

  // Flushes buffered bytes to the OS (and to the disk when durable).
  bool sync(WalSync policy);

  void close();

 private:
  std::FILE* f_ = 0;
  std::string path_;
};

// Replay side.  Streams valid records to a callback until EOF or the
// first invalid frame (torn tail / corruption), whichever comes first.
class WalReader {
 public:
  // Decoded record handed to the callback.  `ids`/`vecs` point into an
  // internal buffer owned by the reader and stay valid until the next
  // next() call.
  struct Record {
    std::uint8_t op = 0;
    std::uint32_t count = 0;
    const std::uint64_t* ids = 0;   // count entries
    const float* vecs = 0;          // count * dim floats (null for remove)
    std::size_t vec_floats = 0;
  };

  // Returns true when `path` was opened; false on I/O error (missing
  // file is NOT an error - an empty database has no WAL).
  bool open(const std::string& path);

  // Decodes the next valid record into `rec`.  Returns false at EOF or
  // at the first invalid frame (the prefix contract).
  bool next(Record* rec);

  void close();

  // Bytes of valid records consumed so far (diagnostics; also used by
  // tests to locate the logical end of the log).
  std::size_t valid_bytes() const { return pos_; }

 private:
  std::vector<char> buf_;    // whole file, loaded in open()
  std::size_t pos_ = 0;      // scan offset into buf_
  std::vector<std::uint64_t> ids_;
  std::vector<float> vecs_;
};

}  // namespace starry

#endif  // STARRY_WAL_HPP
