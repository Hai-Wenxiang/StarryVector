// StarryVector storage-engine fault-injection suite.
//
// A standalone executable (no doctest) so it can be run by hand on any
// checkout:
//
//   ./build/bin/starry_crash_suite            # all scenarios
//   ./build/bin/starry_crash_suite 3 15       # only scenarios 3 and 15
//
// Every scenario builds a database directory under a fresh temp path,
// injects a specific failure mode (truncated WAL tail, corrupted CRC,
// interrupted checkpoint, missing files, simulated crash ...), then
// verifies the recovery contract:
//
//   * a database either reopens with all fully-persisted writes or it
//     reports a clean kIoError - it never crashes, hangs or returns
//     garbage;
//   * the recovered state is always a PREFIX of the write sequence:
//     no torn writes, no reordering, no half batches;
//   * with WalSync::kSyncAlways every acknowledged write survives a
//     simulated crash (no close(), no checkpoint).
//
// Exit code 0 = all requested scenarios passed.
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "starry/db.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;
const char* g_scenario = "?";

// Directory-scoped temp path: /tmp/starry_crash_<name>_<pid>.
std::string temp_dir(const char* name) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "/tmp/starry_crash_%s_%d", name,
                static_cast<int>(getpid()));
  return std::string(buf);
}

void rm_rf(const std::string& path) {
  std::string cmd = "rm -rf '" + path + "'";
  if (std::system(cmd.c_str()) != 0) {
    std::fprintf(stderr, "[%s] rm_rf failed for %s\n", g_scenario,
                 path.c_str());
  }
}

void mkdir_or_die(const std::string& path) {
  if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
    std::fprintf(stderr, "[%s] mkdir %s failed: %s\n", g_scenario,
                 path.c_str(), std::strerror(errno));
    std::exit(2);
  }
}

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "[%s] CHECK FAILED %s:%d: %s\n", g_scenario,    \
                   __FILE__, __LINE__, #cond);                             \
    }                                                                      \
  } while (0)

#define CHECK_FALSE(cond) CHECK(!(cond))

void require_ok(starry::Status s, const char* what) {
  ++g_checks;
  if (s != starry::kOk) {
    ++g_failures;
    std::fprintf(stderr, "[%s] %s returned status %d\n", g_scenario, what,
                 static_cast<int>(s));
  }
}

// Deterministic data helpers (LCG - independent of the library RNG).
float det_float(unsigned seed) {
  seed = seed * 1103515245u + 12345u;
  return ((seed >> 16) % 2000) / 1000.0f - 1.0f;
}

std::vector<float> det_vector(std::size_t dim, unsigned base) {
  std::vector<float> v(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = det_float(static_cast<unsigned>(base + i));
  }
  return v;
}

starry::VectorDB* open_db(const std::string& dir, starry::Status* s) {
  return starry::VectorDB::open(dir, s).release();
}

std::string snapshot_path(const std::string& dir) { return dir + "/snapshot.bin"; }
std::string wal_path(const std::string& dir) { return dir + "/wal.log"; }

std::size_t file_size(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return static_cast<std::size_t>(-1);
  return static_cast<std::size_t>(st.st_size);
}

bool file_exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

// Truncates the file to `bytes` (simulates a torn tail: the last write
// only partially hit the disk).
bool truncate_file(const std::string& path, std::size_t bytes) {
  return ::truncate(path.c_str(), static_cast<off_t>(bytes)) == 0;
}

// Flips `byte_at` XOR 0xFF somewhere in the file (corrupts one byte).
bool flip_byte(const std::string& path, std::size_t byte_at) {
  FILE* f = std::fopen(path.c_str(), "r+b");
  if (f == 0) return false;
  if (std::fseek(f, static_cast<long>(byte_at), SEEK_SET) != 0) {
    std::fclose(f);
    return false;
  }
  int c = std::fgetc(f);
  if (c == EOF) {
    std::fclose(f);
    return false;
  }
  std::fseek(f, static_cast<long>(byte_at), SEEK_SET);
  std::fputc(c ^ 0xFF, f);
  std::fclose(f);
  return true;
}

// Appends `n` garbage bytes (simulates a half-written final record).
bool append_garbage(const std::string& path, std::size_t n) {
  FILE* f = std::fopen(path.c_str(), "ab");
  if (f == 0) return false;
  for (std::size_t i = 0; i < n; ++i) {
    std::fputc(static_cast<int>((i * 31 + 7) & 0xFF), f);
  }
  std::fclose(f);
  return true;
}

bool write_file_bytes(const std::string& path, const char* data,
                      std::size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == 0) return false;
  const bool ok = std::fwrite(data, 1, n, f) == n;
  std::fclose(f);
  return ok;
}

typedef void (*ScenarioFn)();

struct Scenario {
  const char* name;
  ScenarioFn fn;
};

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

// S01: create -> write -> close -> reopen: every acknowledged write there.
void s01_create_write_close_reopen() {
  const std::string dir = temp_dir("s01");
  rm_rf(dir);
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open(fresh)");
    CHECK(db != 0);
    require_ok(db->insert(1, det_vector(4, 10)), "insert 1");
    require_ok(db->insert(2, det_vector(4, 20)), "insert 2");
    require_ok(db->remove(1), "remove 1");
    require_ok(db->insert(3, det_vector(4, 30)), "insert 3");
    CHECK(db->size() == 2);
    require_ok(db->close(), "close");
    delete db;
  }
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "reopen");
    CHECK(db != 0);
    CHECK(db->size() == 2);
    std::vector<float> out;
    CHECK_FALSE(db->get(1, &out));   // tombstone survived
    CHECK(db->get(2, &out));
    CHECK(db->get(3, &out));
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S02: WAL tail torn (truncate the last K bytes): recovery yields the
// complete prefix - earlier writes intact, the torn one absent.
void s02_wal_tail_torn() {
  const std::string dir = temp_dir("s02");
  rm_rf(dir);
  const std::size_t dim = 4;
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 20; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i),
                            det_vector(dim, static_cast<unsigned>(i))),
                 "insert");
    }
    // Simulate crash: no close() (data is in the OS buffers / WAL file).
    delete db;  // destructor must not corrupt anything
  }
  // Torn tail: cut the WAL in the middle of the last record.
  const std::size_t sz = file_size(wal_path(dir));
  CHECK(sz > 16);
  CHECK(truncate_file(wal_path(dir), sz - 9));
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open after torn tail");
    CHECK(db->size() >= 19);  // at most the last record lost, never more
    std::vector<float> out;
    for (starry::id_t i = 0; i < 19; ++i) {
      CHECK(db->get(i, &out));  // prefix fully present
    }
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S03: single corrupted byte in the WAL body: replay stops at the prefix.
void s03_wal_bad_crc() {
  const std::string dir = temp_dir("s03");
  rm_rf(dir);
  const std::size_t dim = 4;
  std::size_t sz = 0;
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 10; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i),
                            det_vector(dim, static_cast<unsigned>(i))),
                 "insert");
    }
    delete db;
    sz = file_size(wal_path(dir));
    CHECK(sz > 32);
  }
  // Corrupt a byte inside the third record's payload (header 8 bytes +
  // two records of >= 8+1+8+4+4*4 bytes each).
  CHECK(flip_byte(wal_path(dir), sz / 3));
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open after corruption");
    // Prefix property: some leading writes present, no crash, no garbage.
    std::vector<float> out;
    std::size_t present = 0;
    for (starry::id_t i = 0; i < 10; ++i) {
      if (db->get(i, &out)) ++present;
    }
    CHECK(present >= 1);
    CHECK(present <= 10);
    // Present ids must be a prefix 0..present-1.
    for (starry::id_t i = 0; i < present; ++i) {
      CHECK(db->get(i, &out));
    }
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S04: garbage appended after valid records: replay ignores it.
void s04_wal_garbage_tail() {
  const std::string dir = temp_dir("s04");
  rm_rf(dir);
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 5; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i), det_vector(4, 7)),
                 "insert");
    }
    delete db;
  }
  CHECK(append_garbage(wal_path(dir), 13));
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open after garbage tail");
    CHECK(db->size() == 5);  // nothing lost, garbage ignored
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S05: snapshot missing, WAL complete: full replay from the log.
void s05_snapshot_missing_wal_replays() {
  const std::string dir = temp_dir("s05");
  rm_rf(dir);
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 30; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i), det_vector(8, 3)),
                 "insert");
    }
    require_ok(db->checkpoint(), "checkpoint");
    for (int i = 30; i < 50; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i), det_vector(8, 3)),
                 "insert");
    }
    delete db;
  }
  CHECK(file_exists(wal_path(dir)));
  rm_rf(snapshot_path(dir));  // lose the snapshot, keep the WAL
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open without snapshot");
    CHECK(db->size() == 50);
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S06: WAL missing, snapshot present: snapshot state loads.
void s06_wal_missing_snapshot_loads() {
  const std::string dir = temp_dir("s06");
  rm_rf(dir);
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 10; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i), det_vector(4, 5)),
                 "insert");
    }
    require_ok(db->checkpoint(), "checkpoint");
    delete db;
  }
  rm_rf(wal_path(dir));
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open without wal");
    CHECK(db->size() == 10);
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S07: empty/new directory without any files opens as an empty database.
void s07_empty_dir_opens() {
  const std::string dir = temp_dir("s07");
  rm_rf(dir);
  mkdir_or_die(dir);
  starry::Status s;
  starry::VectorDB* db = open_db(dir, &s);
  require_ok(s, "open empty dir");
  CHECK(db != 0);
  CHECK(db->size() == 0);
  CHECK(db->search(det_vector(4, 1), 3).empty());
  require_ok(db->close(), "close");
  delete db;
  rm_rf(dir);
}

// S08: leftover snapshot.bin.tmp from an interrupted checkpoint is
// ignored (fresh checkpoint overwrites it).
void s08_stale_tmp_ignored() {
  const std::string dir = temp_dir("s08");
  rm_rf(dir);
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    require_ok(db->insert(1, det_vector(4, 2)), "insert");
    require_ok(db->checkpoint(), "checkpoint");
    delete db;
  }
  CHECK(write_file_bytes(snapshot_path(dir) + ".tmp", "GARBAGE", 7));
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open with stale tmp");
    CHECK(db->size() == 1);
    require_ok(db->insert(2, det_vector(4, 9)), "insert");
    require_ok(db->checkpoint(), "checkpoint over stale tmp");
    delete db;
  }
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "reopen");
    CHECK(db->size() == 2);
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S09: 100 open/write/close cycles: state accumulates without loss.
void s09_reopen_loop() {
  const std::string dir = temp_dir("s09");
  rm_rf(dir);
  const std::size_t rounds = 100;
  for (std::size_t r = 0; r < rounds; ++r) {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    if (r % 3 == 2) {
      require_ok(db->remove(static_cast<starry::id_t>(r - 1)), "remove");
    }
    require_ok(db->insert(static_cast<starry::id_t>(r), det_vector(4, 4)),
               "insert");
    if (r % 10 == 9) {
      require_ok(db->checkpoint(), "checkpoint");
    }
    require_ok(db->close(), "close");
    delete db;
  }
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "final open");
    std::size_t expect = 0;
    std::vector<float> out;
    for (std::size_t r = 0; r < rounds; ++r) {
      const bool alive = (r % 3 != 1) || (r + 1 >= rounds);
      // r-1 removed when r%3==2 and r>=1; id r-1 stays deleted afterwards.
      const bool removed = (r % 3 == 1) && (r + 1 < rounds);
      CHECK(db->get(static_cast<starry::id_t>(r), &out) == !removed);
      if (!removed) ++expect;
    }
    CHECK(db->size() == expect);
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S10: simulated crash (no close, no checkpoint) with kSyncAlways: every
// acknowledged write survives.
void s10_crash_sync_always() {
  const std::string dir = temp_dir("s10");
  rm_rf(dir);
  const std::size_t dim = 8;
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    db->set_wal_sync(starry::kSyncAlways);
    for (int i = 0; i < 15; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i),
                            det_vector(dim, static_cast<unsigned>(i))),
                 "insert");
    }
    // Simulated crash: leak the object, never call close().
  }
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open after crash");
    CHECK(db->size() == 15);
    std::vector<float> out;
    for (starry::id_t i = 0; i < 15; ++i) {
      CHECK(db->get(i, &out));
    }
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S11: crash with the default (relaxed) sync policy: the database still
// opens cleanly; whatever replay finds is a consistent prefix.
void s11_crash_relaxed_consistent() {
  const std::string dir = temp_dir("s11");
  rm_rf(dir);
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 15; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i), det_vector(8, 8)),
                 "insert");
    }
    // Crash: leak, no close.  Writes may or may not have hit the disk;
    // whatever did must be a clean prefix.
  }
  starry::Status s;
  starry::VectorDB* db = open_db(dir, &s);
  require_ok(s, "open after relaxed crash");
  std::vector<float> out;
  std::size_t present = 0;
  for (starry::id_t i = 0; i < 15; ++i) {
    if (db->get(i, &out)) ++present;
  }
  for (starry::id_t i = 0; i < present; ++i) {
    CHECK(db->get(i, &out));  // strictly a prefix
  }
  CHECK(db->size() == present);
  require_ok(db->close(), "close");
  delete db;
  rm_rf(dir);
}

// S12: a torn bulk-insert record recovers as all-or-nothing.
void s12_bulk_atomic() {
  const std::string dir = temp_dir("s12");
  rm_rf(dir);
  const std::size_t dim = 4;
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    require_ok(db->insert(0, det_vector(dim, 1)), "insert 0");
    std::vector<starry::id_t> ids;
    std::vector<float> vecs;
    for (int i = 1; i <= 40; ++i) {
      ids.push_back(static_cast<starry::id_t>(i));
      const std::vector<float> v = det_vector(dim, static_cast<unsigned>(i));
      vecs.insert(vecs.end(), v.begin(), v.end());
    }
    require_ok(db->insert_bulk(ids, vecs), "insert_bulk");
    delete db;  // crash before close
  }
  // Tear the tail of the WAL.
  const std::size_t sz = file_size(wal_path(dir));
  CHECK(sz > 32);
  CHECK(truncate_file(wal_path(dir), sz - 17));
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open after torn bulk");
    std::vector<float> out;
    std::size_t bulk_present = 0;
    for (starry::id_t i = 1; i <= 40; ++i) {
      if (db->get(i, &out)) ++bulk_present;
    }
    CHECK(bulk_present == 0 || bulk_present == 40);  // atomic batch
    require_ok(db->close(), "close");
    delete db;
  }
  rm_rf(dir);
}

// S13: HNSW database: crash + reopen rebuilds the graph deterministically
// and search quality is unchanged.
void s13_hnsw_recovery() {
  const std::string dir = temp_dir("s13");
  rm_rf(dir);
  const std::size_t dim = 16;
  const std::size_t rows = 1200;
  std::vector<float> truth;
  {
    starry::Status s;
    starry::VectorDB* db =
        starry::VectorDB::open(dir, &s, dim, starry::kL2,
                               starry::kHnswIndex)
            .release();
    // ^ open() with explicit schema for a fresh directory.
    require_ok(s, "open hnsw");
    db->set_search_ef(128);
    for (std::size_t i = 0; i < rows; ++i) {
      const std::vector<float> v = det_vector(dim, static_cast<unsigned>(i));
      truth.insert(truth.end(), v.begin(), v.end());
      require_ok(db->insert(static_cast<starry::id_t>(i), v), "hnsw insert");
    }
    const std::vector<float> q = det_vector(dim, 4242);
    const std::vector<starry::SearchResult> before = db->search(q, 10);
    CHECK(before.size() == 10);
    require_ok(db->close(), "close");
    delete db;

    // Reopen: graph rebuilt by replay, deterministic seed => the same
    // search answers.
    starry::Status s2;
    starry::VectorDB* db2 = open_db(dir, &s2);
    require_ok(s2, "reopen hnsw");
    CHECK(db2->kind() == starry::kHnswIndex);
    db2->set_search_ef(128);
    const std::vector<starry::SearchResult> after = db2->search(q, 10);
    CHECK(after.size() == before.size());
    for (std::size_t i = 0; i < before.size() && i < after.size(); ++i) {
      CHECK(before[i].id == after[i].id);
      CHECK(std::fabs(before[i].distance - after[i].distance) < 1e-4f);
    }
    require_ok(db2->close(), "close");
    delete db2;
  }
  rm_rf(dir);
}

// S14: checkpoint rewrites the snapshot and truncates the WAL.
void s14_checkpoint_shrinks_wal() {
  const std::string dir = temp_dir("s14");
  rm_rf(dir);
  const std::size_t dim = 32;
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    for (int i = 0; i < 200; ++i) {
      require_ok(db->insert(static_cast<starry::id_t>(i),
                            det_vector(dim, static_cast<unsigned>(i))),
                 "insert");
    }
    delete db;
  }
  const std::size_t wal_before = file_size(wal_path(dir));
  CHECK(wal_before > 200 * dim * 4 / 2);  // actually holds the writes
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "open");
    require_ok(db->checkpoint(), "checkpoint");
    const std::size_t wal_after = file_size(wal_path(dir));
    CHECK(wal_after < wal_before);
    CHECK(file_exists(snapshot_path(dir)));
    CHECK(db->size() == 200);
    require_ok(db->close(), "close");
    delete db;
  }
  // And the checkpointed state survives a crash-before-close.
  {
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    require_ok(s, "reopen after checkpoint");
    CHECK(db->size() == 200);
    delete db;
  }
  rm_rf(dir);
}

// S15: concurrent readers observe a consistent database while one writer
// mutates it (read-write lock smoke; TSan validates in depth).
void s15_concurrent_readers_writer() {
  const std::string dir = temp_dir("s15");
  rm_rf(dir);
  const std::size_t dim = 8;
  starry::Status s;
  starry::VectorDB* db = open_db(dir, &s);
  require_ok(s, "open");
  db->set_wal_sync(starry::kSyncAlways);
  for (int i = 0; i < 100; ++i) {
    require_ok(db->insert(static_cast<starry::id_t>(i),
                          det_vector(dim, static_cast<unsigned>(i))),
               "seed insert");
  }

  std::atomic<bool> stop(false);
  // int is fine with C++11 atomics? use <atomic> bool.
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.push_back(std::thread([&db, &stop, dim]() {
      std::vector<float> out;
      while (!stop.load()) {
        const std::vector<starry::SearchResult> hits =
            db->search(det_vector(dim, 77), 5);
        CHECK(hits.size() == 5);
        CHECK(db->get(0, &out));
      }
    }));
  }
  for (int i = 100; i < 300; ++i) {
    require_ok(db->insert(static_cast<starry::id_t>(i),
                          det_vector(dim, static_cast<unsigned>(i))),
               "writer insert");
    if (i % 50 == 0) {
      require_ok(db->remove(static_cast<starry::id_t>(i - 50)), "writer remove");
    }
  }
  stop.store(true);
  for (std::size_t t = 0; t < readers.size(); ++t) {
    readers[t].join();
  }
  require_ok(db->close(), "close");
  delete db;
  rm_rf(dir);
}

// S16: fuzz - random byte flips anywhere in the WAL; the database must
// open without crashing and always expose a sane prefix.
void s16_wal_fuzz() {
  const std::string dir = temp_dir("s16");
  std::mt19937 rng(20260816);
  const std::size_t rounds = 40;
  for (std::size_t round = 0; round < rounds; ++round) {
    rm_rf(dir);
    {
      starry::Status s;
      starry::VectorDB* db = open_db(dir, &s);
      require_ok(s, "open");
      for (int i = 0; i < 12; ++i) {
        require_ok(db->insert(static_cast<starry::id_t>(i),
                              det_vector(4, static_cast<unsigned>(i + round))),
                   "insert");
      }
      delete db;  // crash
    }
    const std::size_t sz = file_size(wal_path(dir));
    if (sz == static_cast<std::size_t>(-1) || sz == 0) {
      continue;
    }
    // Flip 1-3 random bytes.
    const int flips = 1 + static_cast<int>(rng() % 3);
    for (int f = 0; f < flips; ++f) {
      const std::size_t at = static_cast<std::size_t>(rng()) % sz;
      flip_byte(wal_path(dir), at);
    }
    // Must open (or cleanly fail) - never crash.
    starry::Status s;
    starry::VectorDB* db = open_db(dir, &s);
    if (db != 0) {
      // Whatever survived must be a prefix of 0..11.
      std::vector<float> out;
      std::size_t present = 0;
      for (starry::id_t i = 0; i < 12; ++i) {
        if (db->get(i, &out)) ++present;
      }
      bool prefix_ok = true;
      for (starry::id_t i = present; i < 12; ++i) {
        if (db->get(i, &out)) prefix_ok = false;
      }
      CHECK(prefix_ok);
      CHECK(db->size() == present);
      require_ok(db->close(), "close");
      delete db;
    } else {
      CHECK(s == starry::kIoError);
    }
  }
  rm_rf(dir);
}

const Scenario kScenarios[] = {
    {"s01_create_write_close_reopen", s01_create_write_close_reopen},
    {"s02_wal_tail_torn", s02_wal_tail_torn},
    {"s03_wal_bad_crc", s03_wal_bad_crc},
    {"s04_wal_garbage_tail", s04_wal_garbage_tail},
    {"s05_snapshot_missing_wal_replays", s05_snapshot_missing_wal_replays},
    {"s06_wal_missing_snapshot_loads", s06_wal_missing_snapshot_loads},
    {"s07_empty_dir_opens", s07_empty_dir_opens},
    {"s08_stale_tmp_ignored", s08_stale_tmp_ignored},
    {"s09_reopen_loop", s09_reopen_loop},
    {"s10_crash_sync_always", s10_crash_sync_always},
    {"s11_crash_relaxed_consistent", s11_crash_relaxed_consistent},
    {"s12_bulk_atomic", s12_bulk_atomic},
    {"s13_hnsw_recovery", s13_hnsw_recovery},
    {"s14_checkpoint_shrinks_wal", s14_checkpoint_shrinks_wal},
    {"s15_concurrent_readers_writer", s15_concurrent_readers_writer},
    {"s16_wal_fuzz", s16_wal_fuzz},
};

}  // namespace

int main(int argc, char** argv) {
  const int n = static_cast<int>(sizeof(kScenarios) / sizeof(kScenarios[0]));
  std::vector<int> run;
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      run.push_back(std::atoi(argv[i]));
    }
  } else {
    for (int i = 1; i <= n; ++i) run.push_back(i);
  }

  std::printf("starry_crash_suite: %d scenario(s) selected\n",
              static_cast<int>(run.size()));
  for (std::size_t i = 0; i < run.size(); ++i) {
    const int idx = run[i] - 1;
    if (idx < 0 || idx >= n) {
      std::fprintf(stderr, "unknown scenario #%d (range 1..%d)\n", run[i], n);
      return 2;
    }
    g_scenario = kScenarios[idx].name;
    const int before = g_failures;
    std::printf("  [%2d] %-36s ... ", idx + 1, g_scenario);
    std::fflush(stdout);
    kScenarios[idx].fn();
    std::printf("%s\n", g_failures == before ? "ok" : "FAILED");
  }
  std::printf("starry_crash_suite: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
