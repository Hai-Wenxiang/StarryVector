// Unit tests for the StarryVector core (M0 scope).
//
// Uses doctest (vendored single header).  The most
// important test pattern here is the "oracle comparison": search results
// are validated against a naive reference implementation (compute all
// distances, sort everything, take the first k).  Every future index
// must pass the same oracle.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "starry/db.hpp"
#include "starry/distance.hpp"
#include "starry/flat_index.hpp"
#include "starry/hnsw_index.hpp"

namespace {

// Fills `out` with `count` uniform floats in [-1, 1) from a private
// seeded generator so test data never depends on execution order.
std::vector<float> make_random_vector(std::size_t count, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::vector<float> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = uniform(rng);
  }
  return out;
}

// Naive reference top-k: all distances, full sort, first k.
std::vector<starry::SearchResult> reference_top_k(
    const std::vector<float>& data, const std::vector<starry::id_t>& ids,
    std::size_t dim, starry::DistanceFn dist, const float* query,
    std::size_t k, const std::unordered_set<starry::id_t>* skip) {
  std::vector<starry::SearchResult> all;
  const std::size_t rows = ids.size();
  for (std::size_t i = 0; i < rows; ++i) {
    if (skip != 0 && skip->find(ids[i]) != skip->end()) {
      continue;
    }
    starry::SearchResult r;
    r.id = ids[i];
    r.distance = dist(query, &data[i * dim], dim);
    all.push_back(r);
  }
  std::sort(all.begin(), all.end(),
            [](const starry::SearchResult& a, const starry::SearchResult& b) {
              return a.distance < b.distance;
            });
  if (all.size() > k) {
    all.resize(k);
  }
  return all;
}

// True when both hit lists contain the same ids in the same order with
// (approximately) equal distances.  Ties in distance can legitimately
// reorder equal ids, but with random continuous data ties are
// measure-zero, so strict order comparison is safe here.
bool same_hits(const std::vector<starry::SearchResult>& a,
               const std::vector<starry::SearchResult>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].id != b[i].id) {
      return false;
    }
    if (std::fabs(a[i].distance - b[i].distance) > 1e-4f) {
      return false;
    }
  }
  return true;
}

}  // namespace

// ---- distance kernels ------------------------------------------------------

TEST_CASE("simd kernels match the scalar oracle within tolerance") {
  if (!starry::simd_kernels_available()) {
    return;
  }
  // Odd sizes exercise the scalar tail; 32/8 multiples exercise the
  // unrolled and 8-wide main loops of the AVX2 kernels.
  const std::size_t dims[] = {1, 7, 8, 31, 32, 33, 64, 127, 1024};
  for (std::size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); ++di) {
    const std::size_t dim = dims[di];
    const std::vector<float> a = make_random_vector(dim, 11);
    const std::vector<float> b = make_random_vector(dim, 12);

    const starry::DistanceFn l2_fast = starry::resolve_distance_fn(starry::kL2);
    const starry::DistanceFn ip_fast =
        starry::resolve_distance_fn(starry::kInnerProduct);
    const starry::DistanceFn l2_ref =
        starry::resolve_distance_fn_scalar(starry::kL2);
    const starry::DistanceFn ip_ref =
        starry::resolve_distance_fn_scalar(starry::kInnerProduct);

    const float e_l2 = std::fabs(l2_fast(&a[0], &b[0], dim) -
                                  l2_ref(&a[0], &b[0], dim));
    const float e_ip = std::fabs(ip_fast(&a[0], &b[0], dim) -
                                 ip_ref(&a[0], &b[0], dim));
    // Relative-ish tolerance: summation order differences scale with the
    // magnitude of the accumulated value.
    CHECK(e_l2 <= 1e-4f * (1.0f + std::fabs(l2_ref(&a[0], &b[0], dim))));
    CHECK(e_ip <= 1e-4f * (1.0f + std::fabs(ip_ref(&a[0], &b[0], dim))));
  }
}

TEST_CASE("l2_distance matches hand-computed values") {
  const float a[] = {0.0f, 0.0f};
  const float b[] = {3.0f, 4.0f};
  // 3^2 + 4^2 = 25, no sqrt (squared convention).
  CHECK(starry::l2_distance(a, b, 2) == doctest::Approx(25.0f));
  CHECK(starry::l2_distance(a, a, 2) == doctest::Approx(0.0f));
}

TEST_CASE("inner_product_distance is the negated dot product") {
  const float a[] = {1.0f, 2.0f};
  const float b[] = {3.0f, 4.0f};
  // dot = 1*3 + 2*4 = 11 -> distance -11.
  CHECK(starry::inner_product_distance(a, b, 2) == doctest::Approx(-11.0f));
}

TEST_CASE("cosine_distance: parallel, orthogonal, opposite, zero vector") {
  const float a[] = {1.0f, 0.0f};
  const float b[] = {2.0f, 0.0f};   // same direction as a
  const float c[] = {0.0f, 5.0f};   // orthogonal to a
  const float d[] = {-1.0f, 0.0f};  // opposite to a
  const float z[] = {0.0f, 0.0f};   // degenerate
  CHECK(starry::cosine_distance(a, b, 2) == doctest::Approx(0.0f).epsilon(1e-5));
  CHECK(starry::cosine_distance(a, c, 2) == doctest::Approx(1.0f).epsilon(1e-5));
  CHECK(starry::cosine_distance(a, d, 2) == doctest::Approx(2.0f).epsilon(1e-5));
  CHECK(starry::cosine_distance(a, z, 2) == doctest::Approx(1.0f));  // guarded
}

TEST_CASE("metric_name / parse_metric round-trip") {
  const char* names[] = {"l2", "ip", "cosine"};
  for (int i = 0; i < 3; ++i) {
    starry::Metric m;
    REQUIRE(starry::parse_metric(names[i], &m));
    CHECK(m == static_cast<starry::Metric>(i));
    CHECK(std::string(starry::metric_name(m)) == names[i]);
  }
  starry::Metric m;
  CHECK_FALSE(starry::parse_metric("manhattan", &m));
}

// ---- FlatIndex vs oracle ----------------------------------------------------

TEST_CASE("flat index search matches the naive oracle on random data") {
  const std::size_t dim = 17;
  const std::size_t rows = 500;
  const std::size_t k = 13;
  const std::uint32_t seed = 7;

  std::vector<float> data = make_random_vector(rows * dim, seed);
  std::vector<starry::id_t> ids(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    ids[i] = 1000 + i;  // non-trivial ids, not equal to row numbers
  }

  for (int mi = 0; mi < 3; ++mi) {
    const starry::Metric metric = static_cast<starry::Metric>(mi);
    starry::FlatIndex index(dim, metric);
    for (std::size_t i = 0; i < rows; ++i) {
      index.add(ids[i], &data[i * dim]);
    }
    // Named variable on purpose: taking &temp[0] of the return value
    // directly would leave a dangling pointer into a destroyed vector.
    const std::vector<float> query_vec = make_random_vector(dim, seed + 1);
    const float* query = &query_vec[0];
    const std::vector<starry::SearchResult> got = index.search(query, k, 0);
    const std::vector<starry::SearchResult> want =
        reference_top_k(data, ids, dim, starry::resolve_distance_fn(metric),
                        query, k, 0);
    CHECK(got.size() == k);
    CHECK(same_hits(got, want));
  }
}

TEST_CASE("flat index skips deleted ids passed via the skip set") {
  const std::size_t dim = 8;
  starry::FlatIndex index(dim, starry::kL2);
  for (starry::id_t id = 0; id < 50; ++id) {
    const std::vector<float> v = make_random_vector(dim, static_cast<std::uint32_t>(id));
    index.add(id, &v[0]);
  }
  std::unordered_set<starry::id_t> skip;
  skip.insert(0);
  skip.insert(25);
  skip.insert(49);

  const std::vector<float> query_vec = make_random_vector(dim, 999);
  const float* query = &query_vec[0];
  const std::vector<starry::SearchResult> hits = index.search(query, 50, &skip);
  REQUIRE(hits.size() == 47);
  for (std::size_t i = 0; i < hits.size(); ++i) {
    CHECK(skip.find(hits[i].id) == skip.end());
  }
}

TEST_CASE("flat index edge cases: empty index, k = 0, k > n") {
  starry::FlatIndex index(4, starry::kL2);
  const float q[] = {0.0f, 0.0f, 0.0f, 0.0f};
  CHECK(index.search(q, 10, 0).empty());          // empty index
  index.add(1, q);
  CHECK(index.search(q, 0, 0).empty());           // k = 0
  CHECK(index.search(q, 10, 0).size() == 1);      // k > n -> clamp to n
}

// ---- VectorDB ----------------------------------------------------------------

TEST_CASE("insert/search/remove/get behave as documented") {
  starry::VectorDB db(4);
  const float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  const float v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
  const float v3[] = {-1.0f, 0.0f, 0.0f, 0.0f};

  CHECK(db.insert(10, v1) == starry::kOk);
  CHECK(db.insert(20, v2) == starry::kOk);
  CHECK(db.insert(10, v3) == starry::kDuplicateId);      // duplicate id
  CHECK(db.insert(30, std::vector<float>(3, 0.0f)) ==
        starry::kInvalidArgument);                        // wrong dimension

  CHECK(db.size() == 2);
  CHECK(db.capacity() == 2);

  // Query near v1: nearest must be id 10.
  const float q[] = {0.9f, 0.1f, 0.0f, 0.0f};
  const std::vector<starry::SearchResult> hits = db.search(q, 2);
  REQUIRE(hits.size() == 2);
  CHECK(hits[0].id == 10);

  // get() copies the payload; after remove() it disappears.
  std::vector<float> out;
  CHECK(db.get(10, &out));
  CHECK(out == std::vector<float>(v1, v1 + 4));
  CHECK(db.remove(10) == starry::kOk);
  CHECK(db.remove(10) == starry::kNotFound);  // double remove
  CHECK(db.remove(99) == starry::kNotFound);  // never existed
  CHECK(db.size() == 1);
  CHECK(db.capacity() == 2);                  // row is still there
  CHECK_FALSE(db.get(10, &out));

  // Deleted id must not surface in search any more.
  const std::vector<starry::SearchResult> after = db.search(q, 2);
  REQUIRE(after.size() == 1);
  CHECK(after[0].id == 20);

  // Re-inserting a deleted id revives it.
  CHECK(db.insert(10, v1) == starry::kOk);
  CHECK(db.size() == 2);
}

TEST_CASE("empty database search returns nothing") {
  starry::VectorDB db(3);
  const std::vector<float> q(3, 0.5f);
  CHECK(db.search(q, 5).empty());
  CHECK(db.search(q, 5).empty());
}

// ---- HNSW ---------------------------------------------------------------------

TEST_CASE("hnsw search reaches >=0.95 recall@10 against the flat oracle") {
  const std::size_t dim = 32;
  const std::size_t rows = 3000;
  const std::size_t k = 10;

  std::vector<float> data = make_random_vector(rows * dim, 21);
  starry::VectorDB flat(dim, starry::kL2, starry::kFlatIndex);
  starry::VectorDB hnsw(dim, starry::kL2, starry::kHnswIndex);
  hnsw.set_search_ef(128);
  for (std::size_t i = 0; i < rows; ++i) {
    const float* v = &data[i * dim];
    REQUIRE(flat.insert(i, v) == starry::kOk);
    REQUIRE(hnsw.insert(i, v) == starry::kOk);
  }

  double hits = 0.0;
  double total = 0.0;
  for (std::uint32_t q = 0; q < 30; ++q) {
    const std::vector<float> query = make_random_vector(dim, 1000 + q);
    const std::vector<starry::SearchResult> want =
        flat.search(&query[0], k);
    const std::vector<starry::SearchResult> got = hnsw.search(&query[0], k);
    REQUIRE(got.size() == k);
    std::unordered_set<starry::id_t> want_ids;
    for (std::size_t i = 0; i < want.size(); ++i) {
      want_ids.insert(want[i].id);
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (want_ids.count(got[i].id) != 0) {
        hits += 1.0;
      }
      // Reported distances must match the exact ones for shared ids
      // (same kernel, same vectors).
      for (std::size_t j = 0; j < want.size(); ++j) {
        if (want[j].id == got[i].id) {
          CHECK(std::fabs(want[j].distance - got[i].distance) < 1e-3f);
        }
      }
    }
    total += static_cast<double>(k);
  }
  CHECK(hits / total >= 0.95);
}

TEST_CASE("hnsw honours tombstones and save/load round-trips the graph") {
  const std::size_t dim = 16;
  const std::size_t rows = 500;
  std::vector<float> data = make_random_vector(rows * dim, 33);
  starry::VectorDB hnsw(dim, starry::kL2, starry::kHnswIndex);
  hnsw.set_search_ef(96);
  for (std::size_t i = 0; i < rows; ++i) {
    REQUIRE(hnsw.insert(static_cast<starry::id_t>(i), &data[i * dim]) ==
            starry::kOk);
  }
  // Delete a band of ids; deleted rows must never surface.
  for (std::size_t i = 0; i < rows; i += 3) {
    REQUIRE(hnsw.remove(static_cast<starry::id_t>(i)) == starry::kOk);
  }
  const std::vector<float> query = make_random_vector(dim, 77);
  const std::vector<starry::SearchResult> after_del =
      hnsw.search(&query[0], 50);
  for (std::size_t i = 0; i < after_del.size(); ++i) {
    CHECK(after_del[i].id % 3 != 0);
  }

  const char* path = "/tmp/starry_test_hnsw.bin";
  REQUIRE(hnsw.save(path) == starry::kOk);
  starry::Status status = starry::kOk;
  std::unique_ptr<starry::VectorDB> loaded = starry::VectorDB::load(path, &status);
  REQUIRE(loaded);
  CHECK(status == starry::kOk);
  CHECK(loaded->kind() == starry::kHnswIndex);
  loaded->set_search_ef(96);
  const std::vector<starry::SearchResult> after_load =
      loaded->search(&query[0], 50);
  CHECK(same_hits(after_del, after_load));  // graph restored bit-identically
  std::remove(path);
}

TEST_CASE("flat databases still load from a V1 file (back-compat)") {
  // A hand-rolled minimal STARRYV1 file with 2 rows of dim 2 (L2).
  const char* path = "/tmp/starry_test_v1.bin";
  FILE* f = std::fopen(path, "wb");
  REQUIRE(f != 0);
  const char magic[8] = {'S', 'T', 'A', 'R', 'R', 'Y', 'V', '1'};
  std::fwrite(magic, 1, 8, f);
  std::uint32_t metric = 0, pad = 0;  // l2, zero padding = flat kind
  std::fwrite(&metric, 4, 1, f);
  std::fwrite(&pad, 4, 1, f);
  std::uint64_t d = 2, n = 2;
  std::fwrite(&d, 8, 1, f);
  std::fwrite(&n, 8, 1, f);
  const float vecs[4] = {0.0f, 0.0f, 1.0f, 1.0f};
  std::fwrite(vecs, 4, 4, f);
  const std::uint64_t ids[2] = {7, 9};
  std::fwrite(ids, 8, 2, f);
  const std::uint64_t del = 0;
  std::fwrite(&del, 8, 1, f);
  std::fclose(f);

  starry::Status status = starry::kOk;
  std::unique_ptr<starry::VectorDB> db = starry::VectorDB::load(path, &status);
  REQUIRE(db);
  CHECK(status == starry::kOk);
  CHECK(db->kind() == starry::kFlatIndex);
  CHECK(db->size() == 2);
  const float q[] = {0.9f, 0.9f};
  const std::vector<starry::SearchResult> hits = db->search(q, 1);
  REQUIRE(hits.size() == 1);
  CHECK(hits[0].id == 9);
  std::remove(path);
}

// ---- persistence ---------------------------------------------------------------

TEST_CASE("save/load round-trip preserves data, search and tombstones") {
  const std::size_t dim = 16;
  const std::size_t rows = 200;
  starry::VectorDB db(dim, starry::kCosine);
  for (std::size_t i = 0; i < rows; ++i) {
    const std::vector<float> v = make_random_vector(dim, static_cast<std::uint32_t>(i));
    REQUIRE(db.insert(static_cast<starry::id_t>(i), &v[0]) == starry::kOk);
  }
  // Delete a few and keep the list to verify tombstones survive.
  std::vector<starry::id_t> deleted;
  for (std::size_t i = 0; i < rows; i += 37) {
    deleted.push_back(static_cast<starry::id_t>(i));
    REQUIRE(db.remove(static_cast<starry::id_t>(i)) == starry::kOk);
  }

  const char* path = "/tmp/starry_test_roundtrip.bin";
  REQUIRE(db.save(path) == starry::kOk);

  starry::Status status = starry::kOk;
  std::unique_ptr<starry::VectorDB> loaded = starry::VectorDB::load(path, &status);
  REQUIRE(loaded);
  CHECK(status == starry::kOk);

  CHECK(loaded->size() == db.size());
  CHECK(loaded->capacity() == db.capacity());
  CHECK(loaded->dim() == dim);

  // Identical search results before and after the round-trip.
  const std::vector<float> query = make_random_vector(dim, 4242);
  CHECK(same_hits(db.search(query, 25), loaded->search(query, 25)));

  // Tombstones travelled along.
  for (std::size_t i = 0; i < deleted.size(); ++i) {
    std::vector<float> v;
    CHECK_FALSE(loaded->get(deleted[i], &v));
  }
  std::remove(path);
}

TEST_CASE("load rejects a corrupted magic") {
  const char* path = "/tmp/starry_test_garbage.bin";
  FILE* f = std::fopen(path, "wb");
  REQUIRE(f != 0);
  std::fputs("NOTMAGIC-AND-SOME-PADDING-BYTES", f);
  std::fclose(f);
  starry::Status status = starry::kOk;
  CHECK(starry::VectorDB::load(path, &status) == std::unique_ptr<starry::VectorDB>());
  CHECK(status == starry::kIoError);
  std::remove(path);
}

// ---- insert_bulk & deterministic parallel HNSW build ------------------------

namespace {

// Reads a whole file into a byte string (for bit-identical graph
// comparison; save() output is fully determined by the database state,
// so equal files <=> equal graphs).
std::string slurp(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Builds an hnsw database from `data` via insert_bulk with `threads`
// workers and returns it; dim/rows derive from the buffer.
std::unique_ptr<starry::VectorDB> bulk_build(const std::vector<float>& data,
                                             std::size_t dim,
                                             std::size_t threads) {
  std::unique_ptr<starry::VectorDB> db(
      new starry::VectorDB(dim, starry::kL2, starry::kHnswIndex));
  db->set_search_ef(128);
  std::vector<starry::id_t> ids(data.size() / dim);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<starry::id_t>(i);
  }
  REQUIRE(db->insert_bulk(ids, data, threads) == starry::kOk);
  return db;
}

}  // namespace

TEST_CASE("insert_bulk on flat matches row-by-row insert and validates atomically") {
  const std::size_t dim = 16;
  const std::size_t rows = 300;
  const std::vector<float> data = make_random_vector(rows * dim, 5);
  std::vector<starry::id_t> ids(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    ids[i] = 500 + static_cast<starry::id_t>(i);  // non-trivial ids
  }

  starry::VectorDB serial(dim, starry::kL2, starry::kFlatIndex);
  starry::VectorDB bulk(dim, starry::kL2, starry::kFlatIndex);
  for (std::size_t i = 0; i < rows; ++i) {
    REQUIRE(serial.insert(ids[i], &data[i * dim]) == starry::kOk);
  }
  REQUIRE(bulk.insert_bulk(ids, data, 0) == starry::kOk);
  CHECK(bulk.size() == serial.size());
  CHECK(bulk.capacity() == serial.capacity());

  // Identical search behaviour (exact index => identical hits).
  for (std::uint32_t s = 0; s < 5; ++s) {
    const std::vector<float> query = make_random_vector(dim, 600 + s);
    CHECK(same_hits(serial.search(query, 25), bulk.search(query, 25)));
  }

  // ---- atomic validation -------------------------------------------------
  const std::size_t size_before = bulk.size();
  std::vector<starry::id_t> bad_ids(ids);
  bad_ids.push_back(ids[7]);  // duplicate inside the batch
  CHECK(bulk.insert_bulk(bad_ids, std::vector<float>((rows + 1) * dim, 0.0f),
                         0) == starry::kDuplicateId);
  CHECK(bulk.insert_bulk(ids, data, 0) == starry::kDuplicateId);  // live dup
  CHECK(bulk.insert_bulk(ids, std::vector<float>(3, 0.0f), 0) ==
        starry::kInvalidArgument);                                // size mismatch
  CHECK(bulk.insert_bulk(std::vector<starry::id_t>(),
                         std::vector<float>(), 0) == starry::kOk);  // empty ok
  CHECK(bulk.size() == size_before);  // nothing leaked in

  // ---- revive semantics matches insert() ----------------------------------
  REQUIRE(bulk.remove(ids[0]) == starry::kOk);
  CHECK(bulk.size() == size_before - 1);
  const std::vector<float> one(&data[0], &data[dim]);
  REQUIRE(bulk.insert_bulk({ids[0]}, one, 0) == starry::kOk);
  CHECK(bulk.size() == size_before);
  std::vector<float> got;
  CHECK(bulk.get(ids[0], &got));
}

TEST_CASE("hnsw bulk build is bit-identical across thread counts") {
  const std::size_t dim = 16;
  // 3000 rows = 11 full waves (256) + a partial 184-row wave: exercises
  // both the multi-wave path and the final short wave.
  const std::size_t rows = 3000;
  const std::vector<float> data = make_random_vector(rows * dim, 9);

  const std::size_t thread_counts[] = {1, 3, 8};
  std::vector<std::string> files;
  for (std::size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]);
       ++t) {
    std::unique_ptr<starry::VectorDB> db =
        bulk_build(data, dim, thread_counts[t]);
    CHECK(db->size() == rows);
    char path[64];
    std::snprintf(path, sizeof(path), "/tmp/starry_test_bulk_t%zu.bin",
                  thread_counts[t]);
    REQUIRE(db->save(path) == starry::kOk);
    files.push_back(slurp(path));
    std::remove(path);
  }
  CHECK(files[0].size() > 0);
  CHECK(files[0] == files[1]);  // 1 vs 3 workers
  CHECK(files[0] == files[2]);  // 1 vs 8 workers
}

TEST_CASE("hnsw bulk build recall is on par with serial and round-trips") {
  const std::size_t dim = 16;
  const std::size_t rows = 3000;
  const std::size_t k = 10;
  const std::vector<float> data = make_random_vector(rows * dim, 13);

  starry::VectorDB flat(dim, starry::kL2, starry::kFlatIndex);
  starry::VectorDB serial(dim, starry::kL2, starry::kHnswIndex);
  serial.set_search_ef(128);
  for (std::size_t i = 0; i < rows; ++i) {
    REQUIRE(flat.insert(static_cast<starry::id_t>(i), &data[i * dim]) ==
            starry::kOk);
    REQUIRE(serial.insert(static_cast<starry::id_t>(i), &data[i * dim]) ==
            starry::kOk);
  }
  std::unique_ptr<starry::VectorDB> bulk = bulk_build(data, dim, 4);

  double recall_serial = 0.0;
  double recall_bulk = 0.0;
  const std::size_t nq = 30;
  for (std::uint32_t q = 0; q < nq; ++q) {
    const std::vector<float> query = make_random_vector(dim, 2000 + q);
    const std::vector<starry::SearchResult> want = flat.search(query, k);
    std::unordered_set<starry::id_t> want_ids;
    for (std::size_t i = 0; i < want.size(); ++i) {
      want_ids.insert(want[i].id);
    }
    double hs = 0.0, hb = 0.0;
    const std::vector<starry::SearchResult> got_s = serial.search(query, k);
    const std::vector<starry::SearchResult> got_b = bulk->search(query, k);
    REQUIRE(got_b.size() == k);
    for (std::size_t i = 0; i < k; ++i) {
      if (want_ids.count(got_s[i].id) != 0) hs += 1.0;
      if (want_ids.count(got_b[i].id) != 0) hb += 1.0;
    }
    recall_serial += hs;
    recall_bulk += hb;
  }
  recall_serial /= static_cast<double>(nq * k);
  recall_bulk /= static_cast<double>(nq * k);
  CHECK(recall_bulk >= 0.95);
  CHECK(std::fabs(recall_serial - recall_bulk) <= 0.02);

  // The bulk-built graph survives save/load bit-identically and keeps
  // serving the same results.
  const char* path = "/tmp/starry_test_bulk_rt.bin";
  REQUIRE(bulk->save(path) == starry::kOk);
  starry::Status status = starry::kOk;
  std::unique_ptr<starry::VectorDB> loaded = starry::VectorDB::load(path, &status);
  REQUIRE(loaded);
  CHECK(status == starry::kOk);
  CHECK(loaded->kind() == starry::kHnswIndex);
  loaded->set_search_ef(128);
  const std::vector<float> query = make_random_vector(dim, 3000);
  CHECK(same_hits(bulk->search(query, 50), loaded->search(query, 50)));
  std::remove(path);
}
