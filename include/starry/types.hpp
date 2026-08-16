// StarryVector - common types shared across the whole library.
//
// This header is intentionally dependency-light (only <cstdint>) so that it
// can later be reused by a plain-C compatibility shim without dragging in
// any C++ machinery.
//
// Language baseline: strict C++11.  Do not use features from C++14 or later
// anywhere in this project (no std::make_unique, no string_view, ...).
#ifndef STARRY_TYPES_HPP
#define STARRY_TYPES_HPP

#include <cstdint>

namespace starry {

// Every vector stored in the database is addressed by a caller-supplied
// 64-bit integer id.  Ids are opaque to the engine: the caller may use row
// numbers, content hashes, or any other stable numbering scheme.
typedef std::uint64_t id_t;

// Distance / similarity metric used for search.
//
// All metrics are normalised internally so that SMALLER DISTANCE ALWAYS
// MEANS CLOSER.  This gives every index and every query path one single
// sorting convention:
//
//   kL2           squared Euclidean distance      sum((a-b)^2)
//   kInnerProduct negated dot product             -dot(a, b)
//   kCosine       cosine distance                 1 - cos(a, b)
//
// Note: for kInnerProduct/kCosine the raw "distance" can be negative; that
// is fine, the convention is only about the ordering.
enum Metric {
  kL2 = 0,
  kInnerProduct = 1,
  kCosine = 2,
};

// A single hit returned by a search.  Result lists are always sorted by
// ascending distance, therefore hits[0] is the nearest neighbour.
struct SearchResult {
  id_t id;
  float distance;
};

// Returns a short lowercase name for a metric: "l2", "ip" or "cosine".
// Used by the benchmark CLI, the validation harness and the on-disk format.
const char* metric_name(Metric metric);

// Parses one of the names produced by metric_name().  Returns false (and
// leaves *out untouched) when the name is not recognised.
bool parse_metric(const char* name, Metric* out);

}  // namespace starry

#endif  // STARRY_TYPES_HPP
