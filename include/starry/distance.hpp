// StarryVector - scalar distance kernels.
//
// These are deliberately the most primitive implementations possible:
// plain loops over the components, no SIMD, no blocking, no unrolling.
// Rationale: this milestone (M0) is the correctness foundation of the
// whole engine.  Optimised kernels (SSE/AVX2/AVX-512 with runtime CPU
// dispatch) will be validated AGAINST these reference implementations,
// so clarity beats cleverness here.
//
// Conventions (see also types.hpp):
//   * a and b point to exactly `dim` consecutive floats.
//   * the returned value is a "distance": smaller = closer.
#ifndef STARRY_DISTANCE_HPP
#define STARRY_DISTANCE_HPP

#include <cstddef>

#include "starry/types.hpp"

namespace starry {

// Function pointer type shared by every kernel.  The FlatIndex stores one
// resolved kernel and calls it through this pointer for each candidate
// vector.  (A plain function pointer keeps call overhead minimal and is
// fully compatible with a future C shim.)
typedef float (*DistanceFn)(const float* a, const float* b, std::size_t dim);

// Squared Euclidean distance.  We use the squared form (no sqrt) because
// ordering is preserved and sqrt is relatively expensive.
float l2_distance(const float* a, const float* b, std::size_t dim);

// Negated dot product: -dot(a, b).  Maximising similarity therefore
// minimises this value.
float inner_product_distance(const float* a, const float* b, std::size_t dim);

// Cosine distance: 1 - dot(a,b) / (|a| * |b|).  Zero means identical
// direction; 2 means exactly opposite.  Degenerate zero vectors are
// treated as distance 1 (orthogonal) to avoid NaNs.
float cosine_distance(const float* a, const float* b, std::size_t dim);

// Resolves a Metric to its kernel function.  Never returns null for a
// valid Metric value.
DistanceFn resolve_distance_fn(Metric metric);

}  // namespace starry

#endif  // STARRY_DISTANCE_HPP
