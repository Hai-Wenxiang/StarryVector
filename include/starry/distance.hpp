// StarryVector - distance kernels.
//
// Two families of kernels live side by side:
//
//   * the scalar reference implementations below - deliberately the most
//     primitive loops possible.  They are the correctness foundation of
//     the whole engine and the oracle every optimised kernel is validated
//     against, so clarity beats cleverness here;
//   * SIMD kernels (AVX2 when the build machine and CPU allow), selected
//     at RUNTIME by resolve_distance_fn() below - never at compile time,
//     so one binary stays portable across heterogeneous machines.
//
// Setting the environment variable STARRY_FORCE_SCALAR=1 disables SIMD
// dispatch entirely (debugging, differential testing, baseline runs).
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

// Scalar-only resolver: always returns one of the three reference
// kernels above, regardless of CPU capabilities.
DistanceFn resolve_distance_fn_scalar(Metric metric);

// Resolves a Metric to its kernel function, dispatching to the best SIMD
// variant supported by the RUNNING CPU (AVX2 when available) unless
// STARRY_FORCE_SCALAR=1 is set in the environment.  Never returns null
// for a valid Metric value.
DistanceFn resolve_distance_fn(Metric metric);

// True when SIMD kernels were compiled in (informational, for tests and
// the benchmark banner).
bool simd_kernels_available();

}  // namespace starry

#endif  // STARRY_DISTANCE_HPP
