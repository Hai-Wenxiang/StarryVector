// Reference (scalar) implementations of the distance kernels.
//
// Every kernel is a straightforward loop.  They will later serve as the
// oracle for SIMD kernel unit tests: for random inputs the SIMD result
// must match the scalar one within a small tolerance (different
// summation orders make bit-exact equality impossible in general).
#include "starry/distance.hpp"

#include <cmath>

namespace starry {

float l2_distance(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float inner_product_distance(const float* a, const float* b, std::size_t dim) {
  float dot = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return -dot;
}

float cosine_distance(const float* a, const float* b, std::size_t dim) {
  float dot = 0.0f;
  float na = 0.0f;
  float nb = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  // Guard against zero vectors: an undefined direction is treated as
  // "orthogonal" (distance 1) instead of producing NaN.
  if (na == 0.0f || nb == 0.0f) {
    return 1.0f;
  }
  const float similarity = dot / (sqrtf(na) * sqrtf(nb));
  return 1.0f - similarity;
}

DistanceFn resolve_distance_fn(Metric metric) {
  switch (metric) {
    case kL2:
      return &l2_distance;
    case kInnerProduct:
      return &inner_product_distance;
    case kCosine:
      return &cosine_distance;
  }
  return &l2_distance;  // defensive fallback for invalid enum values
}

}  // namespace starry
