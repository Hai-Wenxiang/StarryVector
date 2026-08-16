// Scalar reference kernels + runtime SIMD dispatch.
//
// The scalar implementations below are the oracle for SIMD kernel unit
// tests: for random inputs the SIMD result must match the scalar one
// within a small tolerance (different summation orders make bit-exact
// equality impossible in general).
//
// resolve_distance_fn() picks the fastest kernel family the RUNNING CPU
// supports.  Detection happens once per call (callers cache the result);
// STARRY_FORCE_SCALAR=1 in the environment forces the scalar path.
#include "starry/distance.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace starry {

// SIMD kernels, defined in distance_avx2.cpp (compiled with -mavx2 -mfma
// when STARRY_HAVE_AVX2 is defined by the build).  Only called after
// runtime confirmation that the CPU supports them.
#if defined(STARRY_HAVE_AVX2)
float l2_distance_avx2(const float* a, const float* b, std::size_t dim);
float inner_product_distance_avx2(const float* a, const float* b,
                                  std::size_t dim);
#endif

namespace {

// True when the process may use AVX2+FMA kernels.
bool cpu_has_avx2() {
#if defined(STARRY_HAVE_AVX2) && \
    (defined(__x86_64__) || defined(__i386__))
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

bool force_scalar_env() {
  const char* v = std::getenv("STARRY_FORCE_SCALAR");
  return v != 0 && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

bool avx2_usable() {
#if defined(STARRY_HAVE_AVX2)
  static const bool cached = cpu_has_avx2() && !force_scalar_env();
  return cached;
#else
  return false;
#endif
}

}  // namespace

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

DistanceFn resolve_distance_fn_scalar(Metric metric) {
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

DistanceFn resolve_distance_fn(Metric metric) {
  if (avx2_usable() && metric != kCosine) {
    // Cosine keeps the scalar kernel: FlatIndex pre-normalises cosine
    // data and resolves to the inner-product kernel itself (R1), so the
    // cosine path here is only hit by direct callers of the reference.
#if defined(STARRY_HAVE_AVX2)
    switch (metric) {
      case kL2:
        return &l2_distance_avx2;
      case kInnerProduct:
        return &inner_product_distance_avx2;
      case kCosine:
        break;  // unreachable, guarded above
    }
#endif
  }
  return resolve_distance_fn_scalar(metric);
}

bool simd_kernels_available() {
#if defined(STARRY_HAVE_AVX2)
  return true;  // AVX2 kernels are compiled in; runtime gate applies per CPU
#else
  return false;
#endif
}

}  // namespace starry
