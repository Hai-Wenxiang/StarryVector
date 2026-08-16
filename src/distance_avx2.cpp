// AVX2 distance kernels.
//
// This translation unit is compiled with -mavx2 -mfma (see CMakeLists);
// it must NEVER be called before the runtime CPU check in distance.cpp
// (resolve_distance_fn) has confirmed AVX2 + FMA support.
//
// Structure of both kernels:
//   * main loop consumes 32 floats (4x 256-bit vectors) per iteration
//     with four independent accumulators to hide FMA latency;
//   * 8-float loop for the next multiple of 8;
//   * scalar tail keeps arbitrary dims exact.
//
// Results differ from the scalar oracle by summation order (IEEE-754
// non-associativity); unit tests therefore compare with a tolerance.
#include <cstddef>

#include <immintrin.h>

#include "starry/distance.hpp"

namespace starry {

float l2_distance_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256 a0 = _mm256_loadu_ps(a + i);
    const __m256 a1 = _mm256_loadu_ps(a + i + 8);
    const __m256 a2 = _mm256_loadu_ps(a + i + 16);
    const __m256 a3 = _mm256_loadu_ps(a + i + 24);
    const __m256 b0 = _mm256_loadu_ps(b + i);
    const __m256 b1 = _mm256_loadu_ps(b + i + 8);
    const __m256 b2 = _mm256_loadu_ps(b + i + 16);
    const __m256 b3 = _mm256_loadu_ps(b + i + 24);
    const __m256 d0 = _mm256_sub_ps(a0, b0);
    const __m256 d1 = _mm256_sub_ps(a1, b1);
    const __m256 d2 = _mm256_sub_ps(a2, b2);
    const __m256 d3 = _mm256_sub_ps(a3, b3);
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    acc2 = _mm256_fmadd_ps(d2, d2, acc2);
    acc3 = _mm256_fmadd_ps(d3, d3, acc3);
  }
  __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                             _mm256_add_ps(acc2, acc3));
  for (; i + 8 <= dim; i += 8) {
    const __m256 av = _mm256_loadu_ps(a + i);
    const __m256 bv = _mm256_loadu_ps(b + i);
    const __m256 d = _mm256_sub_ps(av, bv);
    acc = _mm256_fmadd_ps(d, d, acc);
  }

  // Horizontal sum of the 8 lanes.
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);           // [l1 l1 l3 l3]
  __m128 sums = _mm_add_ps(lo, shuf);          // [l0+l1 .. l2+l3 ..]
  shuf = _mm_movehl_ps(shuf, sums);            // [l2+l3 ..]
  sums = _mm_add_ss(sums, shuf);
  float sum = _mm_cvtss_f32(sums);

  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float inner_product_distance_avx2(const float* a, const float* b,
                                  std::size_t dim) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256 a0 = _mm256_loadu_ps(a + i);
    const __m256 a1 = _mm256_loadu_ps(a + i + 8);
    const __m256 a2 = _mm256_loadu_ps(a + i + 16);
    const __m256 a3 = _mm256_loadu_ps(a + i + 24);
    const __m256 b0 = _mm256_loadu_ps(b + i);
    const __m256 b1 = _mm256_loadu_ps(b + i + 8);
    const __m256 b2 = _mm256_loadu_ps(b + i + 16);
    const __m256 b3 = _mm256_loadu_ps(b + i + 24);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
    acc3 = _mm256_fmadd_ps(a3, b3, acc3);
  }
  __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                             _mm256_add_ps(acc2, acc3));
  for (; i + 8 <= dim; i += 8) {
    const __m256 av = _mm256_loadu_ps(a + i);
    const __m256 bv = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(av, bv, acc);
  }

  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  float dot = _mm_cvtss_f32(sums);

  for (; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return -dot;
}

}  // namespace starry
