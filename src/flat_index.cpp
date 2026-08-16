// Implementation of the brute-force FlatIndex.
#include "starry/flat_index.hpp"

#include <cmath>

namespace starry {

namespace {

// L2-normalises `dim` floats in place; a zero vector is left unchanged
// (the caller relies on it behaving as "distance 1 to everything").
void l2_normalize(float* v, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += v[i] * v[i];
  }
  if (sum > 0.0f) {
    const float inv = 1.0f / std::sqrt(sum);
    for (std::size_t i = 0; i < dim; ++i) {
      v[i] *= inv;
    }
  }
}

}  // namespace

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim),
      // With pre-normalisation the cosine scan is a plain inner product;
      // search() adds the constant +1 to get back to cosine distance.
      dist_(metric == kCosine ? &inner_product_distance
                              : resolve_distance_fn(metric)),
      cosine_pre_(metric == kCosine) {
  // Reserve nothing yet; the caller-driven insert pattern of the first
  // milestone relies on vector growth being amortised O(1), which is
  // good enough for bulk loads of a few million vectors.
}

void FlatIndex::add(id_t id, const float* vec) {
  // Bulk append of one row keeps the buffer contiguous (and therefore
  // prefetcher-friendly even in this scalar version).
  storage_.insert(storage_.end(), vec, vec + dim_);
  ids_.push_back(id);
  if (cosine_pre_) {
    l2_normalize(&storage_[storage_.size() - dim_], dim_);
  }
}

std::vector<SearchResult> FlatIndex::search(
    const float* query, std::size_t k,
    const std::unordered_set<id_t>* skip) const {
  std::vector<SearchResult> best;  // kept sorted by ascending distance
  if (k == 0) {
    return best;
  }
  best.reserve(k + 1);

  // Cosine searches normalise the query once instead of paying two norm
  // reductions per database row.  C++17: scratch is thread-local by
  // lifetime, sized per call; dim_ is small so this stays cache friendly.
  std::vector<float> query_scratch;
  const float* q = query;
  if (cosine_pre_) {
    query_scratch.assign(query, query + dim_);
    l2_normalize(&query_scratch[0], dim_);
    q = &query_scratch[0];
  }

  const std::size_t n = ids_.size();
  for (std::size_t i = 0; i < n; ++i) {
    float d = dist_(q, &storage_[i * dim_], dim_);
    if (cosine_pre_) {
      // ip kernel returned -dot; unit vectors: cosine distance = 1 - dot.
      d += 1.0f;
    }

    // Cheap rejection first: if the buffer is full and the candidate is
    // not better than the current k-th hit, drop it without touching
    // the skip set (avoids a hash lookup for the common case).
    if (best.size() == k && d >= best[k - 1].distance) {
      continue;
    }
    // Second rejection: soft-deleted ids pretend not to exist.
    if (skip != 0 && skip->find(ids_[i]) != skip->end()) {
      continue;
    }

    // Insert the candidate into its sorted position.  A shift-back
    // insertion over at most k elements is the classic primitive top-k
    // selection; k is small (typically 10..100) so this beats any
    // heap for clarity while staying cache friendly.
    SearchResult hit;
    hit.id = ids_[i];
    hit.distance = d;
    best.push_back(hit);
    std::size_t j = best.size() - 1;
    while (j > 0 && best[j - 1].distance > d) {
      best[j] = best[j - 1];
      --j;
    }
    best[j] = hit;
    if (best.size() > k) {
      best.pop_back();
    }
  }
  return best;
}

}  // namespace starry
