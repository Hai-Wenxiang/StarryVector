// Implementation of the brute-force FlatIndex.
#include "starry/flat_index.hpp"

#include <algorithm>
#include <cmath>

namespace starry {

namespace {

// Max-heap ordering over the candidate buffer: the heap top is the
// WORST (largest-distance) hit, so evictions and rejections are O(1).
struct Farther {
  bool operator()(const SearchResult& a, const SearchResult& b) const {
    return a.distance < b.distance;
  }
};

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
      metric_(metric),
      // With pre-normalisation the cosine scan is a plain inner product
      // (SIMD-dispatched like every other metric); search() adds the
      // constant +1 to get back to cosine distance.
      dist_(metric == kCosine ? resolve_distance_fn(kInnerProduct)
                              : resolve_distance_fn(metric)),
      cosine_pre_(metric == kCosine) {
  // Reserve nothing yet; the caller-driven insert pattern of the first
  // milestone relies on vector growth being amortised O(1), which is
  // good enough for bulk loads of a few million vectors.
}

void FlatIndex::materialise() {
  if (view_storage_ == 0) {
    return;
  }
  storage_.assign(view_storage_, view_storage_ + view_n_ * dim_);
  ids_.assign(view_ids_, view_ids_ + view_n_);
  view_storage_ = 0;
  view_ids_ = 0;
  view_n_ = 0;
}

void FlatIndex::adopt(const float* storage, const id_t* ids,
                      std::size_t n) {
  // Adopting replaces ALL content; a prior adopted view is simply
  // dropped (its owner - the MappedFile in VectorDB - keeps the bytes
  // alive independently).
  storage_.clear();
  storage_.shrink_to_fit();
  ids_.clear();
  ids_.shrink_to_fit();
  view_storage_ = storage;
  view_ids_ = ids;
  view_n_ = n;
}

void FlatIndex::add(id_t id, const float* vec) {
  // First write after a zero-copy adopt pays one full copy here.
  materialise();
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
  std::vector<SearchResult> best;  // heap of the k best, worst on top
  if (k == 0) {
    return best;
  }

  // Cosine searches normalise the query once instead of paying two norm
  // reductions per database row.
  std::vector<float> query_scratch;
  const float* q = query;
  if (cosine_pre_) {
    query_scratch.assign(query, query + dim_);
    l2_normalize(&query_scratch[0], dim_);
    q = &query_scratch[0];
  }

  const std::size_t n = ids_.size() + view_n_;  // == size()
  const float* rows = data();
  const id_t* ids = ids_data();
  float worst = 0.0f;  // distance of the current k-th best (heap top)
  for (std::size_t i = 0; i < n; ++i) {
    float d = dist_(q, rows + i * dim_, dim_);
    if (cosine_pre_) {
      // ip kernel returned -dot; unit vectors: cosine distance = 1 - dot.
      d += 1.0f;
    }

    // Cheap rejection first: when the buffer is full, anything not
    // better than the current k-th hit is dropped without touching the
    // skip set (avoids a hash lookup for the common case).
    if (best.size() == k && d >= worst) {
      continue;
    }
    // Second rejection: soft-deleted ids pretend not to exist.
    if (skip != 0 && skip->find(ids[i]) != skip->end()) {
      continue;
    }

    SearchResult hit;
    hit.id = ids[i];
    hit.distance = d;
    if (best.size() < k) {
      best.push_back(hit);
      std::push_heap(best.begin(), best.end(), Farther());
      worst = best.front().distance;
    } else {
      std::pop_heap(best.begin(), best.end(), Farther());
      best.back() = hit;
      std::push_heap(best.begin(), best.end(), Farther());
      worst = best.front().distance;
    }
  }

  // The caller contract is ascending distance order.
  std::sort_heap(best.begin(), best.end(), Farther());
  return best;
}

}  // namespace starry
