// Implementation of the brute-force FlatIndex.
#include "starry/flat_index.hpp"

namespace starry {

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), dist_(resolve_distance_fn(metric)) {
  // Reserve nothing yet; the caller-driven insert pattern of the first
  // milestone relies on vector growth being amortised O(1), which is
  // good enough for bulk loads of a few million vectors.
}

void FlatIndex::add(id_t id, const float* vec) {
  // Bulk append of one row keeps the buffer contiguous (and therefore
  // prefetcher-friendly even in this scalar version).
  storage_.insert(storage_.end(), vec, vec + dim_);
  ids_.push_back(id);
}

std::vector<SearchResult> FlatIndex::search(
    const float* query, std::size_t k,
    const std::unordered_set<id_t>* skip) const {
  std::vector<SearchResult> best;  // kept sorted by ascending distance
  if (k == 0) {
    return best;
  }
  best.reserve(k + 1);

  const std::size_t n = ids_.size();
  for (std::size_t i = 0; i < n; ++i) {
    const float d = dist_(query, &storage_[i * dim_], dim_);

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
