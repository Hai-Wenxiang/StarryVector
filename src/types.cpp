// Implementation of the small helpers declared in types.hpp.
#include "starry/types.hpp"

#include <cstring>

namespace starry {

const char* metric_name(Metric metric) {
  switch (metric) {
    case kL2:
      return "l2";
    case kInnerProduct:
      return "ip";
    case kCosine:
      return "cosine";
  }
  // Unreachable for valid Metric values; returned defensively so the
  // function never dereferences garbage in release builds either.
  return "unknown";
}

bool parse_metric(const char* name, Metric* out) {
  if (name == 0 || out == 0) {
    return false;
  }
  if (std::strcmp(name, "l2") == 0) {
    *out = kL2;
    return true;
  }
  if (std::strcmp(name, "ip") == 0) {
    *out = kInnerProduct;
    return true;
  }
  if (std::strcmp(name, "cosine") == 0) {
    *out = kCosine;
    return true;
  }
  return false;
}

}  // namespace starry
