#pragma once
#include <array>
#include <cmath>
namespace gemm_policy_pilot {
using Features = std::array<double, 11>;
// Feature order: stages, loop_trips, tile_m, tile_n, tile_k, warps, a_descriptor, a_k_contiguous, a_compute, b_descriptor, input_bits
inline bool apply_unroll(const Features &x, bool isBmg = true, bool isBf16 = true) {
  if (!isBmg || !isBf16) return false;
  if (!std::isfinite(x[0]) || x[0] < 1.0 || x[0] > 1.0) return false;
  if (x[0] != 1.0) return false;
  if (!std::isfinite(x[1]) || x[1] < 32.0 || x[1] > 768.0) return false;
  if (!std::isfinite(x[2]) || x[2] < 256.0 || x[2] > 256.0) return false;
  if (x[2] != 256.0) return false;
  if (!std::isfinite(x[3]) || x[3] < 256.0 || x[3] > 256.0) return false;
  if (x[3] != 256.0) return false;
  if (!std::isfinite(x[4]) || x[4] < 32.0 || x[4] > 32.0) return false;
  if (x[4] != 32.0) return false;
  if (!std::isfinite(x[5]) || x[5] < 32.0 || x[5] > 32.0) return false;
  if (x[5] != 32.0) return false;
  if (!std::isfinite(x[6]) || x[6] < 0.0 || x[6] > 1.0) return false;
  if (x[6] != 0.0 && x[6] != 1.0) return false;
  if (!std::isfinite(x[7]) || x[7] < 0.0 || x[7] > 1.0) return false;
  if (x[7] != 0.0 && x[7] != 1.0) return false;
  if (!std::isfinite(x[8]) || x[8] < 0.0 || x[8] > 0.0) return false;
  if (x[8] != 0.0) return false;
  if (!std::isfinite(x[9]) || x[9] < 0.0 || x[9] > 1.0) return false;
  if (x[9] != 0.0 && x[9] != 1.0) return false;
  if (!std::isfinite(x[10]) || x[10] < 16.0 || x[10] > 16.0) return false;
  if (x[10] != 16.0) return false;
  return true;
}
inline bool apply_layout(const Features &x, bool isBmg = true, bool isBf16 = true) {
  if (!isBmg || !isBf16) return false;
  if (!std::isfinite(x[0]) || x[0] < 2.0 || x[0] > 4.0) return false;
  if (x[0] != 2.0 && x[0] != 3.0 && x[0] != 4.0) return false;
  if (!std::isfinite(x[1]) || x[1] < 257.0 || x[1] > 768.0) return false;
  if (!std::isfinite(x[2]) || x[2] < 256.0 || x[2] > 256.0) return false;
  if (x[2] != 256.0) return false;
  if (!std::isfinite(x[3]) || x[3] < 256.0 || x[3] > 256.0) return false;
  if (x[3] != 256.0) return false;
  if (!std::isfinite(x[4]) || x[4] < 32.0 || x[4] > 32.0) return false;
  if (x[4] != 32.0) return false;
  if (!std::isfinite(x[5]) || x[5] < 32.0 || x[5] > 32.0) return false;
  if (x[5] != 32.0) return false;
  if (!std::isfinite(x[6]) || x[6] < 0.0 || x[6] > 1.0) return false;
  if (x[6] != 0.0 && x[6] != 1.0) return false;
  if (!std::isfinite(x[7]) || x[7] < 0.0 || x[7] > 1.0) return false;
  if (x[7] != 0.0 && x[7] != 1.0) return false;
  if (!std::isfinite(x[8]) || x[8] < 0.0 || x[8] > 1.0) return false;
  if (x[8] != 0.0 && x[8] != 1.0) return false;
  if (!std::isfinite(x[9]) || x[9] < 0.0 || x[9] > 1.0) return false;
  if (x[9] != 0.0 && x[9] != 1.0) return false;
  if (!std::isfinite(x[10]) || x[10] < 16.0 || x[10] > 16.0) return false;
  if (x[10] != 16.0) return false;
  if (x[0] <= 2.5) {
    if (x[6] <= 0.5) {
      return false;
    } else {
      if (x[7] <= 0.5) {
        return false;
      } else {
        return true;
      }
    }
  } else {
    if (x[8] <= 0.5) {
      return false;
    } else {
      return true;
    }
  }
}
} // namespace gemm_policy_pilot
