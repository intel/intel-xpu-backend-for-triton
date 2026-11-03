#pragma once
#include <array>
#include <cmath>
namespace gemm_policy_alignment {
using Features = std::array<double, 13>;
// Feature order: stages, loop_trips, tile_m, tile_n, tile_k, warps, a_descriptor, a_k_contiguous, a_compute, b_descriptor, input_bits, a_pitch_alignment, b_pitch_alignment
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
  if (!std::isfinite(x[6]) || x[6] < 1.0 || x[6] > 1.0) return false;
  if (x[6] != 1.0) return false;
  if (!std::isfinite(x[7]) || x[7] < 0.0 || x[7] > 1.0) return false;
  if (x[7] != 0.0 && x[7] != 1.0) return false;
  if (!std::isfinite(x[8]) || x[8] < 0.0 || x[8] > 0.0) return false;
  if (x[8] != 0.0) return false;
  if (!std::isfinite(x[9]) || x[9] < 1.0 || x[9] > 1.0) return false;
  if (x[9] != 1.0) return false;
  if (!std::isfinite(x[10]) || x[10] < 16.0 || x[10] > 16.0) return false;
  if (x[10] != 16.0) return false;
  if (!std::isfinite(x[11]) || x[11] < 64.0 || x[11] > 32768.0) return false;
  if (!std::isfinite(x[12]) || x[12] < 64.0 || x[12] > 16384.0) return false;
  if (x[7] <= 0.5) {
    return true;
  } else {
    if (x[11] <= 96.0) {
      return false;
    } else {
      if (x[1] <= 366.0) {
        return false;
      } else {
        return true;
      }
    }
  }
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
  if (!std::isfinite(x[6]) || x[6] < 1.0 || x[6] > 1.0) return false;
  if (x[6] != 1.0) return false;
  if (!std::isfinite(x[7]) || x[7] < 0.0 || x[7] > 1.0) return false;
  if (x[7] != 0.0 && x[7] != 1.0) return false;
  if (!std::isfinite(x[8]) || x[8] < 0.0 || x[8] > 1.0) return false;
  if (x[8] != 0.0 && x[8] != 1.0) return false;
  if (!std::isfinite(x[9]) || x[9] < 1.0 || x[9] > 1.0) return false;
  if (x[9] != 1.0) return false;
  if (!std::isfinite(x[10]) || x[10] < 16.0 || x[10] > 16.0) return false;
  if (x[10] != 16.0) return false;
  if (!std::isfinite(x[11]) || x[11] < 64.0 || x[11] > 131072.0) return false;
  if (!std::isfinite(x[12]) || x[12] < 64.0 || x[12] > 16384.0) return false;
  if (x[7] <= 0.5) {
    return false;
  } else {
    if (x[12] <= 12288.0) {
      if (x[0] <= 2.5) {
        if (x[1] <= 366.0) {
          if (x[8] <= 0.5) {
            return false;
          } else {
            return true;
          }
        } else {
          return true;
        }
      } else {
        if (x[1] <= 727.0) {
          if (x[11] <= 6144.0) {
            return false;
          } else {
            return true;
          }
        } else {
          return false;
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
}
} // namespace gemm_policy_alignment
