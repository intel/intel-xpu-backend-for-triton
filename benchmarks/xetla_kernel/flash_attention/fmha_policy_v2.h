#pragma once

#include "xetla.hpp"
#include <sycl.hpp>

namespace gpu::xetla {

struct fmha_policy_base {
  static constexpr uint32_t accum_step = 32;
  static constexpr uint32_t stages = 3;
  static constexpr uint32_t sync_freq = 0;
};

/*
Note:
  kHm / kSgHm == kBc / kSgBc
  kSgHm and kSgBc should be a multiple of 16
  kSgBr should be a multiple of 8
*/

struct fmha_policy_64x128x64 : fmha_policy_base {
  static constexpr uint32_t kBr = 128;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 64;
  static constexpr uint32_t kSgBc = 64;
  static constexpr uint32_t kHm = 64;
  static constexpr uint32_t kSgHm = 64;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x128x128 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 128;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 128;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_8x256x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 8;
  static constexpr uint32_t kSgBr = 8;
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_32x256x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 32;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 256;
  static constexpr uint32_t kSgBc = 32;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

struct fmha_policy_64x512x256 : fmha_policy_base {
  static constexpr uint32_t kBr = 64;
  static constexpr uint32_t kSgBr = 16;
  static constexpr uint32_t kBc = 512;
  static constexpr uint32_t kSgBc = 64;
  static constexpr uint32_t kHm = 256;
  static constexpr uint32_t kSgHm = 32;
  static constexpr uint32_t thread_num = (kBr / kSgBr) * (kBc / kSgBc);
};

} // namespace gpu::xetla
