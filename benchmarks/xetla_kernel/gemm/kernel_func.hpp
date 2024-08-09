/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#pragma once

#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

namespace gpu::xetla::kernel {
template <int wg_num_m_, gpu_arch arch_tag_> struct group_swizzle_m_first {
public:
  static constexpr gpu_arch arch_tag = arch_tag_;
  inline group_swizzle_m_first() = default;

  // get dim0 group id
  template <int idx>
  static __XETLA_API typename std::enable_if_t<idx == 0, int>
  get_tile_idx(sycl::nd_item<3> &item) {
    return item.get_group(idx);
  }
  // get transformed dim1 group id
  template <int idx>
  static __XETLA_API typename std::enable_if_t<idx == 2, int>
  get_tile_idx(sycl::nd_item<3> &item) {
    uint32_t wg_inner_id = get_2d_group_linear_id(item);
    uint32_t wg_coord_n = wg_inner_id / wg_num_m;
    int start_n_id = wg_coord_n;
    return wg_coord_n;
  }
  // get transformed dim2 group id
  template <int idx>
  static __XETLA_API typename std::enable_if_t<idx == 1, int>
  get_tile_idx(sycl::nd_item<3> &item) {
    uint32_t wg_inner_id = get_2d_group_linear_id(item);
    uint32_t wg_coord_m = wg_inner_id % wg_num_m;
    int start_m_id = wg_coord_m;
    return wg_coord_m;
  }
  // correct group range, workgroup will be padded to fit the given wg_num_n
  // under this swizzle policy
  static __XETLA_API void update_group_range(uint32_t &group_range_m,
                                             uint32_t &group_range_n) {
    group_range_m = (group_range_m + wg_num_m - 1) / wg_num_m * wg_num_m;
    group_range_n = (group_range_n + wg_num_n - 1) / wg_num_n * wg_num_n;
  }

private:
  static constexpr uint32_t max_wg_num = arch_attr_t<arch_tag>::max_wg_num;
  static constexpr uint32_t wg_num_n = max_wg_num / wg_num_m_;
  // static_assert(!(max_wg_num % wg_num_n),
  //         "max_wg_num cannot be divisible by given wg_num_n!");
  static constexpr uint32_t wg_num_m = wg_num_m_;
};
} // namespace gpu::xetla::kernel

template <typename dtype_a, typename dtype_b, typename dtype_c,
          typename dtype_acc, typename swizzle, uint32_t wg_m, uint32_t wg_n,
          uint32_t sg_m, uint32_t sg_n, uint32_t sg_k, mem_layout layout_a,
          mem_layout layout_b, uint32_t global_kslicing,
          uint32_t local_kslicing>
struct bf16_gemm_test_func {
  using tile_shape = tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
  static constexpr uint32_t periodic_sync_interval = 8;
  static constexpr uint32_t prefetch_distance = 3;
  using gemm_t = typename gemm_selector_t<
      dtype_a, dtype_b, layout_a, layout_b, mem_space::global,
      mem_space::global, 8, 8, dtype_acc, tile_shape, sg_k, mma_engine::xmx,
      gpu_arch::Xe, prefetch_distance, periodic_sync_interval>::gemm;

  using epilogue_t =
      epilogue_t<epilogue_policy_default<gpu_arch::Xe>, tile_shape,
                 mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>>;

  using group_swizzle = swizzle;

  using dispatch_policy =
      dispatch_policy_kslicing<group_swizzle, global_kslicing, local_kslicing>;

  using gemm_op_t = gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;

  static const char *func_name() { return "bf16_gemm_test_func"; }

  static inline void run(sycl::nd_item<3> &item, dtype_a *A, dtype_b *B,
                         dtype_c *C, uint32_t mat_m, uint32_t mat_n,
                         uint32_t mat_k, dtype_acc *Acc, uint32_t *Cnt) {
    typename gemm_op_t::arguments_t arg(
        mat_m, mat_k, mat_n, A,
        layout_a == mem_layout::col_major ? mat_m : mat_k, B,
        layout_b == mem_layout::col_major ? mat_k : mat_n, C, mat_n, Acc, Cnt);
    gemm_op_t gemm_op;
    gemm_op(item, arg);
  }
};
