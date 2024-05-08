/*******************************************************************************
 * Copyright (c) 2023-2024 Intel Corporation
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

template <typename dtype_a, typename dtype_b, typename dtype_c,
          typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
          uint32_t sg_n, uint32_t sg_k, gpu::xetla::mem_layout layout_a,
          gpu::xetla::mem_layout layout_b, uint32_t global_kslicing,
          uint32_t local_kslicing>
struct bgemm_test_func {
  using tile_shape = gpu::xetla::group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
  static constexpr uint32_t periodic_sync_interval = 8;
  static constexpr uint32_t prefetch_distance = 3;
  using gemm_t = typename gpu::xetla::group::gemm_selector_t<
      dtype_a, dtype_b, layout_a, layout_b, gpu::xetla::mem_space::global,
      gpu::xetla::mem_space::global, 8, 8, dtype_acc, tile_shape, sg_k,
      gpu::xetla::mma_engine::xmx, gpu::xetla::gpu_arch::Xe, prefetch_distance,
      periodic_sync_interval>::gemm;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_default<gpu::xetla::gpu_arch::Xe>,
      tile_shape,
      gpu::xetla::mem_desc_t<dtype_c, gpu::xetla::mem_layout::row_major,
                             gpu::xetla::mem_space::global>>;

  using group_swizzle_t =
      gpu::xetla::kernel::group_swizzle_default<gpu::xetla::gpu_arch::Xe>;

  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_kslicing<
          group_swizzle_t, global_kslicing, local_kslicing>,
      gemm_t, epilogue_t>;

  static constexpr uint32_t barrier_count = gemm_op_t::get_barrier_count();
  static constexpr uint32_t slm_size = gemm_op_t::get_slm_size();
  static const char *func_name() { return "bgemm_test_func"; }

  static inline void run(sycl::nd_item<3> &item, dtype_a *A, dtype_b *B,
                         dtype_c *C, uint32_t mat_m, uint32_t mat_n,
                         uint32_t mat_k, uint32_t lda, uint32_t ldb,
                         uint32_t ldc, dtype_acc *acc_ptr, uint32_t *cnt_ptr) {
    typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, A, lda, B, ldb, C,
                                        ldc, acc_ptr, cnt_ptr);
    gemm_op_t gemm_op;
    gemm_op(item, arg);
  }
};
