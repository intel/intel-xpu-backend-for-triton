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
#ifndef TRITONBENCHMARK_SPLIT_K_GEMM_H
#define TRITONBENCHMARK_SPLIT_K_GEMM_H

#include "xetla.hpp"
#include <sycl.hpp>

enum class kslicing_impl_t : uint8_t { none = 0, global = 1, local = 2 };

template <int m, int k, int n,
          kslicing_impl_t kslicing_type = kslicing_impl_t::none>
sycl::event split_k_gemm_run(void *_A, void *_B, void *_C, void *_Acc,
                             void *_Cnt, sycl::queue &queue) {

  // GEMM_UNIVERSAL input size
  size_t matrix_m = m;
  size_t matrix_n = n;
  size_t matrix_k = k;

  size_t size_a = matrix_m * matrix_k;
  size_t size_b = matrix_k * matrix_n;
  size_t size_c = matrix_m * matrix_n;

  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;

  data_type_a *A = static_cast<data_type_a *>(_A);
  data_type_b *B = static_cast<data_type_b *>(_B);
  data_type_c *C = static_cast<data_type_c *>(_C);

  // Define the shape of workgroup
  // It's tunable parameters based on different input shape and hardware for
  // better performance
  constexpr uint32_t wg_tile_m =
      (kslicing_type != kslicing_impl_t::local) ? 256 : 64;
  constexpr uint32_t wg_tile_n =
      (kslicing_type != kslicing_impl_t::local) ? 256 : 128;

  // specify the range k_w/k_s by setting the corresponding ratio
  // splitk using global memory
  constexpr uint32_t num_global_splitk =
      (kslicing_type == kslicing_impl_t::global) ? 2 : 1;
  // splitk using local memory
  constexpr uint32_t num_local_splitk =
      (kslicing_type == kslicing_impl_t::local) ? 2 : 1;

  // Mirco-kernel configuration
  using tune_option =
      dict_t<elem_v_t<tune_key::param_optimizer_type,
                      tune_key_value::param_optimizer_decision_tree>,
             elem_t_t<tune_key::data_type_acc, data_type_acc>,
             elem_v_t<tune_key::dispatch_policy,
                      tune_key_value::dispatch_policy_kslicing>,
             elem_v_t<tune_key::global_kslicing_ratio, num_global_splitk>,
             elem_v_t<tune_key::local_kslicing_ratio, num_local_splitk>,
             elem_t_t<tune_key::wg_tile_shape, shape<wg_tile_n, wg_tile_m>>>;
  using gemm_op_t = gpu::xetla::kernel::default_gemm_t<
      data_type_a,           // input datatype for A
      mem_layout::row_major, // memory layout for A
      8,           // leading dimension alignment for A, in unit of element
      data_type_b, // input datatype for B
      mem_layout::row_major, // memory layout for B
      8,           // leading dimension alignment for B, in unit of element
      data_type_c, // output datatype for C
      mem_layout::row_major, // memory layout for C
      8,             // leading dimension alignment for C, in unit of element
      data_type_acc, // accumulator data type for intermediate resutls
      gpu_arch::Xe,  // GPU arch
      tune_option>;

  // allocate temp buffers for global split
  size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
  size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

  data_type_acc *Acc = static_cast<data_type_acc *>(_Acc);
  uint32_t *Cnt = static_cast<uint32_t *>(_Cnt);

  // set up gemm_universal arguments
  typename gemm_op_t::arguments_t gemm_arg(matrix_m, matrix_k, matrix_n, A,
                                           matrix_k, B, matrix_n, C, matrix_n,
                                           Acc, Cnt);

  cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);

  auto gpu_event = queue.submit([&](sycl::handler &cgh) {
    // GPU kernel
    cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
      // allocate slm and nbarrier resource
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    });
  });
  return gpu_event;
}

#endif // TRITONBENCHMARK_SPLIT_K_GEMM_H
