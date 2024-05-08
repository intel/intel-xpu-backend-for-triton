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
#ifndef TRITONBENCHMARK_BGEMM_H
#define TRITONBENCHMARK_BGEMM_H

#include "bgemm_config.hpp"
#include "kernel_func.hpp"

template <class Test>
void bgemm_run(void *_A, void *_B, void *_C, void *_Acc, void *_Cnt,
               sycl::queue &queue) {
  // Accept incoming parameters
  size_t matrix_m = Test::mat_m;
  size_t matrix_n = Test::mat_n;
  size_t matrix_k = Test::mat_k;
  constexpr size_t wg_tile_m = Test::wg_m;
  constexpr size_t wg_tile_n = Test::wg_n;
  constexpr size_t sg_tile_m = Test::sg_m;
  constexpr size_t sg_tile_n = Test::sg_n;
  constexpr size_t sg_tile_k = Test::sg_k;
  using data_type_a = typename Test::data_type_a;
  using data_type_b = typename Test::data_type_b;
  using data_type_c = typename Test::data_type_c;
  using data_type_acc = float;
  using bgemm_functor =
      bgemm_test_func<data_type_a, data_type_b, data_type_c, data_type_acc,
                      wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k,
                      Test::layout_a, Test::layout_b, Test::global_kslicing,
                      Test::local_kslicing>;
  using gemm_op_t = typename bgemm_functor::gemm_op_t;

  size_t lda =
      Test::layout_a == gpu::xetla::mem_layout::col_major ? matrix_m : matrix_k;
  size_t ldb =
      Test::layout_b == gpu::xetla::mem_layout::col_major ? matrix_k : matrix_n;
  size_t ldc = matrix_n;

  std::string mem_layout_a_str =
      Test::layout_a == gpu::xetla::mem_layout::col_major
          ? "gpu::xetla::mem_layout::col_major"
          : "gpu::xetla::mem_layout::row_major";
  std::string mem_layout_b_str =
      Test::layout_b == gpu::xetla::mem_layout::col_major
          ? "gpu::xetla::mem_layout::col_major"
          : "gpu::xetla::mem_layout::row_major";

  constexpr bool is_col_major_a =
      Test::layout_a == gpu::xetla::mem_layout::col_major;
  constexpr bool is_col_major_b =
      Test::layout_b == gpu::xetla::mem_layout::col_major;

  size_t size_a = matrix_m * matrix_k;
  size_t size_b = matrix_k * matrix_n;
  size_t size_c = matrix_m * matrix_n;
  size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
  size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

  auto context = queue.get_info<sycl::info::queue::context>();
  data_type_a *A = static_cast<data_type_a *>(_A);
  data_type_b *B = static_cast<data_type_b *>(_B);
  data_type_c *C = static_cast<data_type_c *>(_C);
  data_type_acc *Acc = static_cast<data_type_acc *>(_Acc);
  uint32_t *Cnt = static_cast<uint32_t *>(_Cnt);
  //  uint32_t *Cnt = static_cast<uint32_t *>(
  //      malloc_shared(size_cnt * sizeof(uint32_t), queue.get_device(),
  //      context));

  // here keep the same dim in CM and esimd, diff the index in kernel code
  size_t group_range_m = (matrix_m % wg_tile_m == 0)
                             ? matrix_m / wg_tile_m
                             : (matrix_m / wg_tile_m) + 1;
  size_t group_range_n = (matrix_n % wg_tile_n == 0)
                             ? matrix_n / wg_tile_n
                             : (matrix_n / wg_tile_n) + 1;
  size_t subgroup_range_m = (wg_tile_m % sg_tile_m == 0)
                                ? wg_tile_m / sg_tile_m
                                : (wg_tile_m / sg_tile_m) + 1;
  size_t subgroup_range_n = (wg_tile_n % sg_tile_n == 0)
                                ? wg_tile_n / sg_tile_n
                                : (wg_tile_n / sg_tile_n) + 1;
  cl::sycl::range<3> group_range{Test::global_kslicing, group_range_m,
                                 group_range_n};
  cl::sycl::range<3> local_range{Test::local_kslicing, subgroup_range_m,
                                 subgroup_range_n};
  cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

  std::vector<sycl::kernel_id> kernelId = {sycl::get_kernel_id<Test>()};

  static std::once_flag jit_once;
  std::call_once(jit_once, [&]() {
    auto inputBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(context, kernelId);
    // setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
    //        " -vc-codegen -doubleGRF -vc-disable-indvars-opt "
    //        " -Xfinalizer '-printregusage -enableBCR -DPASTokenReduction '",
    //        1);
    sycl::kernel_bundle<sycl::bundle_state::executable> exeBundle =
        build(inputBundle);
    unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");
  });

  auto exeBundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      context, kernelId);
  try {
    auto e_esimd = queue.submit([&](sycl::handler &cgh) {
      cgh.use_kernel_bundle(exeBundle);
      cgh.parallel_for<Test>(nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
        constexpr uint32_t barrier_count = bgemm_functor::barrier_count;
        constexpr uint32_t slm_size = bgemm_functor::slm_size;
        if constexpr (barrier_count != 0) {
          gpu::xetla::xetla_nbarrier_init<barrier_count>();
        }
        if constexpr (slm_size != 0) {
          gpu::xetla::xetla_local_init<slm_size>();
        }
        bgemm_functor::run(item, A, B, C, matrix_m, matrix_n, matrix_k, lda,
                           ldb, ldc, Acc, Cnt);
      });
    });
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    abort();
  }
}

#endif // TRITONBENCHMARK_BGEMM_H
