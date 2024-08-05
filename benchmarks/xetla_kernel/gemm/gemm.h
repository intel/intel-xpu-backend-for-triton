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
#ifndef TRITONBENCHMARK_GEMM_H
#define TRITONBENCHMARK_GEMM_H

#include "gemm_config.hpp"
#include "kernel_func.hpp"

#define BARNUM (32)
#define SLMSIZE (128 * 1024)

template <class Test>
sycl::event gemm_run(void *_A, void *_B, void *_C, void *_Acc, void *_Cnt,
                     sycl::queue &queue) {
  // Accept incoming parameters
  size_t matrix_m = Test::mat_m;
  size_t matrix_n = Test::mat_n;
  size_t matrix_k = Test::mat_k;
  size_t batch = Test::batch;
  constexpr size_t wg_tile_m = Test::wg_m;
  constexpr size_t wg_tile_n = Test::wg_n;
  constexpr size_t sg_tile_m = Test::sg_m;
  constexpr size_t sg_tile_n = Test::sg_n;
  constexpr size_t sg_tile_k = Test::sg_k;
  using data_type_a = typename Test::data_type_a;
  using data_type_b = typename Test::data_type_b;
  using data_type_c = typename Test::data_type_c;
  using data_type_acc = float;
  using gemm_functor =
      bf16_gemm_test_func<data_type_a, data_type_b, data_type_c, data_type_acc,
                          wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k,
                          Test::layout_a, Test::layout_b, Test::global_kslicing,
                          Test::local_kslicing>;
  using gemm_op_t = typename gemm_functor::gemm_op_t;

  size_t size_a = matrix_m * matrix_k;
  size_t size_b = matrix_k * matrix_n;
  size_t size_c = matrix_m * matrix_n;
  size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
  size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

  auto context = queue.get_info<sycl::info::queue::context>();
  auto device = queue.get_info<sycl::info::queue::device>();
  data_type_a *A = static_cast<data_type_a *>(_A);
  data_type_b *B = static_cast<data_type_b *>(_B);
  data_type_c *C = static_cast<data_type_c *>(_C);
  data_type_acc *Acc = static_cast<data_type_acc *>(_Acc);
  uint32_t *Cnt = static_cast<uint32_t *>(_Cnt);

  cl::sycl::range<3> group_range = {batch,
                                    (matrix_m + wg_tile_m - 1) / wg_tile_m,
                                    (matrix_n + wg_tile_n - 1) / wg_tile_n};
  cl::sycl::range<3> local_range = {1, (wg_tile_m + sg_tile_m - 1) / sg_tile_m,
                                    (wg_tile_n + sg_tile_n - 1) / sg_tile_n};
  cl::sycl::nd_range<3> nd_range = {group_range * local_range, local_range};

  std::vector<sycl::kernel_id> kernelId = {sycl::get_kernel_id<Test>()};

  static std::once_flag jit_once;
  std::call_once(jit_once, [&]() {
    auto inputBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(context, kernelId);
    setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
           " -vc-codegen -doubleGRF -vc-disable-indvars-opt "
           " -Xfinalizer '-printregusage -enableBCR -DPASTokenReduction '",
           1);
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
        int batch_idx = item.get_group(0);
        auto A_ptr = A + batch_idx * size_a;
        auto B_ptr = B + batch_idx * size_b;
        auto C_ptr = C + batch_idx * size_c;
        auto Acc_ptr = Acc + batch_idx * size_acc;
        auto Cnt_ptr = Cnt + batch_idx * size_cnt;
        gpu::xetla::xetla_local_init<SLMSIZE>();
        gpu::xetla::xetla_nbarrier_init<BARNUM>();
        gemm_functor::run(item, A_ptr, B_ptr, C_ptr, matrix_m, matrix_n,
                          matrix_k, Acc_ptr, Cnt_ptr);
      });
    });
    return e_esimd;
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    abort();
  }
}

#endif // TRITONBENCHMARK_GEMM_H
