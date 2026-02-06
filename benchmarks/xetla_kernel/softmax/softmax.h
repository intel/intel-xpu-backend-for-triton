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
#ifndef TRITONBENCHMARK_SOFTMAX_H
#define TRITONBENCHMARK_SOFTMAX_H

#include "kernel_func.hpp"
#include "softmax_config.hpp"

template <typename Config>
sycl::event softmax_forward(void *input, void *output, sycl::queue &queue) {
  // Accept incoming parameters
  size_t mat_n = Config::mat_n;
  size_t mat_m = Config::mat_m;
  constexpr size_t sg_n = Config::sg_n;
  constexpr size_t sg_m = Config::sg_m;
  constexpr size_t wg_n = Config::wg_n;
  constexpr size_t wg_m = Config::wg_m;

  using data_type_in = typename Config::data_type_in;
  using data_type_acc = typename Config::data_type_acc;
  using data_type_out = typename Config::data_type_out;

  data_type_in *buffer_in = static_cast<data_type_in *>(input);
  data_type_out *buffer_out = static_cast<data_type_out *>(output);
  data_type_acc sqrt_dk_inv = 0.125f;

  size_t group_range_m = (mat_m + wg_m - 1) / wg_m;
  size_t group_range_n = (mat_n + wg_n - 1) / wg_n;
  size_t subgroup_range_m = (wg_m + sg_m - 1) / sg_m;
  size_t subgroup_range_n = (wg_n + sg_n - 1) / sg_n;

  cl::sycl::range<3> group_range{1, group_range_m, group_range_n};
  cl::sycl::range<3> local_range{1, subgroup_range_m, subgroup_range_n};
  cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

  auto context = queue.get_info<sycl::info::queue::context>();
  std::vector<sycl::kernel_id> kernelId = {sycl::get_kernel_id<Config>()};

  static std::once_flag jit_once;
  std::call_once(jit_once, [&]() {
    auto inputBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(context, kernelId);
    setenv("SYCL_PROGRAM_COMPILE_OPTIONS",
           " -vc-codegen -doubleGRF  -Xfinalizer ' "
           "-printregusage -enableBCR  "
           "-DPASTokenReduction '",
           1);
    sycl::kernel_bundle<sycl::bundle_state::executable> exeBundle =
        build(inputBundle);
    unsetenv("SYCL_PROGRAM_COMPILE_OPTIONS");
  });

  auto exeBundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      context, kernelId);
  try {
    auto e_softmax_fwd = queue.submit([&](sycl::handler &cgh) {
      cgh.use_kernel_bundle(exeBundle);
      cgh.parallel_for<Config>(
          nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
            using softmax_fwd_func =
                softmax_fwd_test_func<data_type_in, data_type_out,
                                      data_type_acc, wg_n, wg_m, sg_n, sg_m>;
            constexpr uint32_t barrier_count = softmax_fwd_func::barrier_count;
            constexpr uint32_t slm_size = softmax_fwd_func::slm_size;
            if constexpr (barrier_count != 0) {
              xetla_nbarrier_init<barrier_count>();
            }
            if constexpr (slm_size != 0) {
              xetla_local_init<slm_size>();
            }
            softmax_fwd_func::run(item, buffer_in, buffer_out, mat_m, mat_n,
                                  mat_n, sqrt_dk_inv);
          });
    });
    // e_softmax_fwd.wait();
    // double time = (e_softmax_fwd.template get_profiling_info<
    //                    sycl::info::event_profiling::command_end>() -
    //                e_softmax_fwd.template get_profiling_info<
    //                    sycl::info::event_profiling::command_start>()) /
    //               (1000.0f * 1000.0f * 1000.f);

    // printf("M: %d, N: %d Data_type_in: %d, Bandwidth: GB/S: %f \n", mat_m,
    // mat_n,
    //        sizeof(data_type_in),
    //        ((mat_m * mat_n * sizeof(data_type_in) * 2 / 1e9) / time));
    return e_softmax_fwd;
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    throw;
  }
}

#endif // TRITONBENCHMARK_SOFTMAX_H
