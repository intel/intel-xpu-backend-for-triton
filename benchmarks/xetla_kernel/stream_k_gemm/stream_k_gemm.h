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
#ifndef TRITONBENCHMARK_STREAM_K_GEMM_H
#define TRITONBENCHMARK_STREAM_K_GEMM_H

#include "xetla.hpp"
#include <sycl.hpp>

sycl::event stream_k_gemm_run(void *_A, void *_B, void *_C, void *_Acc,
                              void *_Cnt, sycl::queue &queue) {

  // GEMM input size
  uint32_t matrix_m = 3072;
  uint32_t matrix_n = 3072;
  uint32_t matrix_k = 4096;

  uint32_t size_a = matrix_m * matrix_k;
  uint32_t size_b = matrix_k * matrix_n;
  uint32_t size_c = matrix_m * matrix_n;

  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;

  data_type_a *A = static_cast<data_type_a *>(_A);
  data_type_b *B = static_cast<data_type_b *>(_B);
  data_type_c *C = static_cast<data_type_c *>(_C);

  constexpr uint32_t wg_tile_m = 256;
  constexpr uint32_t wg_tile_n = 256;
  constexpr uint32_t sg_tile_m = 32;
  constexpr uint32_t sg_tile_n = 64;

  // There are implicit requirement for sg_tile_k range
  constexpr uint32_t sg_tile_k = 32;

  // StreamK parameters - xecores available for stream_k dispatch
  uint32_t avail_xecores = 64;

  // Org the compute shape for sub-matrix
  using tile_shape =
      gpu::xetla::group::tile_shape_t<wg_tile_n,  // workgroup size in dim0
                                      wg_tile_m,  //	workgroup size in dim1
                                      sg_tile_n,  //	subgroup size in dim0
                                      sg_tile_m>; //	subgroup size in dim1

  // Mirco-kernel configuration
  using gemm_config = gpu::xetla::group::gemm_selector_t<
      data_type_a,                       // input datatype for A
      data_type_b,                       // input datatype for B
      gpu::xetla::mem_layout::row_major, // memory layout for A
      gpu::xetla::mem_layout::row_major, // memory layout for B
      gpu::xetla::mem_space::global,     // memory reading from global mem for A
      gpu::xetla::mem_space::global,     // memory reading from global mem for B
      8,               // leading dimension for A, in unit of element
      8,               // leading dimension for B, in unit of element
      data_type_acc,   // accumulator data type for intermediate resutls
      tile_shape,      // computation tile shape
      sg_tile_k,       // elements in each iteration
      mma_engine::xmx, // compute engine
      gpu_arch::Xe, 3,
      4> // GPU arch, prefetch stages, periodic sync frequency
      ::gemm;

  using dispatch_stream_k =
      gpu::xetla::kernel::dispatch_policy_stream_k<gpu_arch::Xe>;

  using epilogue_t = gpu::xetla::group::epilogue_t<
      gpu::xetla::group::epilogue_policy_default<gpu_arch::Xe>, tile_shape,
      mem_desc_t<data_type_c, gpu::xetla::mem_layout::row_major,
                 gpu::xetla::mem_space::global>>;

  using gemm_op_t =
      gpu::xetla::kernel::gemm_universal_t<dispatch_stream_k, gemm_config,
                                           epilogue_t>;

  // setup stream_k workgroup split
  dispatch_stream_k stream_k(matrix_m, matrix_k, matrix_n, wg_tile_m,
                             gemm_config::k_stride, wg_tile_n, sg_tile_m,
                             sg_tile_n, avail_xecores);

  // allocate temp buffers for global split
  size_t size_acc = gemm_op_t::get_acc_buf_size(stream_k);
  size_t size_cnt = gemm_op_t::get_cnt_buf_size(stream_k);

  data_type_acc *Acc = static_cast<data_type_acc *>(_Acc);
  uint32_t *Cnt = static_cast<uint32_t *>(_Cnt);

  // set up gemm arguments
  typename gemm_op_t::arguments_t gemm_arg(
      matrix_m, matrix_k, matrix_n, A, matrix_k, B, matrix_n, C, matrix_n, Acc,
      matrix_n, Cnt, size_cnt, stream_k);

  cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(gemm_arg);

  auto gpu_event = queue.submit([&](sycl::handler &cgh) {
    // GPU kernel
    cgh.parallel_for(NDRange, [=](sycl::nd_item<3> item) KERNEL_MAIN {
      // allocate slm and nbarrier resource
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(item, gemm_arg);
    });
  });
  return gpu_event;
}

#endif // TRITONBENCHMARK_STREAM_K_GEMM_H
