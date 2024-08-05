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
#ifndef TRITON_MICROBENCH_GEMM_CONFIG_H
#define TRITON_MICROBENCH_GEMM_CONFIG_H

#include "common/common.hpp"
#include "common/core/common.hpp"
#include <sycl.hpp>

class Test_1x1024x1024x1024_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 1024;
  static constexpr size_t mat_k = 1024;
  static constexpr size_t mat_n = 1024;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 64;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x2048x2048x2048_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 2048;
  static constexpr size_t mat_k = 2048;
  static constexpr size_t mat_n = 2048;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x4096x4096x4096_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x8192x8192x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 8192;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x1x5120x13824_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 1;
  static constexpr size_t mat_k = 5120;
  static constexpr size_t mat_n = 13824;
  static constexpr size_t wg_m = 8;
  static constexpr size_t wg_n = 512;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x4x4096x12288_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 4;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t mat_n = 12288;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x512x8192x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 512;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x512x8192x32768_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 512;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t mat_n = 32768;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x512x32768x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 512;
  static constexpr size_t mat_k = 32768;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x1024x16384x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 1024;
  static constexpr size_t mat_k = 16384;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x1024x28672x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 1024;
  static constexpr size_t mat_k = 28672;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x3072x4096x3072_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 3072;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t mat_n = 3072;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x4096x16384x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_k = 16384;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x8192x16384x1024_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 8192;
  static constexpr size_t mat_k = 16384;
  static constexpr size_t mat_n = 1024;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x8192x16384x4096_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 8192;
  static constexpr size_t mat_k = 16384;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x16384x1024x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 16384;
  static constexpr size_t mat_k = 1024;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x16384x4096x8192_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 16384;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t mat_n = 8192;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x16384x8192x1024_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 16384;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t mat_n = 1024;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_1x16384x8192x4096_row_row {
public:
  static constexpr size_t batch = 1;
  static constexpr size_t mat_m = 16384;
  static constexpr size_t mat_k = 8192;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_4x32768x128x4096_row_row {
public:
  static constexpr size_t batch = 4;
  static constexpr size_t mat_m = 32768;
  static constexpr size_t mat_k = 128;
  static constexpr size_t mat_n = 4096;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_4x32768x4096x128_row_row {
public:
  static constexpr size_t batch = 4;
  static constexpr size_t mat_m = 32768;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t mat_n = 128;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_32x4096x4096x128_row_row {
public:
  static constexpr size_t batch = 32;
  static constexpr size_t mat_m = 4096;
  static constexpr size_t mat_k = 4096;
  static constexpr size_t mat_n = 128;
  static constexpr size_t wg_m = 64;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_4096x8x128x16384_row_row {
public:
  static constexpr size_t batch = 4096;
  static constexpr size_t mat_m = 8;
  static constexpr size_t mat_k = 128;
  static constexpr size_t mat_n = 16384;
  static constexpr size_t wg_m = 8;
  static constexpr size_t wg_n = 512;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

class Test_4096x8x16384x128_row_row {
public:
  static constexpr size_t batch = 4096;
  static constexpr size_t mat_m = 8;
  static constexpr size_t mat_k = 16384;
  static constexpr size_t mat_n = 128;
  static constexpr size_t wg_m = 8;
  static constexpr size_t wg_n = 512;
  static constexpr size_t sg_m = 8;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a =
      gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b =
      gpu::xetla::mem_layout::row_major;
  using data_type_a = sycl::ext::oneapi::bfloat16;
  using data_type_b = sycl::ext::oneapi::bfloat16;
  using data_type_c = float;
  using data_type_acc = float;
};

#endif // TRITON_MICROBENCH_GEMM_CONFIG_H
