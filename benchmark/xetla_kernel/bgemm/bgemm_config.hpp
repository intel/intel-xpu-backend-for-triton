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
#ifndef TRITON_MICROBENCH_BGEMM_CONFIG_H
#define TRITON_MICROBENCH_BGEMM_CONFIG_H

#include <sycl.hpp>
#include "common/core/common.hpp"
#include "common/common.hpp"

class Test_256x256x256_row_row {
public:
  static constexpr size_t mat_m = 256;
  static constexpr size_t mat_k = 256;
  static constexpr size_t mat_n = 256;
  static constexpr size_t wg_m = 16;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 8;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_384x384x384_row_row {
public:
  static constexpr size_t mat_m = 384;
  static constexpr size_t mat_k = 384;
  static constexpr size_t mat_n = 384;
  static constexpr size_t wg_m = 32;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_512x512x512_row_row {
public:
  static constexpr size_t mat_m = 512;
  static constexpr size_t mat_k = 512;
  static constexpr size_t mat_n = 512;
  static constexpr size_t wg_m = 32;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 4;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_640x640x640_row_row {
public:
  static constexpr size_t mat_m = 640;
  static constexpr size_t mat_k = 640;
  static constexpr size_t mat_n = 640;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 64;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 16;
  static constexpr size_t sg_k = 64;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_768x768x768_row_row {
public:
  static constexpr size_t mat_m = 768;
  static constexpr size_t mat_k = 768;
  static constexpr size_t mat_n = 768;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 16;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 64;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_896x896x896_row_row {
public:
  static constexpr size_t mat_m = 896;
  static constexpr size_t mat_k = 896;
  static constexpr size_t mat_n = 896;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 128;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 2;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1024x1024x1024_row_row {
public:
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
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1152x1152x1152_row_row {
public:
  static constexpr size_t mat_m = 1152;
  static constexpr size_t mat_k = 1152;
  static constexpr size_t mat_n = 1152;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1280x1280x1280_row_row {
public:
  static constexpr size_t mat_m = 1280;
  static constexpr size_t mat_k = 1280;
  static constexpr size_t mat_n = 1280;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1408x1408x1408_row_row {
public:
  static constexpr size_t mat_m = 1408;
  static constexpr size_t mat_k = 1408;
  static constexpr size_t mat_n = 1408;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1536x1536x1536_row_row {
public:
  static constexpr size_t mat_m = 1536;
  static constexpr size_t mat_k = 1536;
  static constexpr size_t mat_n = 1536;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1664x1664x1664_row_row {
public:
  static constexpr size_t mat_m = 1664;
  static constexpr size_t mat_k = 1664;
  static constexpr size_t mat_n = 1664;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1792x1792x1792_row_row {
public:
  static constexpr size_t mat_m = 1792;
  static constexpr size_t mat_k = 1792;
  static constexpr size_t mat_n = 1792;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_1920x1920x1920_row_row {
public:
  static constexpr size_t mat_m = 1920;
  static constexpr size_t mat_k = 1920;
  static constexpr size_t mat_n = 1920;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 64;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2048x2048x2048_row_row {
public:
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
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2176x2176x2176_row_row {
public:
  static constexpr size_t mat_m = 2176;
  static constexpr size_t mat_k = 2176;
  static constexpr size_t mat_n = 2176;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2304x2304x2304_row_row {
public:
  static constexpr size_t mat_m = 2304;
  static constexpr size_t mat_k = 2304;
  static constexpr size_t mat_n = 2304;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2432x2432x2432_row_row {
public:
  static constexpr size_t mat_m = 2432;
  static constexpr size_t mat_k = 2432;
  static constexpr size_t mat_n = 2432;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2560x2560x2560_row_row {
public:
  static constexpr size_t mat_m = 2560;
  static constexpr size_t mat_k = 2560;
  static constexpr size_t mat_n = 2560;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2688x2688x2688_row_row {
public:
  static constexpr size_t mat_m = 2688;
  static constexpr size_t mat_k = 2688;
  static constexpr size_t mat_n = 2688;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2816x2816x2816_row_row {
public:
  static constexpr size_t mat_m = 2816;
  static constexpr size_t mat_k = 2816;
  static constexpr size_t mat_n = 2816;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_2944x2944x2944_row_row {
public:
  static constexpr size_t mat_m = 2944;
  static constexpr size_t mat_k = 2944;
  static constexpr size_t mat_n = 2944;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3072x3072x3072_row_row {
public:
  static constexpr size_t mat_m = 3072;
  static constexpr size_t mat_k = 3072;
  static constexpr size_t mat_n = 3072;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3200x3200x3200_row_row {
public:
  static constexpr size_t mat_m = 3200;
  static constexpr size_t mat_k = 3200;
  static constexpr size_t mat_n = 3200;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3328x3328x3328_row_row {
public:
  static constexpr size_t mat_m = 3328;
  static constexpr size_t mat_k = 3328;
  static constexpr size_t mat_n = 3328;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3456x3456x3456_row_row {
public:
  static constexpr size_t mat_m = 3456;
  static constexpr size_t mat_k = 3456;
  static constexpr size_t mat_n = 3456;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3584x3584x3584_row_row {
public:
  static constexpr size_t mat_m = 3584;
  static constexpr size_t mat_k = 3584;
  static constexpr size_t mat_n = 3584;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3712x3712x3712_row_row {
public:
  static constexpr size_t mat_m = 3712;
  static constexpr size_t mat_k = 3712;
  static constexpr size_t mat_n = 3712;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3840x3840x3840_row_row {
public:
  static constexpr size_t mat_m = 3840;
  static constexpr size_t mat_k = 3840;
  static constexpr size_t mat_n = 3840;
  static constexpr size_t wg_m = 128;
  static constexpr size_t wg_n = 512;
  static constexpr size_t sg_m = 64;
  static constexpr size_t sg_n = 32;
  static constexpr size_t sg_k = 16;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_3968x3968x3968_row_row {
public:
  static constexpr size_t mat_m = 3968;
  static constexpr size_t mat_k = 3968;
  static constexpr size_t mat_n = 3968;
  static constexpr size_t wg_m = 256;
  static constexpr size_t wg_n = 256;
  static constexpr size_t sg_m = 32;
  static constexpr size_t sg_n = 64;
  static constexpr size_t sg_k = 32;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

class Test_4096x4096x4096_row_row {
public:
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
  static constexpr gpu::xetla::mem_layout layout_a = gpu::xetla::mem_layout::row_major;
  static constexpr gpu::xetla::mem_layout layout_b = gpu::xetla::mem_layout::row_major;
  using data_type_a = cl::sycl::half;
  using data_type_b = cl::sycl::half;
  using data_type_c = cl::sycl::half;
};

#endif // TRITON_MICROBENCH_BGEMM_CONFIG_H