/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace cute;

namespace {

// Command line options parsing
struct FMHAOptions {

  bool error;

  int batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
      seq_len_kv_cache, head_size_qk, head_size_vo, page_size;
  float softmax_scale;
  std::string bm_name;

  FMHAOptions()
      : error(false), batch(32), num_heads_q(16), num_heads_kv(16),
        seq_len_qo(1), head_size_qk(128), seq_len_kv(512), seq_len_kv_cache(0),
        page_size(128), head_size_vo(128), softmax_scale(1.f),
        bm_name("Flash Attention v2") {}

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/";
    std::string const test_name_suffix =
        std::to_string(batch) + "x" + std::to_string(num_heads_q) + "x" +
        std::to_string(num_heads_kv) + "x" + std::to_string(seq_len_qo) + "x" +
        std::to_string(head_size_qk) + "x" + std::to_string(seq_len_kv) + "x" +
        std::to_string(seq_len_kv_cache) + "x" + std::to_string(head_size_vo);
    full_name << test_name_suffix;

    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
template <typename SrcT, typename DstT> class ConvertTensorKernelTag {};

template <typename SrcT, typename DstT>
void convert_tensor(const SrcT *d_src, DstT *d_dst, size_t size) {
  using Tag = ConvertTensorKernelTag<SrcT, DstT>;
  compat::get_default_queue()
      .parallel_for<Tag>(
          size,
          [=](auto indx) { d_dst[indx] = static_cast<DstT>(d_src[indx]); })
      .wait();
}

template <typename InT>
inline auto in_memory(cutlass::DeviceAllocation<InT> &in) {
  using OutT = cute::conditional_t<(sizeof_bits_v<InT> <= 8), half_t, InT>;
  if constexpr (!is_same_v<InT, OutT>) {
    cutlass::DeviceAllocation<OutT> out(in.size());
    convert_tensor<InT, OutT>(in.get(), out.get(), in.size());
    return out;
  } else {
    return in;
  };
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class FMHAConfiguration> struct FMHARunner {
  using FMHAKernel = typename FMHAConfiguration::FMHAKernel;

  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using LayoutQ = typename FMHAConfiguration::LayoutQ;
  using LayoutK = typename FMHAConfiguration::LayoutK;
  using LayoutV = typename FMHAConfiguration::LayoutV;
  using LayoutO = typename FMHAConfiguration::LayoutO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = typename FMHAConfiguration::ProblemShapeType;
  static constexpr bool Causal = FMHAConfiguration::Causal;
  static constexpr bool isVarLen = FMHAConfiguration::VarLen;
  static constexpr bool CachedKV = FMHAConfiguration::CachedKV;
  static constexpr bool PagedKV = FMHAConfiguration::PagedKV;
  static constexpr bool Persistent = FMHAConfiguration::Persistent;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  StrideK stride_K_cache;
  StrideV stride_V_cache;

  uint64_t seed = 0;

  ProblemShapeType initialize(const FMHAOptions &options) {
    auto problem_shape_in = cute::make_tuple(
        options.batch, options.num_heads_q, options.num_heads_kv,
        options.seq_len_qo, options.seq_len_kv, options.seq_len_kv_cache,
        options.head_size_qk, options.head_size_vo);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    static_assert(isVarLen == false,
                  "Variable length sequences not supported in this runner");
    TORCH_CHECK(options.seq_len_kv_cache == 0,
                "sycl-tla attention: KV cache not supported (seq_len_kv_cache=",
                options.seq_len_kv_cache, ")");
    problem_size = problem_shape_in;
    shape.batch = options.batch;
    shape.num_heads_q = options.num_heads_q;
    shape.num_heads_kv = options.num_heads_kv;
    shape.seq_len_qo = options.seq_len_qo;
    shape.seq_len_kv = options.seq_len_kv;
    shape.seq_len_kv_cache = options.seq_len_kv_cache;
    shape.head_size_qk = options.head_size_qk;
    shape.head_size_vo = options.head_size_vo;

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv,
          seq_len_kv_cache, head_size_qk, head_size_vo] = problem_size;
    auto shape_Q =
        cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch);
    auto shape_K =
        cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V =
        cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_K_cache =
        cute::make_shape(seq_len_kv_cache, head_size_qk, num_heads_kv, batch);
    auto shape_V_cache =
        cute::make_shape(head_size_vo, seq_len_kv_cache, num_heads_kv, batch);
    auto shape_O =
        cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch);

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
    stride_K_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
    stride_V_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);

    static_assert(PagedKV == false, "PagedKV not supported in this runner");
    static_assert(isVarLen == false, "only support fixed length");

    return shape;
  }

  static void run(typename FMHAKernel::Params params) {

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
#if (SYCL_INTEL_TARGET == 35)
        intelex::grf_size<512>
#else
        intelex::grf_size<256>
#endif
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block,
                                               launch_props, kernel_props};
    auto event =
        compat::experimental::launch<cutlass::device_kernel<FMHAKernel>,
                                     FMHAKernel>(policy, params);

    EventManager::getInstance().addEvent(event);
  }

  void run(const FMHAOptions &options,
           const cutlass::KernelHardwareInfo &hw_info, const at::Tensor &Q,
           const at::Tensor &K, const at::Tensor &V, at::Tensor &O) {
    RECORD_FUNCTION("sycl-tla fa", {});

    ProblemShapeType problem_size = initialize(options);

    typename FMHAKernel::Arguments arguments{
        {
            problem_size,
            static_cast<const cutlass::half_t *>(Q.data_ptr()),
            stride_Q,
            static_cast<const cutlass::half_t *>(K.data_ptr()),
            stride_K,
            static_cast<const cutlass::half_t *>(V.data_ptr()),
            stride_V,
            static_cast<float *>(O.data_ptr()),
            stride_O,
            nullptr,
            stride_K_cache,
            nullptr,
            stride_V_cache,
        },
        {options.softmax_scale, nullptr, 0, nullptr},
        {},
        hw_info};
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    TORCH_CHECK(FMHAKernel::can_implement(arguments),
                "sycl-tla attention: kernel cannot implement problem size ",
                options.batch, 'x', options.num_heads_q, 'x',
                options.seq_len_qo, 'x', options.seq_len_kv, 'x',
                options.head_size_qk);
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));
    typename FMHAKernel::Params params =
        FMHAKernel::to_underlying_arguments(arguments, workspace.get());
    run(params);

    compat::wait();
  }
};

} // namespace
