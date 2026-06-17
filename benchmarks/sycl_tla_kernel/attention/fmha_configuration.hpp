/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
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

#include <type_traits>

#include "cutlass/gemm/dispatch_policy.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

namespace cutlass {
namespace flash_attention {

template <typename ElementQ, typename ElementK, typename ElementV,
          typename ElementO, typename LayoutQ_, typename LayoutK_,
          typename LayoutV_, typename LayoutO_, typename TileShapeQK,
          typename TileShapePV, typename TileShapeOutput,
          typename SubgroupLayoutQK, typename SubgroupLayoutPV_, bool Causal_,
          bool VarLen_, bool CachedKV_, bool PagedKV_, bool Persistent_,
          int PipelineStages, typename GmemTiledCopyQ = void,
          typename GmemTiledCopyK = void, typename GmemTiledCopyV = void,
          typename GmemTiledCopyO = void, typename MMAOperation_ = void,
          typename StrideQ = Stride<int, _1, int, int>,
          typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>,
          typename StrideO = Stride<int, _1, int, int>>
struct FMHAConfig {
  using LayoutQ = LayoutQ_;
  using LayoutK = LayoutK_;
  using LayoutV = LayoutV_;
  using LayoutO = LayoutO_;

  static constexpr bool Causal = Causal_;
  static constexpr bool VarLen = VarLen_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool Persistent = Persistent_;
  static_assert(!(Persistent & Causal),
                "persistent SDPA kernel not support Causal yet");

  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();

  using DefaultMMA = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;
  using MMAOperationPV = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementV>;

  using MMAOperation =
      cute::conditional_t<is_void_v<MMAOperation_>, DefaultMMA, MMAOperation_>;

  using SubgroupLayoutPV =
      cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                          decltype(cutlass::fmha::collective::get_sg_layout_pv(
                              SubgroupLayoutQK{})),
                          SubgroupLayoutPV_>;

  using TiledMMAQK =
      typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>,
                              SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV =
      typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>,
                              SubgroupLayoutPV>::TiledMMA;
  static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
                "Output tile and P*V tile have different sizes in Q dimension");
  static constexpr int VTiles =
      get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

  template <typename ElementType, typename Stride>
  static constexpr auto make_dummy_tensor(ElementType val, Stride stride) {
    return make_tensor(
        make_gmem_ptr(&val),
        make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };

  using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
  using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
  using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
  using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
  using TensorK_cache = TensorK;
  using TensorV_cache = TensorV;
  using GmemTiledCopyK_cache = GmemTiledCopyK;
  using GmemTiledCopyV_cache = GmemTiledCopyV;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<VarLen>;

  // Mainloop
  using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
      MainloopDispatchPolicy, Causal, CachedKV, PagedKV, TiledMMAQK, TiledMMAPV,
      VTiles, TensorQ, TensorK, TensorV, TensorK_cache, TensorV_cache,
      GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV, GmemTiledCopyK_cache,
      GmemTiledCopyV_cache>;

  using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
      CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

  using Scheduler = cute::conditional_t<
      Persistent,
      cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler,
      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
  using FMHAKernel = cute::conditional_t<
      Persistent,
      cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<
          ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>,
      cutlass::fmha::kernel::XeFMHAFwdKernel<
          ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>>;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum class FMHAMode { Decode, Prefill };

template <FMHAMode Mode, int HeadDim, bool Persistent> struct ShapeConfig;

// Prefill configs
template <bool Persistent>
struct ShapeConfig<FMHAMode::Prefill, 16, Persistent> {
  using ShapeQK = Shape<_16, _16, _32>;
  using ShapePV = Shape<_16, _32, _16>;
  using ShapeOutput = Shape<_16, _16>;
  using SubgroupLayout = Layout<Shape<_1, _1, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Prefill, 64, Persistent> {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _64>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Prefill, 96, Persistent> {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _96>;
  using SubgroupLayout = Layout<Shape<_8, _1, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Prefill, 128, Persistent> {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOutput = Shape<_128, _128>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Prefill, 192, Persistent> {
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOutput = Shape<_256, _192>;
  using SubgroupLayout = Layout<Shape<_16, _1, _1>>;
};

// Decode configs
template <bool Persistent>
struct ShapeConfig<FMHAMode::Decode, 16, Persistent> {
  using ShapeQK = Shape<_1, _16, _16>;
  using ShapePV = Shape<_1, _16, _16>;
  using ShapeOutput = Shape<_1, _16>;
  using SubgroupLayout = Layout<Shape<_1, _2, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Decode, 64, Persistent> {
  using num_sg = cute::conditional_t<Persistent, _16, _8>;
  using kv_tile_size = cute::conditional_t<Persistent, _256, _512>;
  using ShapeQK = Shape<_1, kv_tile_size, _64>;
  using ShapePV = Shape<_1, _32, kv_tile_size>;
  using ShapeOutput = Shape<_1, _64>;
  using SubgroupLayout = Layout<Shape<_1, num_sg, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Decode, 96, Persistent> {
  using num_sg = cute::conditional_t<Persistent, _16, _8>;
  using kv_tile_size = cute::conditional_t<Persistent, _256, _512>;
  using ShapeQK = Shape<_1, kv_tile_size, _64>;
  using ShapePV = Shape<_1, _32, kv_tile_size>;
  using ShapeOutput = Shape<_1, _96>;
  using SubgroupLayout = Layout<Shape<_1, num_sg, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Decode, 128, Persistent> {
  using num_sg = cute::conditional_t<Persistent, _16, _8>;
  using kv_tile_size = cute::conditional_t<Persistent, _256, _512>;
  using ShapeQK = Shape<_1, kv_tile_size, _64>;
  using ShapePV = Shape<_1, _32, kv_tile_size>;
  using ShapeOutput = Shape<_1, _128>;
  using SubgroupLayout = Layout<Shape<_1, num_sg, _1>>;
};

template <bool Persistent>
struct ShapeConfig<FMHAMode::Decode, 192, Persistent> {
  using num_sg = cute::conditional_t<Persistent, _16, _8>;
  using kv_tile_size = cute::conditional_t<Persistent, _256, _512>;
  using ShapeQK = Shape<_1, kv_tile_size, _64>;
  using ShapePV = Shape<_1, _32, kv_tile_size>;
  using ShapeOutput = Shape<_1, _192>;
  using SubgroupLayout = Layout<Shape<_1, num_sg, _1>>;
};

// Mode-dependent pipeline stages
template <FMHAMode Mode> struct PipelineStagesConfig;
template <> struct PipelineStagesConfig<FMHAMode::Decode> {
  static constexpr int value = 1;
};
template <> struct PipelineStagesConfig<FMHAMode::Prefill> {
  static constexpr int value = 2;
};

// FMHAConfigGen using ShapeConfig lookup table
template <FMHAMode Mode, class ElementQ, class ElementK, class ElementV,
          class ElementO, class LayoutQ, class LayoutK, class LayoutV,
          class LayoutO, bool Causal, bool VarLen, bool CachedKV, bool PagedKV,
          bool Persistent, int HeadDim>
struct FMHAConfigGen {
  using TileShapeConfig = ShapeConfig<Mode, HeadDim, Persistent>;
  using type = cutlass::flash_attention::FMHAConfig<
      ElementQ, ElementK, ElementV, ElementO, LayoutQ, LayoutK, LayoutV,
      LayoutO, typename TileShapeConfig::ShapeQK,
      typename TileShapeConfig::ShapePV, typename TileShapeConfig::ShapeOutput,
      typename TileShapeConfig::SubgroupLayout, void, Causal, VarLen, CachedKV,
      PagedKV, Persistent, PipelineStagesConfig<Mode>::value>;
};

// FMHAConfigGen with explicit tile and subgroup specification
template <FMHAMode Mode, class ElementQ, class ElementK, class ElementV,
          class ElementO, class LayoutQ, class LayoutK, class LayoutV,
          class LayoutO, bool Causal, bool VarLen, bool CachedKV, bool PagedKV,
          bool Persistent, int WgTileQ, int WgTileK, int WgTileV, int SgTileQ,
          int SgTileK, int HeadDimQK, int HeadDimV>
struct FMHAConfigGenWithTileShape {
  using ShapeQK = Shape<Int<WgTileQ>, Int<WgTileK>, Int<HeadDimQK>>;
  using ShapePV = Shape<Int<WgTileQ>, Int<WgTileV>,
                        Int<WgTileK>>; // Third dimension = WgTileK (K sequence
                                       // tile, shared with ShapeQK[1])
  using ShapeOutput = Shape<Int<WgTileQ>, Int<HeadDimV>>;

  // Derive subgroup counts from tile ratios for QK matmul
  static_assert(WgTileQ % SgTileQ == 0, "WgTileQ must be divisible by SgTileQ");
  static_assert(WgTileK % SgTileK == 0, "WgTileK must be divisible by SgTileK");

  // SubgroupLayoutQK: (num_sg_q, num_sg_k, 1)
  // Head dimension is never split across subgroups (always _1)
  using SubgroupLayoutQK =
      Layout<Shape<Int<WgTileQ / SgTileQ>, Int<WgTileK / SgTileK>, _1>>;

  // SubgroupLayoutPV: (num_sg_p = num_sg_q, 1, num_sg_k)
  // number of subgroups in PV GEMM equals that in QK GEMM
  using SubgroupLayoutPV =
      Layout<Shape<Int<WgTileQ / SgTileQ>, _1, Int<WgTileK / SgTileK>>>;

  using type = cutlass::flash_attention::FMHAConfig<
      ElementQ, ElementK, ElementV, ElementO, LayoutQ, LayoutK, LayoutV,
      LayoutO, ShapeQK, ShapePV, ShapeOutput, SubgroupLayoutQK,
      SubgroupLayoutPV, Causal, VarLen, CachedKV, PagedKV, Persistent,
      PipelineStagesConfig<Mode>::value>;
};

} // namespace flash_attention
} // namespace cutlass
