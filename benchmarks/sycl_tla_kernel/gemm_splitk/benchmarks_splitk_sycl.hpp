#pragma once

#include "gemm/legacy/gemm_configuration_sycl.hpp"

using SplitKScheduler = cutlass::gemm::device::Scheduler;

using SplitKMMAAtom = MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>;

template <typename TileShape, typename Tiler, typename GmemTiledCopyA,
          typename GmemTiledCopyB>
using Gemm_Bench_SplitK_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe, cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, float,
    cutlass::layout::RowMajor, float, TileShape, SplitKScheduler::GemmSplitK,
    Tiler, GmemTiledCopyA, GmemTiledCopyB>;

using SplitKTile_1 =
    TiledMMA<SplitKMMAAtom, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
             Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                  Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;
using PvcGemmSplitKBF16BF16FP32_RRR_1 =
    Gemm_Bench_SplitK_BF16FP32_RRR<Shape<_256, _256, _32>, SplitKTile_1,
                                   XE_2D_U16x32x32_LD_N, XE_2D_U16x32x32_LD_V>;

// The stream-K reference runs the same kernel (the Xe stream-K tile scheduler
// serves both decompositions; the mode is a runtime argument), but has its own
// config name and shape table so that it can be tuned independently of split-K.
using PvcGemmStreamKBF16BF16FP32_RRR_1 =
    Gemm_Bench_SplitK_BF16FP32_RRR<Shape<_256, _256, _32>, SplitKTile_1,
                                   XE_2D_U16x32x32_LD_N, XE_2D_U16x32x32_LD_V>;

CUTLASS_CREATE_GEMM_BENCHMARK(PvcGemmSplitKBF16BF16FP32_RRR_1);
CUTLASS_CREATE_GEMM_BENCHMARK(PvcGemmStreamKBF16BF16FP32_RRR_1);
