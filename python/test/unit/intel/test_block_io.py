import os
import itertools

import numpy as np
import pytest
import torch
import pathlib

import triton
from triton._internal_testing import is_xpu

os.environ["TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"] = "1"


class DpasLayout:

    def __init__(self, repeatCount, systolic_depth, execution_size, ops_per_chan, threads_per_warp, warps_per_cta,
                 rep_cluster):
        self.repeatCount = repeatCount
        self.systolic_depth = systolic_depth
        self.execution_size = execution_size
        self.ops_per_chan = ops_per_chan
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.rep_cluster = rep_cluster

    def __str__(self):
        return f"#ttig.dpas<{{repeatCount={self.repeatCount}, systolicDepth={self.systolic_depth}, executionSize = {self.execution_size}, opsPerChan = {self.ops_per_chan}, threadsPerWarp = {self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, repCluster={self.rep_cluster}}}>"


class DotOperandLayout:

    def __init__(self, parent, op_idx, k_width):
        self.parent = parent
        self.op_idx = op_idx
        self.k_width = k_width
        self.threads_per_warp = parent.threads_per_warp

    def __str__(self):
        return f"#ttg.dot_op<{{parent={self.parent}, opIdx={self.op_idx}, kWidth={self.k_width}}}>"


class SliceLayout:

    def __init__(self, dim, parent):
        self.dim = dim
        self.parent = parent
        self.threads_per_warp = parent.threads_per_warp

    def __str__(self):
        return f"#ttg.slice<{{dim = {self.dim}, parent = {self.parent}}}>"


class BlockedLayout:

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga=[1, 1],
                 cta_split_num=[1, 1], cta_order=[0, 1]):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#ttg.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


def warps_per_cta(layout):
    if isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent)
    else:
        return layout.warps_per_cta


layouts = [
    BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    # DPAS layout
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
               warps_per_cta=[2, 2], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=4, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    # DotOp A
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=0, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=16,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=0, k_width=1),
    # DotOp B
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=1, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=1, k_width=2),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=1, k_width=4),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=1, k_width=1),
    # Slice layout
    SliceLayout(dim=1, parent=BlockedLayout([1, 4, 1], [2, 1, 16], [2, 1, 2], [2, 1, 0], [1, 1, 1], [1, 1, 1],
                                            [0, 1, 2])),
]


@pytest.mark.parametrize("M, N", [[M, N] for M, N in itertools.product([32, 64, 128], [64, 128])])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("load_block_ptr, store_block_ptr", [(True, True), (False, False), (True, False),
                                                             (False, True)])
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_block_io(M, N, dtype_str, layout, load_block_ptr, store_block_ptr, device, tmp_path: pathlib.Path):

    warps = warps_per_cta(layout)
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    threads_per_warp = int(np.prod(threads_per_warp))

    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]

    support_block_io = torch.xpu.get_device_capability()['has_subgroup_2d_block_io']

    if load_block_ptr:
        load_ops = f"""
            %src_ptr = tt.make_tensor_ptr %src, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%c0_i32, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #layout>>
            %store_val = tt.load %src_ptr {{ttig.block_io = "row_major", boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}} : !tt.ptr<tensor<{M}x{N}x{ty}, #layout>>
            """
    else:
        load_ops = f"""
            %src_base = tt.splat %src : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %src_ptr = tt.addptr %src_base, %row_major_off : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            %store_val = tt.load %src_ptr {{ttig.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            """
    if store_block_ptr:
        store_ops = f"""
            %blk_ptr = tt.make_tensor_ptr %dst, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%c0_i32, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #layout>>
            tt.store %blk_ptr, %store_val {{ttig.block_io = "row_major", boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{M}x{N}x{ty}, #layout>>
            """
    else:
        store_ops = f"""
            %dst_base = tt.splat %dst : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %dst_ptr = tt.addptr %dst_base, %row_major_off : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            tt.store %dst_ptr, %store_val {{ttig.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            """

    ir = f"""
    #layout = {layout}
    module attributes {{{"ttig.support_sg_2d_block," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @block_store(%src: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %dst: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{

            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            %stride_N = arith.constant dense<{N}> : tensor<{M}x1xi32, #layout>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #layout}}>>
            %2 = tt.expand_dims %1 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #layout}}>> -> tensor<{M}x1xi32, #layout>
            %row_stride = arith.muli %2, %stride_N : tensor<{M}x1xi32, #layout>
            %4 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #layout}}>>
            %5 = tt.expand_dims %4 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #layout}}>> -> tensor<1x{N}xi32, #layout>
            %6 = tt.broadcast %row_stride : tensor<{M}x1xi32, #layout> -> tensor<{M}x{N}xi32, #layout>
            %7 = tt.broadcast %5 : tensor<1x{N}xi32, #layout> -> tensor<{M}x{N}xi32, #layout>
            %row_major_off = arith.addi %6, %7 : tensor<{M}x{N}xi32, #layout>

            {load_ops}
            {store_ops}

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)

    temp_file = tmp_path / "test_block_io.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)

    if support_block_io:
        assert 'spirv_Subgroup2DBlockStoreINTEL' in kernel.asm['llir'] or 'GenISA.LSC2DBlockWrite' in kernel.asm['llir']
