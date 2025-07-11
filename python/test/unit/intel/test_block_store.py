import os
import itertools

import numpy as np
import pytest
import torch
import pathlib

import triton
from triton._internal_testing import is_xpu

os.environ["TRITON_INTEL_ENABLE_BLOCK_IO_STORE_ON_REGULAR_PTR"] = "1"
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


class LinearLayout:

    def __init__(self, register, lane, warp, block):
        self.register = register
        self.lane = lane
        self.warp = warp
        self.block = block

    def __str__(self):
        return f"#ttg.linear<{{register={self.register}, lane={self.lane}, warp={self.warp}, block={self.block}}}>"


def bases_per_dim(layout, dim, rank, skip_broadcast=True):
    assert isinstance(layout, LinearLayout)
    bases = getattr(layout, dim)
    result = [1] * rank

    if not bases:
        return result

    non_zero_idx = None

    for basis in bases:
        # Find the first non-zero index in the current basis
        idx = next((i for i, v in enumerate(basis) if v != 0), None)
        if idx is not None:
            non_zero_idx = idx
            result[idx] *= 2
        elif not skip_broadcast:
            # If no non-zero found and we're not skipping broadcasts, use the last found non-zero index
            assert non_zero_idx is not None
            result[non_zero_idx] *= 2

    return result


def warps_per_cta(layout, shape):
    if isinstance(layout, LinearLayout):
        return bases_per_dim(layout, 'warp', len(shape))
    elif isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent, shape)
    else:
        return layout.warps_per_cta


layouts = [
    BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    # All DPAS layout could use block store
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
    # Slice layout.
    SliceLayout(dim=1, parent=BlockedLayout([1, 4, 2], [2, 1, 16], [2, 1, 2], [2, 1, 0], [1, 1, 1], [1, 1, 1],
                                            [0, 1, 2])),
]


@pytest.mark.parametrize("M, N", [[M, N] for M, N in itertools.product([32, 64, 128], [64, 128])])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_tensor_pointer_block_store(M, N, dtype_str, layout, device, tmp_path: pathlib.Path):

    warps = warps_per_cta(layout, (M, N))
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    threads_per_warp = int(np.prod(threads_per_warp))

    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]

    support_block_io = torch.xpu.get_device_capability()['has_subgroup_2d_block_io']

    ir = f"""
    #layout = {layout}
    module attributes {{{"ttig.support_sg_2d_block," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @tensor_pointer_block_store(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{

            %stride = arith.constant dense<{N}> : tensor<{M}x1xi32, #layout>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #layout}}>>
            %2 = tt.expand_dims %1 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #layout}}>> -> tensor<{M}x1xi32, #layout>
            %3 = arith.muli %2, %stride : tensor<{M}x1xi32, #layout>
            %4 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #layout}}>>
            %5 = tt.expand_dims %4 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #layout}}>> -> tensor<1x{N}xi32, #layout>
            %6 = tt.broadcast %3 : tensor<{M}x1xi32, #layout> -> tensor<{M}x{N}xi32, #layout>
            %7 = tt.broadcast %5 : tensor<1x{N}xi32, #layout> -> tensor<{M}x{N}xi32, #layout>
            %8 = arith.addi %6, %7 : tensor<{M}x{N}xi32, #layout>

            %9 = tt.splat %arg0 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %10 = tt.addptr %9, %8 : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %12 = tt.splat %arg1 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %13 = tt.addptr %12, %8 : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            tt.store %13, %11 {{ttig.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>

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

    temp_file = tmp_path / "test_tensor_pointer_block_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)

    if support_block_io:
        assert 'spirv_Subgroup2DBlockStoreINTEL' in kernel.asm['llir'] or 'GenISA.LSC2DBlockWrite' in kernel.asm['llir']
