import itertools

import numpy as np
import pytest
import torch
import pathlib

import triton
from triton._internal_testing import is_xpu


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


def warps_per_cta(layout):
    return layout.warps_per_cta


layouts = [
    # Layout for Xe
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[1, 1]),
]


@pytest.mark.parametrize("M, N", [[M, N] for M, N in itertools.product([32, 64, 128, 256], [32, 64, 128, 256])])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_tensor_pointer_block_store(M, N, dtype_str, layout, device, tmp_path: pathlib.Path):

    warps = warps_per_cta(layout)
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    ops_per_chan = layout.ops_per_chan
    A_width = 1 if ops_per_chan == 1 else ops_per_chan // 2
    B_width = ops_per_chan

    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]

    support_block_io = torch.xpu.get_device_capability()['has_subgroup_2d_block_io']

    ir = f"""
    #mma = {layout}
    #dot_a = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>
    #dot_b = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>
    module attributes {{{"ttig.support_sg_2d_block," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @tensor_pointer_block_store(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) {{

            // A matrix
            %stride_a = arith.constant dense<{N}> : tensor<{M}x1xi32, #dot_a>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_a}}>>
            %2 = tt.expand_dims %1 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_a}}>> -> tensor<{M}x1xi32, #dot_a>
            %3 = arith.muli %2, %stride_a : tensor<{M}x1xi32, #dot_a>
            %4 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_a}}>>
            %5 = tt.expand_dims %4 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_a}}>> -> tensor<1x{N}xi32, #dot_a>
            %6 = tt.broadcast %3 : tensor<{M}x1xi32, #dot_a> -> tensor<{M}x{N}xi32, #dot_a>
            %7 = tt.broadcast %5 : tensor<1x{N}xi32, #dot_a> -> tensor<{M}x{N}xi32, #dot_a>
            %8 = arith.addi %6, %7 : tensor<{M}x{N}xi32, #dot_a>

            %9 = tt.splat %arg0 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %10 = tt.addptr %9, %8 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>, tensor<{M}x{N}xi32, #dot_a>
            %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %12 = tt.splat %arg1 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %13 = tt.addptr %12, %8 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>, tensor<{M}x{N}xi32, #dot_a>
            tt.store %13, %11 {{ttig.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>

            // B matrix
            %stride_b = arith.constant dense<{N}> : tensor<{M}x1xi32, #dot_b>
            %21 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_b}}>>
            %22 = tt.expand_dims %21 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_b}}>> -> tensor<{M}x1xi32, #dot_b>
            %23 = arith.muli %22, %stride_b : tensor<{M}x1xi32, #dot_b>
            %24 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_b}}>>
            %25 = tt.expand_dims %24 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_b}}>> -> tensor<1x{N}xi32, #dot_b>
            %26 = tt.broadcast %23 : tensor<{M}x1xi32, #dot_b> -> tensor<{M}x{N}xi32, #dot_b>
            %27 = tt.broadcast %25 : tensor<1x{N}xi32, #dot_b> -> tensor<{M}x{N}xi32, #dot_b>
            %28 = arith.addi %26, %27 : tensor<{M}x{N}xi32, #dot_b>

            %29 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %30 = tt.addptr %29, %28 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>, tensor<{M}x{N}xi32, #dot_b>
            %31 = tt.load %30 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %32 = tt.splat %arg3 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %33 = tt.addptr %32, %28 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>, tensor<{M}x{N}xi32, #dot_b>
            tt.store %33, %31 {{ttig.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>

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
    y = torch.empty_like(a)

    temp_file = tmp_path / "test_tensor_pointer_block_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x, a, y)
    assert torch.equal(a, x) and torch.equal(a, y)
