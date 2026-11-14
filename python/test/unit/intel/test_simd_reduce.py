import os
import re
import numpy as np
from numpy.random import RandomState
import pytest
import torch
import pathlib

import triton
import triton.language as tl
from triton._internal_testing import numpy_random, to_numpy

MIN_GROUP_SIZE = torch.xpu.get_device_capability()['sub_group_sizes'][0]

os.environ["TRITON_INTEL_ENABLE_SIMD_REDUCE"] = "1"


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
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


layouts = [
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
               warps_per_cta=[2, 2], rep_cluster=[1, 1]),
    BlockedLayout([1, 1], [1, 16], [2, 2], [1, 0])
]

if MIN_GROUP_SIZE == 16:
    # Add threads_per_warp=32 cases.
    layouts += [
        DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
                   warps_per_cta=[2, 2], rep_cluster=[1, 1]),
        BlockedLayout([1, 1], [1, 32], [2, 2], [1, 0])
    ]


def warps_per_cta(layout, shape):
    return layout.warps_per_cta


GPU_DIALECT = "ttg"


@pytest.mark.parametrize("M, N", [[8, 64], [16, 64], [32, 64], [64, 64], [128, 64]])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("dtype_str", ["float32", "float16"])
@pytest.mark.parametrize("reduce_op", ["sum", "max"])
def test_horizontal_simd_reduce(M, N, src_layout, dtype_str, reduce_op, device, tmp_path: pathlib.Path):
    ty = {"int32": "i32", "float32": "f32", "float16": "f16"}[dtype_str]
    arith_op = {
        "max": {"int32": "arith.maxsi", "float32": "arith.maxnumf", "float16": "arith.maxnumf"},  #
        "sum": {"int32": "arith.addi", "float32": "arith.addf", "float16": "arith.addf"}
    }[reduce_op][dtype_str]
    numpy_op = {"max": np.max, "sum": np.sum}[reduce_op]
    rdims_1d = f"{M}"
    rdims_2d = f"{M}x1"
    store_range = "%1"
    warps = src_layout.warps_per_cta
    threads_per_warp = int(np.prod(src_layout.threads_per_warp))
    num_warps = int(np.prod(warps))
    blocked = BlockedLayout([1, 1], [16, threads_per_warp // 16], [4, num_warps // 4], [0, 1], [1, 1], [1, 1], [0, 1])
    one_d_layout = BlockedLayout([1], [threads_per_warp], [num_warps], [0], [1], [1], [0])

    ir = f"""
    #blocked = {blocked}
    #src = {src_layout}
    #one_d_layout = {one_d_layout}
    module attributes {{"ttg.num-warps" = {num_warps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32, "ttig.min_sg_size" = {MIN_GROUP_SIZE} }} {{
    tt.func public @kernel_0d1d2c3d4c(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: i32 {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %2 = tt.splat %arg1 : i32 -> tensor<{M}x1xi32, #blocked>
        %3 = arith.muli %1, %2 : tensor<{M}x1xi32, #blocked>
        %4 = tt.splat %arg0 : !tt.ptr<{ty}> -> tensor<{M}x1x!tt.ptr<{ty}>, #blocked>
        %5 = tt.addptr %4, %3 : tensor<{M}x1x!tt.ptr<{ty}>, #blocked>, tensor<{M}x1xi32, #blocked>
        %6 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #blocked}}>>
        %7 = tt.expand_dims %6 {{axis = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1x!tt.ptr<{ty}>, #blocked> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #blocked>
        %9 = tt.broadcast %7 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %10 = tt.addptr %8, %9 : tensor<{M}x{N}x!tt.ptr<{ty}>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<{ty}>, #blocked>
        %12 = {GPU_DIALECT}.convert_layout %11 : tensor<{M}x{N}x{ty}, #blocked> -> tensor<{M}x{N}x{ty}, #src>
        %13 = "tt.reduce"(%12) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}
          tt.reduce.return %17 : {ty}
        }}) {{axis = 1 : i32}} : (tensor<{M}x{N}x{ty}, #src>) -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
        %14 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>
        %15 = tt.addptr %14, {store_range} : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>, tensor<{rdims_2d}xi32, #blocked>
        %16 = {GPU_DIALECT}.convert_layout %13 : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>>
        %17 = tt.expand_dims %16 {{axis = 1 : i32}} : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{rdims_2d}x{ty}, #blocked>
        tt.store %15, %17 : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>
        tt.return
        }}
        }}
    """

    temp_file = tmp_path / "test_reduce_layouts.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = numpy_random((M, N), dtype_str=dtype_str, rs=rs, low=0, high=10)
    z_shape = (M, 1)
    z = np.zeros(z_shape).astype(dtype_str)

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, x_tri.stride(0), z_tri)
    z_ref = numpy_op(x, axis=1, keepdims=True)

    if dtype_str == 'float16':
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)

    llir = kernel.asm['llir']
    assert re.search(r'call .* asm', llir), 'no inline visa in llir'  # inline visa is used


@pytest.mark.parametrize("xblock, num_warps", [(1, 2), (8, 4), (32, 8)])
def test_simd_reduce_slice_stack_sum(xblock, num_warps, device):

    @triton.jit
    def kernel(input_ptr, copy_ptr, sum_ptr, XBLOCK: tl.constexpr):
        x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        r = tl.arange(0, 64)[None, :]
        column = x % 128
        batch = x // 128
        values = tl.load(input_ptr + column + 129 * r + 8256 * batch).to(tl.float32)
        total = tl.sum(values, 1)[:, None]
        tl.store(copy_ptr + column + 128 * r + 8192 * batch, values)
        tl.store(sum_ptr + column + 896 * batch, total)

    torch.manual_seed(17)
    inputs = torch.randn((256, 64, 129), device=device, dtype=torch.float16)
    copied = torch.empty((256, 64, 128), device=device, dtype=torch.float16)
    sum_storage = torch.full((256, 896), -1, device=device, dtype=torch.float16)
    summed = sum_storage[:, 768:]
    compiled = kernel[(32768 // xblock, )](inputs, copied, summed, xblock, num_warps=num_warps)
    torch.testing.assert_close(copied, inputs[:, :, :128], rtol=0, atol=0)
    torch.testing.assert_close(summed, inputs[:, :, :128].float().sum(1).half(), rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(sum_storage[:, :768], torch.full_like(sum_storage[:, :768], -1), rtol=0, atol=0)
    if xblock == 8:
        assert re.search(r'call .* asm', compiled.asm['llir']), 'regression must exercise SIMD reduction assembly'
