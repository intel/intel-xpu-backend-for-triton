import re
import math
import pathlib

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

from triton._internal_testing import (
    integral_dtypes,
    uint_dtypes,
    is_xpu,
    numpy_random,
    to_triton,
    to_numpy,
)

GPU_DIALECT = "ttg"
THREADS_PER_WARP = 32


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

    def __str__(self):
        return f"#{GPU_DIALECT}.dot_op<{{parent={self.parent}, opIdx={self.op_idx}, kWidth={self.k_width}}}>"


class SliceLayout:

    def __init__(self, dim, parent):
        self.dim = dim
        self.parent = parent

    def __str__(self):
        return f"#{GPU_DIALECT}.slice<{{dim = {self.dim}, parent = {self.parent}}}>"


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


class SwizzledSharedLayout:

    def __init__(self, vec, per_phase, max_phase, order, ctas_per_cga, cta_split_num, cta_order):
        self.vec = vec
        self.per_phase = per_phase
        self.max_phase = max_phase
        self.order = order
        self.ctas_per_cga = ctas_per_cga
        self.cta_split_num = cta_split_num
        self.cta_order = cta_order

    def __str__(self):
        return f"#{GPU_DIALECT}.swizzled_shared<{{vec={self.vec}, perPhase={self.per_phase}, maxPhase={self.max_phase}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class PaddedSharedLayout:

    def __init__(self, interval_padding_pairs, linear_layout_offset_bases, linear_layout_block_bases):
        self.interval_padding_pairs = "[" + ", ".join(f"{v[0]}:{v[1]:+d}" for v in interval_padding_pairs) + "]"
        self.offset_bases = linear_layout_offset_bases
        self.block_bases = linear_layout_block_bases

    def __str__(self):
        return f"#{GPU_DIALECT}.padded_shared<{self.interval_padding_pairs} {{offset={self.offset_bases}, block={self.block_bases}}}>"


class LinearLayout:

    def __init__(self, register, lane, warp, block):
        self.register = register
        self.lane = lane
        self.warp = warp
        self.block = block

    def __str__(self):
        return f"#{GPU_DIALECT}.linear<{{register={self.register}, lane={self.lane}, warp={self.warp}, block={self.block}}}>"


# Python impl of LinearEncodingAttr::basesPerDim
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


def is_layout_applicable(layout) -> bool:
    if isinstance(layout, (BlockedLayout, SwizzledSharedLayout, LinearLayout)):
        return True
    elif isinstance(layout, SliceLayout):
        return is_layout_applicable(layout.parent)
    elif is_xpu():
        mma_layout = layout.parent if isinstance(layout, DotOperandLayout) else layout
        return isinstance(mma_layout, DpasLayout)
    else:
        return True


def filter_layouts(layouts):
    return [l for l in layouts if is_layout_applicable(l)]


def get_reduce_input(dtype_str, shape):
    # limit the range of integers so that reduce ops do not overflow
    low = 0 if dtype_str in uint_dtypes else -10 if dtype_str in integral_dtypes else None
    high = 10 if dtype_str in integral_dtypes else None
    return numpy_random(shape, dtype_str=dtype_str, low=low, high=high)


scan_layouts = [
    BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 2], [1, THREADS_PER_WARP // 1], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[32, 16], [32, 32], [32, 64], [64, 32]])
@pytest.mark.parametrize("src_layout", scan_layouts)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("add_overflow_check", [False, True])
def test_scan_layouts(M, N, src_layout, axis, add_overflow_check, device, tmp_path: pathlib.Path):

    overflow_check = """
        %17 = arith.extsi %arg2 : i32 to i64
        %18 = arith.extsi %arg3 : i32 to i64
        %19 = arith.addi %17, %18 : i64
        %i32.min = arith.constant -2147483648: i64
        %i32.max = arith.constant 2147483647: i64
        %20 = arith.cmpi slt, %19, %i32.max : i64
        %21 = arith.cmpi sge, %19, %i32.min : i64
        %22 = arith.andi %20, %21 : i1
        tt.assert %22, "overflow detected" : i1
    """

    ir = f"""
    #blocked = {src_layout}
    module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
      %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
      %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
      %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
      %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #blocked>
      %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %4 = tt.addptr %3, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
      %6 = tt.expand_dims %5 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
      %7 = tt.broadcast %4 : tensor<{M}x1x!tt.ptr<i32>, #blocked> -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %8 = tt.broadcast %6 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
      %9 = tt.addptr %7, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %11 = "tt.scan"(%10) <{{axis = {axis} : i32, reverse = false}}> ({{
      ^bb0(%arg2: i32, %arg3: i32):
        %16 = arith.addi %arg2, %arg3 : i32{overflow_check if add_overflow_check else ""}
        tt.scan.return %16 : i32
      }}) : (tensor<{M}x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %12 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x1x!tt.ptr<i32>, #blocked>
      %13 = tt.addptr %12, %2 : tensor<{M}x1x!tt.ptr<i32>, #blocked>, tensor<{M}x1xi32, #blocked>
      %14 = tt.broadcast %13 : tensor<{M}x1x!tt.ptr<i32>, #blocked> -> tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      %15 = tt.addptr %14, %8 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      tt.store %15, %11 : tensor<{M}x{N}x!tt.ptr<i32>, #blocked>
      tt.return
    }}
    }}
    """

    temp_file = tmp_path / "test_scan_layouts.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(-100, 100, (M, N)).astype('int32')

    z = np.zeros((M, N)).astype('int32')
    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, z_tri)

    z_ref = np.cumsum(x, axis=axis)

    np.testing.assert_equal(z_ref, z_tri.cpu().numpy())


layouts = [
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
               warps_per_cta=[2, 2], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=4, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    LinearLayout(register=[[0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane=[[0, 0], [0, 1], [0, 2], [0, 4],
                                                                                    [0, 8]], warp=[[32, 0], [0, 32]],
                 block=[]),
]


@pytest.mark.parametrize("M, N", [[128, 16], [32, 128], [32, 32], [16, 16]])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("epilogue_kind", ['reduce1d', 'reduce2d', 'expand_reduce2d'])
@pytest.mark.parametrize("dtype_str,add_overflow_check", [("int32", False), ("int32", True), ("float32", False),
                                                          ("float16", False)])
@pytest.mark.parametrize("reduce_op", ["sum", "max"])
def test_reduce_layouts(M, N, src_layout, axis, epilogue_kind, dtype_str, add_overflow_check, reduce_op, device,
                        tmp_path: pathlib.Path):
    if reduce_op == "sum" and dtype_str == "float16" and M * N > 1024:
        pytest.xfail("Skipping sum reduction on float16 due to accuracy issues")
    if isinstance(src_layout, LinearLayout) and THREADS_PER_WARP != (1 << len(src_layout.lane)):
        pytest.xfail(f"Skipping. This LinearLayout assumes {1 << len(src_layout.lane)} threads per warp")

    overflow_check = """
        %18 = arith.extsi %arg3 : i32 to i64
        %19 = arith.extsi %arg4 : i32 to i64
        %20 = arith.addi %18, %19 : i64
        %i32.min = arith.constant -2147483648: i64
        %i32.max = arith.constant 2147483647: i64
        %21 = arith.cmpi slt, %20, %i32.max : i64
        %22 = arith.cmpi sge, %20, %i32.min : i64
        %23 = arith.andi %21, %22 : i1
        tt.assert %23, "overflow detected" : i1
    """

    ty = {"int32": "i32", "float32": "f32", "float16": "f16"}[dtype_str]
    arith_op = {
        "max": {"int32": "arith.maxsi", "float32": "arith.maximumf", "float16": "arith.maximumf"},  #
        "sum": {"int32": "arith.addi", "float32": "arith.addf", "float16": "arith.addf"}
    }[reduce_op][dtype_str]
    numpy_op = {"max": np.max, "sum": np.sum}[reduce_op]
    rdims_1d = f"{N}" if axis == 0 else f"{M}"
    rdims_2d = f"1x{N}" if axis == 0 else f"{M}x1"
    store_range = "%7" if axis == 0 else "%1"
    warps = warps_per_cta(src_layout, [M, N])
    num_warps = int(np.prod(warps))
    blocked = BlockedLayout([1, 1], [32, THREADS_PER_WARP // 32], [4, num_warps // 4], [0, 1], [1, 1], [1, 1], [0, 1])
    one_d_layout = BlockedLayout([1], [THREADS_PER_WARP], [num_warps], [0], [1], [1], [0])

    expanded_shape = f"1x{N}" if axis == 0 else f"{M}x1"
    other_axis = 1 - axis
    epilogue = {
        "reduce1d":
        f"""
        %14 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>
        %15 = tt.addptr %14, {store_range} : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>, tensor<{rdims_2d}xi32, #blocked>
        %16 = {GPU_DIALECT}.convert_layout %13 : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>> -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #blocked}}>>
        %17 = tt.expand_dims %16 {{axis = {axis} : i32}} : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #blocked}}>> -> tensor<{rdims_2d}x{ty}, #blocked>
        tt.store %15, %17 : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>
        tt.return
        }}
        }}
    """, "reduce2d":
        f"""
        %14 = "tt.reduce"(%13) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}{overflow_check if add_overflow_check else ""}
          tt.reduce.return %17 : {ty}
        }}) {{axis = 0 : i32}} : (tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>) -> {ty}
        tt.store %arg2, %14 : !tt.ptr<{ty}>
        tt.return
        }}
        }}
    """, "expand_reduce2d":
        f"""
        %14 = tt.expand_dims %13 {{axis = {axis} : i32}} : tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>> -> tensor<{expanded_shape}x{ty}, #src>
        %15 = "tt.reduce"(%14) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}{overflow_check if add_overflow_check else ""}
          tt.reduce.return %17 : {ty}
        }}) {{axis = {other_axis} : i32}} : (tensor<{expanded_shape}x{ty}, #src>) -> (tensor<1x{ty}, #{GPU_DIALECT}.slice<{{dim = {other_axis}, parent = #src}}>>)
        %16 = ttg.convert_layout %15 : tensor<1x{ty}, #{GPU_DIALECT}.slice<{{dim = {other_axis}, parent = #src}}>> -> tensor<1x{ty}, #one_d_layout>
        %17 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<1x!tt.ptr<{ty}>, #one_d_layout>
        tt.store %17, %16 : tensor<1x!tt.ptr<{ty}>, #one_d_layout>
        tt.return
        }}
        }}
                """
    }[epilogue_kind]

    ir = f"""
    #blocked = {blocked}
    #src = {src_layout}
    #one_d_layout = {one_d_layout}
    module attributes {{"ttg.num-warps" = {num_warps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
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
          %17 = {arith_op} %arg3, %arg4 : {ty}{overflow_check if add_overflow_check else ""}
          tt.reduce.return %17 : {ty}
        }}) {{axis = {axis} : i32}} : (tensor<{M}x{N}x{ty}, #src>) -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>
    """ + epilogue

    temp_file = tmp_path / "test_reduce_layouts.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    x = get_reduce_input(dtype_str, (M, N))
    reduce2d = 'reduce2d' in epilogue_kind
    z_shape = (1, 1) if reduce2d else (1, N) if axis == 0 else (M, 1)
    z = np.zeros(z_shape).astype(dtype_str)

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, x_tri.stride(0), z_tri)
    z_ref = numpy_op(x) if reduce2d else numpy_op(x, axis=axis, keepdims=True)

    if dtype_str == 'float16':
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
]


@pytest.mark.parametrize("M", [32, 64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
def test_store_op(M, src_layout, device, tmp_path: pathlib.Path):

    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "{GPU_DIALECT}.num-ctas" = 1 : i32, "{GPU_DIALECT}.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %2 = tt.addptr %1, %0 : tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<f32>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %4 = tt.expand_dims %3 {{axis = 1 : i32}} : tensor<{M}xf32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xf32, #src>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %6 = tt.expand_dims %5 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
            %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<{M}x1x!tt.ptr<f32>, #src>
            %8 = tt.addptr %7, %6 : tensor<{M}x1x!tt.ptr<f32>, #src>, tensor<{M}x1xi32, #src>
            tt.store %8, %4 : tensor<{M}x1x!tt.ptr<f32>, #src>
            tt.return
        }}
    }}
    """

    temp_file = tmp_path / "test_store_op.ttgir"
    temp_file.write_text(ir)
    store_kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, 1)).astype('float32')
    y = np.zeros((M, 1), dtype='float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)

    store_kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1])
]


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("dst_layout", filter_layouts(layouts))
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
def test_convert1d(M, src_layout, dst_layout, src_dim, dst_dim, device, tmp_path: pathlib.Path):

    ir = f"""
    #dst = {dst_layout}
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %2 = tt.addptr %0, %1 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %6 = tt.addptr %4, %5 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %7 = {GPU_DIALECT}.convert_layout %3 : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>> -> tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.store %6, %7 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.return
        }}
    }}
    """
    temp_file = tmp_path / "test_convert1d.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, )).astype('int32')
    y = np.zeros((M, ), dtype='int32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("dst_layout", filter_layouts(layouts))
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
def test_convert1d_bool(M, src_layout, dst_layout, src_dim, dst_dim, device, tmp_path: pathlib.Path):

    ir = f"""
    #dst = {dst_layout}
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<i8> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %cst = arith.constant dense<0> : tensor<{M}xi8, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %0 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<{M}x!tt.ptr<i8>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %2 = tt.addptr %0, %1 : tensor<{M}x!tt.ptr<i8>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %3 = tt.load %2 : tensor<{M}x!tt.ptr<i8>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %4 = arith.cmpi ne, %3, %cst : tensor<{M}xi8, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %6 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %7 = tt.addptr %5, %6 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %8 = {GPU_DIALECT}.convert_layout %4 : tensor<{M}xi1, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>> -> tensor<{M}xi1, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %9 = arith.extui %8 : tensor<{M}xi1, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>> to tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.store %7, %9 : tensor<{M}x!tt.ptr<i32>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.return
        }}
    }}
    """
    temp_file = tmp_path / "test_convert1d.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 2, (M, )).astype('bool')
    y = np.zeros((M, ), dtype='int32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0, atol=0)


layouts = [
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 32, 32], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1])
]


@pytest.mark.parametrize("M, N", [[128, 128], [256, 128], [256, 256], [128, 256]])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("first_axis", [0, 1])
def test_chain_reduce(M, N, src_layout, op, device, first_axis, tmp_path: pathlib.Path):

    op_str = ""
    if op == "sum":
        op_str = """
        %13 = arith.addi %arg2, %arg3 : i32
        tt.reduce.return %13 : i32"""
    elif op == "max":
        op_str = """
        %13 = arith.cmpi "sgt", %arg2, %arg3 : i32
        %14 = arith.select %13, %arg2, %arg3 : i32
        tt.reduce.return %14 : i32"""
    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
        %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #src>
        %3 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>>
        %4 = tt.expand_dims %3 {{axis = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
        %5 = tt.broadcast %2 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %6 = tt.broadcast %4 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %7 = arith.addi %5, %6 : tensor<{M}x{N}xi32, #src>
        %8 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %9 = tt.addptr %8, %7 : tensor<{M}x{N}x!tt.ptr<i32>, #src>, tensor<{M}x{N}xi32, #src>
        %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %11 = "tt.reduce"(%10) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = {first_axis} : i32}} : (tensor<{M}x{N}xi32, #src>) -> tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>
        %12 = "tt.reduce"(%11) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = 0 : i32}} : (tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>) -> i32
        tt.store %arg1, %12 : !tt.ptr<i32>
        tt.return
    }}
    }}
    """
    temp_file = tmp_path / "test_chain_reduce.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, N)).astype('int32')

    z = np.zeros((1, )).astype('int32')

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, z_tri)
    if op == "sum":
        z_ref = np.sum(x)
    elif op == "max":
        z_ref = np.max(x)

    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


# -----------------------
# test layout conversions
# -----------------------
# TODO: backend should be tested separately

layouts = [
    BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 16], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=2, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1])
]

intermediate_layouts = [
    None,
    SwizzledSharedLayout(1, 1, 1, [0, 1], [1, 1], [1, 1], [0, 1]),
    SwizzledSharedLayout(1, 1, 1, [1, 0], [1, 1], [1, 1], [0, 1]),
    SwizzledSharedLayout(4, 2, 4, [1, 0], [1, 1], [1, 1], [0, 1]),
    SwizzledSharedLayout(2, 2, 4, [1, 0], [1, 1], [1, 1], [0, 1]),
]


def compute_rep_shape(layout):
    if type(layout) is BlockedLayout:
        warp_shape = np.multiply(layout.sz_per_thread, layout.threads_per_warp)
        rep_shape = np.multiply(warp_shape, layout.warps_per_cta)
        return rep_shape
    elif type(layout) is DpasLayout:
        warp_shape = [layout.repeatCount, layout.execution_size]
        warp_shape = np.multiply(warp_shape, layout.rep_cluster)
        rep_shape = np.multiply(warp_shape, layout.warps_per_cta)
        return rep_shape
    else:
        assert False, "TODO: support compute_rep_shape for layout " + str(type(layout))


# This function gives a lower bound approximation of scratch buffer shape for convert_layout operation
def compute_scratch_buffer_shape(src_layout, dst_layout, shape):
    src_rep_shape = compute_rep_shape(src_layout)
    dst_rep_shape = compute_rep_shape(dst_layout)
    full_scratch_shape = np.maximum(src_rep_shape, dst_rep_shape)
    return np.minimum(full_scratch_shape, shape)


@pytest.mark.parametrize("M, N", [[64, 1], [64, 64], [64, 128], [1, 64]])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("src_layout", filter_layouts(layouts))
@pytest.mark.parametrize("interm_layout", intermediate_layouts)
@pytest.mark.parametrize("dst_layout", filter_layouts(layouts))
def test_convert2d(M, N, src_layout, interm_layout, dst_layout, dtype, device, tmp_path: pathlib.Path):
    if str(src_layout) == str(dst_layout):
        pytest.xfail("Do not convert same layout")
    if is_xpu():
        try:
            scratch_shape = compute_scratch_buffer_shape(src_layout, dst_layout, (M, N))
        except AssertionError:
            if is_xpu():
                # expect compute scratch buffer to not error on xpu
                raise
            pytest.skip("Can't compute scratch buffer size")
        lds_size = triton.runtime.driver.active.utils.get_device_properties(
            triton.runtime.driver.active.get_current_device())["max_shared_mem"]
        # consider int32 dtype in scratch buffer size,
        # because it is the largest dtype used in convert_layout in this test
        int32_size = 4
        # skip even if scratch buffer equal to lds_size, because real scratch buffer is typically larger due to padding
        if scratch_shape[0] * scratch_shape[1] * int32_size >= lds_size:
            pytest.xfail("Scratch buffer is too large")

    layouts = f"""
    #src = {src_layout}
    #dst = {dst_layout}
    #smem = #ttg.shared_memory
    """ if interm_layout is None else f"""
    #src = {src_layout}
    #interm = {interm_layout}
    #dst = {dst_layout}
    #smem = #ttg.shared_memory
    """

    conversion = f"""
    %12 = ttg.convert_layout %9 : tensor<{M}x{N}xi32, #src> -> tensor<{M}x{N}xi32, #dst>
    %13 = ttg.convert_layout %11 : tensor<{M}x{N}xf16, #src> -> tensor<{M}x{N}xf16, #dst>
    """ if interm_layout is None else f"""
    %15 = ttg.local_alloc %9 : (tensor<{M}x{N}xi32, #src>) -> !ttg.memdesc<{M}x{N}xi32, #interm, #smem>
    %16 = ttg.local_load %15 : !ttg.memdesc<{M}x{N}xi32, #interm, #smem> -> tensor<{M}x{N}xi32, #src>
    %17 = ttg.local_alloc %11 : (tensor<{M}x{N}xf16, #src>) -> !ttg.memdesc<{M}x{N}xf16, #interm, #smem>
    %18 = ttg.local_load %17 : !ttg.memdesc<{M}x{N}xf16, #interm, #smem> -> tensor<{M}x{N}xf16, #src>

    %12 = ttg.convert_layout %16 : tensor<{M}x{N}xi32, #src> -> tensor<{M}x{N}xi32, #dst>
    %13 = ttg.convert_layout %18 : tensor<{M}x{N}xf16, #src> -> tensor<{M}x{N}xf16, #dst>
    """

    ir = layouts + f"""
    module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
  tt.func public @kernel_0d1d(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
    %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #src>
    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #src>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
    %7 = tt.broadcast %6 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
    %8 = tt.broadcast %5 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
    %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #src>
    %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #src>, tensor<{M}x{N}xi32, #src>
    %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<f16>, #src>
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #dst>
    """ + conversion + f"""
    %14 = tt.addptr %3, %12 : tensor<{M}x{N}x!tt.ptr<f16>, #dst>, tensor<{M}x{N}xi32, #dst>
    tt.store %14, %13 : tensor<{M}x{N}x!tt.ptr<f16>, #dst>
    tt.return
  }}
}}
"""

    x = to_triton(numpy_random((M, N), dtype_str=dtype), device=device)
    z = torch.empty_like(x, device=device)

    temp_file = tmp_path / "test_convert2d.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    torch.testing.assert_close(z, x, rtol=0, atol=0)


layouts_3d = [
    BlockedLayout([4, 4, 1], [1, 8, THREADS_PER_WARP // 8], [2, 2, 1], [2, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
    BlockedLayout([1, 1, 4], [8, THREADS_PER_WARP // 8, 1], [2, 1, 2], [1, 2, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
]

shared_layouts_3d = [
    SwizzledSharedLayout(1, 1, 1, [2, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
    SwizzledSharedLayout(4, 2, 4, [1, 2, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
    SwizzledSharedLayout(8, 2, 4, [0, 2, 1], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
    SwizzledSharedLayout(4, 2, 1, [2, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 2]),
]


@pytest.mark.parametrize("M, N, K", [[8, 16, 32]])
@pytest.mark.parametrize("shared_layout", shared_layouts_3d)
@pytest.mark.parametrize("dist_layout", filter_layouts(layouts_3d))
def test_local_load_store(M, N, K, dist_layout, shared_layout, device, tmp_path: pathlib.Path):
    layouts = f"""
    #dist = {dist_layout}
    #shared = {shared_layout}
    #smem = #ttg.shared_memory
    """
    ir = layouts + f"""
  module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
  tt.func public @kernel(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
    %cst = arith.constant dense<{K}> : tensor<1x{N}x1xi32, #dist>
    %cst_0 = arith.constant dense<{K*N}> : tensor<{M}x1x1xi32, #dist>
    %cst_1 = arith.constant dense<{K*N}> : tensor<{M}x1x1xi32, #dist>
    %cst_2 = arith.constant dense<{K}> : tensor<1x{N}x1xi32, #dist>
    %0 = tt.make_range {{end = {K} : i32, start = 0 : i32}} : tensor<{K}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 1, parent = #dist}}>}}>>
    %1 = tt.expand_dims %0 {{axis = 0 : i32}} : tensor<{K}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 1, parent = #dist}}>}}>> -> tensor<1x{K}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>>
    %2 = tt.expand_dims %1 {{axis = 1 : i32}} : tensor<1x{K}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>> -> tensor<1x1x{K}xi32, #dist>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x{K}x!tt.ptr<i32>, #dist>
    %4 = tt.addptr %3, %2 : tensor<1x1x{K}x!tt.ptr<i32>, #dist>, tensor<1x1x{K}xi32, #dist>
    %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>>
    %6 = tt.expand_dims %5 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>> -> tensor<1x{N}xi32, #ttg.slice<{{dim = 2, parent = #dist}}>>
    %7 = tt.expand_dims %6 {{axis = 2 : i32}} : tensor<1x{N}xi32, #ttg.slice<{{dim = 2, parent = #dist}}>> -> tensor<1x{N}x1xi32, #dist>
    %8 = arith.muli %7, %cst_2 : tensor<1x{N}x1xi32, #dist>
    %9 = tt.broadcast %4 : tensor<1x1x{K}x!tt.ptr<i32>, #dist> -> tensor<1x{N}x{K}x!tt.ptr<i32>, #dist>
    %10 = tt.broadcast %8 : tensor<1x{N}x1xi32, #dist> -> tensor<1x{N}x{K}xi32, #dist>
    %11 = tt.addptr %9, %10 : tensor<1x{N}x{K}x!tt.ptr<i32>, #dist>, tensor<1x{N}x{K}xi32, #dist>
    %12 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>>
    %13 = tt.expand_dims %12 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>> -> tensor<{M}x1xi32, #ttg.slice<{{dim = 2, parent = #dist}}>>
    %14 = tt.expand_dims %13 {{axis = 2 : i32}} : tensor<{M}x1xi32, #ttg.slice<{{dim = 2, parent = #dist}}>> -> tensor<{M}x1x1xi32, #dist>
    %15 = arith.muli %14, %cst_1 : tensor<{M}x1x1xi32, #dist>
    %16 = tt.broadcast %11 : tensor<1x{N}x{K}x!tt.ptr<i32>, #dist> -> tensor<{M}x{N}x{K}x!tt.ptr<i32>, #dist>
    %17 = tt.broadcast %15 : tensor<{M}x1x1xi32, #dist> -> tensor<{M}x{N}x{K}xi32, #dist>
    %18 = tt.addptr %16, %17 : tensor<{M}x{N}x{K}x!tt.ptr<i32>, #dist>, tensor<{M}x{N}x{K}xi32, #dist>
    %19 = tt.load %18 : tensor<{M}x{N}x{K}x!tt.ptr<i32>, #dist>
    %20 = ttg.local_alloc %19 : (tensor<{M}x{N}x{K}xi32, #dist>) -> !ttg.memdesc<{M}x{N}x{K}xi32, #shared, #smem>
    %21 = ttg.local_load %20 : !ttg.memdesc<{M}x{N}x{K}xi32, #shared, #smem> -> tensor<{M}x{N}x{K}xi32, #dist>
    %22 = tt.make_range {{end = {K} : i32, start = 0 : i32}} : tensor<{K}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 1, parent = #dist}}>}}>>
    %23 = tt.expand_dims %22 {{axis = 0 : i32}} : tensor<{K}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 1, parent = #dist}}>}}>> -> tensor<1x{K}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>>
    %24 = tt.expand_dims %23 {{axis = 1 : i32}} : tensor<1x{K}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>> -> tensor<1x1x{K}xi32, #dist>
    %25 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x1x{K}x!tt.ptr<i32>, #dist>
    %26 = tt.addptr %25, %24 : tensor<1x1x{K}x!tt.ptr<i32>, #dist>, tensor<1x1x{K}xi32, #dist>
    %27 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>>
    %28 = tt.expand_dims %27 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>> -> tensor<1x{N}xi32, #ttg.slice<{{dim = 2, parent = #dist}}>>
    %29 = tt.expand_dims %28 {{axis = 2 : i32}} : tensor<1x{N}xi32, #ttg.slice<{{dim = 2, parent = #dist}}>> -> tensor<1x{N}x1xi32, #dist>
    %30 = arith.muli %29, %cst : tensor<1x{N}x1xi32, #dist>
    %31 = tt.broadcast %26 : tensor<1x1x{K}x!tt.ptr<i32>, #dist> -> tensor<1x{N}x{K}x!tt.ptr<i32>, #dist>
    %32 = tt.broadcast %30 : tensor<1x{N}x1xi32, #dist> -> tensor<1x{N}x{K}xi32, #dist>
    %33 = tt.addptr %31, %32 : tensor<1x{N}x{K}x!tt.ptr<i32>, #dist>, tensor<1x{N}x{K}xi32, #dist>
    %34 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>>
    %35 = tt.expand_dims %34 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #ttg.slice<{{dim = 2, parent = #dist}}>}}>> -> tensor<{M}x1xi32, #ttg.slice<{{dim = 2, parent = #dist}}>>
    %36 = tt.expand_dims %35 {{axis = 2 : i32}} : tensor<{M}x1xi32, #ttg.slice<{{dim = 2, parent = #dist}}>> -> tensor<{M}x1x1xi32, #dist>
    %37 = arith.muli %36, %cst_0 : tensor<{M}x1x1xi32, #dist>
    %38 = tt.broadcast %33 : tensor<1x{N}x{K}x!tt.ptr<i32>, #dist> -> tensor<{M}x{N}x{K}x!tt.ptr<i32>, #dist>
    %39 = tt.broadcast %37 : tensor<{M}x1x1xi32, #dist> -> tensor<{M}x{N}x{K}xi32, #dist>
    %40 = tt.addptr %38, %39 : tensor<{M}x{N}x{K}x!tt.ptr<i32>, #dist>, tensor<{M}x{N}x{K}xi32, #dist>
    tt.store %40, %21 : tensor<{M}x{N}x{K}x!tt.ptr<i32>, #dist>
    tt.return
  }}
}}
"""

    x = torch.arange(0, M * N * K, device=device, dtype=torch.int32).reshape(M, N, K)
    z = torch.empty_like(x, device=device)

    temp_file = tmp_path / "test_local_load_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](x, z)
    assert torch.equal(z, x)


dot_layouts = [
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
                          warps_per_cta=[4, 1], rep_cluster=[1, 1]), op_idx=0, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
                          warps_per_cta=[2, 1], rep_cluster=[1, 1]), op_idx=0, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=32,
                          warps_per_cta=[2, 1], rep_cluster=[1, 1]), op_idx=0, k_width=2),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
                          warps_per_cta=[4, 1], rep_cluster=[1, 1]), op_idx=1, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=2, threads_per_warp=32,
                          warps_per_cta=[4, 1], rep_cluster=[1, 1]), op_idx=1, k_width=2),
]

shared_layouts = [
    SwizzledSharedLayout(4, 2, 4, [0, 1], [1, 1], [1, 1], [0, 1]),
    SwizzledSharedLayout(8, 1, 8, [1, 0], [1, 1], [1, 1], [0, 1]),
    SwizzledSharedLayout(16, 1, 16, [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N, M_tile_size, N_tile_size",
                         [[128, 128, 64, 64], [128, 128, 64, 32], [128, 64, 64, 32], [256, 128, 64, 64]])
def test_split_subview(M, N, M_tile_size, N_tile_size, device, tmp_path: pathlib.Path):
    num_rows_per_warp = THREADS_PER_WARP // 4
    num_repeats_M = triton.cdiv(M, M_tile_size)
    num_repeats_N = triton.cdiv(N, N_tile_size)

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread=[1, 8], threadsPerWarp=[{num_rows_per_warp}, 4], warpsPerCTA=[4, 1], order=[1, 0], CTAsPerCGA=[1, 1], CTASplitNum=[1, 1], CTAOrder=[0, 1]}}>
    #shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}}>
    #smem = #ttg.shared_memory

    module attributes {{"ttg.num-ctas" = 1, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
        %cst_n = arith.constant dense<{N_tile_size}> : tensor<{M_tile_size}x1xi32, #blocked>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
        %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{M}x1xi32, #blocked>
        %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #blocked>
        %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x{N}xi32, #blocked>
        %7 = tt.broadcast %6 : tensor<1x{N}xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : tensor<{M}x1xi32, #blocked> -> tensor<{M}x{N}xi32, #blocked>
        %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #blocked>
        %ptrs = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %ptrs {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>

        %c0_i32 = arith.constant 0 : i32

        %12 = ttg.local_alloc : () -> !ttg.memdesc<1x{M}x{N}xf16, #shared, #smem, mutable>
        %13 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<1x{M}x{N}xf16, #shared, #smem, mutable> -> !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable>
        ttg.local_store %11, %13 : tensor<{M}x{N}xf16, #blocked> -> !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable>

    """

    for m in range(num_repeats_M):
        for n in range(num_repeats_N):
            linear_idx = n + m * num_repeats_N
            m_offset = m * M_tile_size
            n_offset = n * N_tile_size
            ir += f"""
        %view{linear_idx} = ttg.memdesc_subslice %13[{m_offset}, {n_offset}] : !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable> -> !ttg.memdesc<{M_tile_size}x{N_tile_size}xf16, #shared, #smem, mutable, {M}x{N}>
        %data{linear_idx} = ttg.local_load %view{linear_idx} : !ttg.memdesc<{M_tile_size}x{N_tile_size}xf16, #shared, #smem, mutable, {M}x{N}> -> tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        %inc{linear_idx} = arith.constant dense<{linear_idx}.0> : tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>

        %res{linear_idx} = arith.addf %data{linear_idx}, %inc{linear_idx} : tensor<{M_tile_size}x{N_tile_size}xf16, #blocked>
        ttg.local_store %res{linear_idx}, %view{linear_idx} : tensor<{M_tile_size}x{N_tile_size}xf16, #blocked> -> !ttg.memdesc<{M_tile_size}x{N_tile_size}xf16, #shared, #smem, mutable, {M}x{N}>
        """

    ir += f"""
        %res = ttg.local_load %13 : !ttg.memdesc<{M}x{N}xf16, #shared, #smem, mutable> -> tensor<{M}x{N}xf16, #blocked>
        tt.store %ptrs, %res : tensor<{M}x{N}x!tt.ptr<f16>, #blocked>
        tt.return
    }}
    }}
    """

    temp_file = tmp_path / "test_split_subview.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    triton_result = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel[(1, 1, 1)](triton_result.data_ptr())

    rows = []
    for m in range(num_repeats_M):
        columns = []
        for n in range(num_repeats_N):
            linear_idx = n + m * num_repeats_N
            tile = float(linear_idx) * torch.ones((M_tile_size, N_tile_size), device=device, dtype=torch.float16)
            columns.append(tile)
        rows.append(torch.cat(columns, dim=1))
    expected_result = torch.cat(rows, dim=0)

    test_result = torch.equal(triton_result, expected_result)
    assert test_result


@pytest.mark.parametrize("M, N", [[16, 32]])
@pytest.mark.parametrize("dtype", ['float16', 'float8e5', 'float32'])
@pytest.mark.parametrize("shared_layout", shared_layouts)
@pytest.mark.parametrize("dist_layout", filter_layouts(dot_layouts))
def test_local_load_store_dot(M, N, dtype, dist_layout, shared_layout, device, tmp_path: pathlib.Path):
    if dtype == "float32":
        mlir_dtype = "f32"
    elif dtype == "float16":
        mlir_dtype = "f16"
    elif dtype == "float8e5":
        mlir_dtype = "f8E5M2"

    num_warps = 4
    if isinstance(dist_layout, DotOperandLayout) and isinstance(dist_layout.parent, DpasLayout):
        num_warps = math.prod(dist_layout.parent.warps_per_cta)

    layouts = f"""
    #dist = {dist_layout}
    #shared = {shared_layout}
    #smem = #ttg.shared_memory
    """
    ir = layouts + f"""
  module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @kernel(%arg0: !tt.ptr<{mlir_dtype}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{mlir_dtype}> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
    %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #dist>
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dist}}>>
    %2 = tt.splat %arg0 : !tt.ptr<{mlir_dtype}> -> tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dist>
    %3 = tt.splat %arg1 : !tt.ptr<{mlir_dtype}> -> tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dist>
    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>> -> tensor<{M}x1xi32, #dist>
    %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #dist>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dist}}>> -> tensor<1x{N}xi32, #dist>
    %7 = tt.broadcast %6 : tensor<1x{N}xi32, #dist> -> tensor<{M}x{N}xi32, #dist>
    %8 = tt.broadcast %5 : tensor<{M}x1xi32, #dist> -> tensor<{M}x{N}xi32, #dist>
    %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #dist>
    %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dist>, tensor<{M}x{N}xi32, #dist>
    %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dist>
    %12 = ttg.local_alloc %11 : (tensor<{M}x{N}x{mlir_dtype}, #dist>) -> !ttg.memdesc<{M}x{N}x{mlir_dtype}, #shared, #smem>
    %13 = ttg.local_load %12 : !ttg.memdesc<{M}x{N}x{mlir_dtype}, #shared, #smem> -> tensor<{M}x{N}x{mlir_dtype}, #dist>
    %14 = tt.addptr %3, %9 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dist>, tensor<{M}x{N}xi32, #dist>
    tt.store %14, %13 : tensor<{M}x{N}x!tt.ptr<{mlir_dtype}>, #dist>
    tt.return
  }}
}}
"""

    x = to_triton(numpy_random((M, N), dtype_str=dtype), device=device)
    z = torch.empty_like(x, device=device)

    temp_file = tmp_path / "test_local_load_store_dot.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](x, z)
    assert torch.equal(z, x)


mma_layouts = [
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[1, 1]),
]

shared_layouts = [
    SwizzledSharedLayout(8, 1, 1, [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[128, 128]])
@pytest.mark.parametrize("mma_layout", filter_layouts(mma_layouts))
@pytest.mark.parametrize("shared_layout", shared_layouts)
def test_local_load_store_mma(M, N, mma_layout, shared_layout, device, tmp_path: pathlib.Path):
    num_warps = np.prod(mma_layout.warps_per_cta)

    layouts = f"""
    #dist = {mma_layout}
    #shared = {shared_layout}
    #smem = #ttg.shared_memory
    """
    ir = layouts + f"""
  module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
  tt.func public @kernel(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
    %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #dist>
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dist}}>>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #dist>
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #dist>
    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dist}}>> -> tensor<{M}x1xi32, #dist>
    %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #dist>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dist}}>> -> tensor<1x{N}xi32, #dist>
    %7 = tt.broadcast %6 : tensor<1x{N}xi32, #dist> -> tensor<{M}x{N}xi32, #dist>
    %8 = tt.broadcast %5 : tensor<{M}x1xi32, #dist> -> tensor<{M}x{N}xi32, #dist>
    %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #dist>
    %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #dist>, tensor<{M}x{N}xi32, #dist>
    %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<f16>, #dist>
    %12 = ttg.local_alloc %11 : (tensor<{M}x{N}xf16, #dist>) -> !ttg.memdesc<{M}x{N}xf16, #shared, #smem>
    %13 = ttg.local_load %12 : !ttg.memdesc<{M}x{N}xf16, #shared, #smem> -> tensor<{M}x{N}xf16, #dist>
    %14 = tt.addptr %3, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #dist>, tensor<{M}x{N}xi32, #dist>
    tt.store %14, %13 : tensor<{M}x{N}x!tt.ptr<f16>, #dist>
    tt.return
  }}
}}
"""

    x = torch.arange(0, M * N, device=device, dtype=torch.float16).reshape(M, N)
    z = torch.empty_like(x, device=device)

    temp_file = tmp_path / "test_local_load_store_mma.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](x, z)
    assert torch.equal(z, x)


def filter_layout_pairs(layout_pairs):
    return [pair for pair in layout_pairs if is_layout_applicable(pair[0]) and is_layout_applicable(pair[1])]


mma_pairs = [
    [
        DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=1, threads_per_warp=32,
                   warps_per_cta=[4, 1], rep_cluster=[1, 1]),
        DpasLayout(repeatCount=8, systolic_depth=8, execution_size=8, ops_per_chan=2, threads_per_warp=32,
                   warps_per_cta=[2, 2], rep_cluster=[1, 1]),
    ],
]


@pytest.mark.parametrize("M, N", [[16, 16], [64, 1], [1, 64], [64, 64], [128, 128], [256, 256]])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("mma_pair", filter_layout_pairs(mma_pairs))
def test_convert_mma2mma(M, N, mma_pair, dtype, device, tmp_path: pathlib.Path):
    src_layout, _ = mma_pair
    num_warps = np.prod(src_layout.warps_per_cta)
    warp_size = THREADS_PER_WARP

    def do_test(src_layout, dst_layout):
        layouts = f"""
        #src = {src_layout}
        #dst = {dst_layout}
        """

        ir = layouts + f"""
        module attributes {{"ttg.num-warps" = {num_warps} : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {warp_size} : i32}} {{
        tt.func public @kernel_0d1d(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>>
        %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>>
        %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #src>
        %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #dst>
        %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
        %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #src>
        %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
        %7 = tt.broadcast %6 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %8 = tt.broadcast %5 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #src>
        %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #src>, tensor<{M}x{N}xi32, #src>
        %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<f16>, #src>
        %12 = ttg.convert_layout %9 : tensor<{M}x{N}xi32, #src> -> tensor<{M}x{N}xi32, #dst>
        %13 = ttg.convert_layout %11 : tensor<{M}x{N}xf16, #src> -> tensor<{M}x{N}xf16, #dst>
        %14 = tt.addptr %3, %12 : tensor<{M}x{N}x!tt.ptr<f16>, #dst>, tensor<{M}x{N}xi32, #dst>
        tt.store %14, %13 : tensor<{M}x{N}x!tt.ptr<f16>, #dst>
        tt.return
        }}
        }}
        """

        x = to_triton(numpy_random((M, N), dtype_str=dtype), device=device)
        z = torch.empty_like(x)

        temp_file = tmp_path / "test_convert_mma2mma.ttgir"
        temp_file.write_text(ir)
        kernel = triton.compile(str(temp_file))

        kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

        assert torch.equal(z, x)

    do_test(mma_pair[0], mma_pair[1])
    do_test(mma_pair[1], mma_pair[0])


single_warp_layouts = [
    BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [1, 1], [1, 0]),
    BlockedLayout([1, 1], [THREADS_PER_WARP // 2, 2], [1, 1], [1, 0]),
    BlockedLayout([1, 1], [THREADS_PER_WARP // 4, 4], [1, 1], [1, 0]),
    BlockedLayout([1, 1], [THREADS_PER_WARP // 8, 8], [1, 1], [1, 0]),
    BlockedLayout([1, 1], [THREADS_PER_WARP // 16, 16], [1, 1], [1, 0]),
    BlockedLayout([1, 1], [THREADS_PER_WARP // 32, 32], [1, 1], [1, 0]),
    BlockedLayout([32, 1], [1, THREADS_PER_WARP], [1, 1], [1, 0]),
    BlockedLayout([16, 1], [2, THREADS_PER_WARP // 2], [1, 1], [1, 0]),
    BlockedLayout([1, 4], [THREADS_PER_WARP, 1], [1, 1], [1, 0]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 2, 2], [1, 1], [1, 0]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 4, 4], [1, 1], [1, 0]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 8, 8], [1, 1], [1, 0]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 16, 16], [1, 1], [1, 0]),
    BlockedLayout([1, 4], [THREADS_PER_WARP // 32, 32], [1, 1], [1, 0]),
]


@pytest.mark.parametrize("M, N", [[32, 32], [64, 64]])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("src_layout", single_warp_layouts)
@pytest.mark.parametrize("dst_layout", single_warp_layouts)
def test_convert_warp_local(M, N, src_layout, dst_layout, dtype, device, tmp_path: pathlib.Path):
    if str(src_layout) == str(dst_layout):
        pytest.xfail()
    if np.prod(src_layout.threads_per_warp) == 0 or np.prod(dst_layout.threads_per_warp) == 0:
        pytest.xfail()

    # Test layout pairs that are likely to codegen warp shuffles.
    a, b = list(np.array(src_layout.threads_per_warp) // np.array(dst_layout.threads_per_warp))
    c = a if a != 0 else b
    if c > 2:
        pytest.xfail()

    layouts = f"""
    #src = {src_layout}
    #dst = {dst_layout}
    #smem = #ttg.shared_memory
    """

    conversion = f"""
    %12 = ttg.convert_layout %9 : tensor<{M}x{N}xi32, #src> -> tensor<{M}x{N}xi32, #dst>
    %13 = ttg.convert_layout %11 : tensor<{M}x{N}xf16, #src> -> tensor<{M}x{N}xf16, #dst>
    """

    ir = layouts + f"""
    module attributes {{"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
  tt.func public @kernel_0d1d(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
    %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #src>
    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #src>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
    %7 = tt.broadcast %6 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
    %8 = tt.broadcast %5 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
    %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #src>
    %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16>, #src>, tensor<{M}x{N}xi32, #src>
    %11 = tt.load %10 : tensor<{M}x{N}x!tt.ptr<f16>, #src>
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<{M}x{N}x!tt.ptr<f16>, #dst>
    """ + conversion + f"""
    %14 = tt.addptr %3, %12 : tensor<{M}x{N}x!tt.ptr<f16>, #dst>, tensor<{M}x{N}xi32, #dst>
    tt.store %14, %13 : tensor<{M}x{N}x!tt.ptr<f16>, #dst>
    tt.return
  }}
}}
"""

    x = to_triton(numpy_random((M, N), dtype_str=dtype), device=device)
    z = torch.empty_like(x, device=device)

    temp_file = tmp_path / "test_convert_warp_local.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    torch.testing.assert_close(z, x, rtol=0, atol=0)


@triton.jit
def gather_test_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                       src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr,
                       idx_dim1: tl.constexpr, idx_stride0: tl.constexpr, idx_stride1: tl.constexpr,
                       out_dim0: tl.constexpr, out_dim1: tl.constexpr, out_stride0: tl.constexpr,
                       out_stride1: tl.constexpr):
    src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
    src = tl.load(src_ptr + src_offs)

    idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, axis)

    out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
    tl.store(out_ptr + out_offs, out)


def gen_gather_warp_shuffle_cases():
    if THREADS_PER_WARP == 32:
        return [
            ([32, 16], [32, 16], 0,
             "linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>",
             "linear<{register = [[2, 0], [0, 2]], lane = [[0, 8], [16, 0], [1, 0], [8, 0], [4, 0]], warp = [[0, 1], [0, 4]], block = []}>"
             ),
            ([128, 64], [256, 64], 0,
             "linear<{register = [[0, 2], [32, 0], [2, 0], [0, 16], [0, 32], [64, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>",
             "linear<{register = [[0, 2], [32, 0], [0, 32], [2, 0], [0, 16], [64, 0], [128, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>"
             ),
        ]
    elif THREADS_PER_WARP == 64:
        return [
            ([64, 16], [64, 16], 0,
             "linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0], [32, 0]], warp = [[0, 1], [0, 4]], block = []}>",
             "linear<{register = [[2, 0], [0, 2]], lane = [[0, 8], [16, 0], [1, 0], [8, 0], [4, 0], [32, 0]], warp = [[0, 1], [0, 4]], block = []}>"
             ),
            ([128, 64], [256, 64], 0,
             "linear<{register = [[0, 2], [2, 0], [0, 16], [0, 32]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0], [32, 0]], warp = [[0, 1], [0, 4]], block = []}>",
             "linear<{register = [[0, 2], [0, 32], [2, 0], [0, 16], [64, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0], [32, 0]], warp = [[0, 1], [0, 4]], block = []}>"
             ),
        ]
    else:
        return []


# These layouts are specially chosen to trigger the warp shuffle codegen.
@pytest.mark.parametrize("src_shape, indices_shape, axis, src_layout, indices_layout", gen_gather_warp_shuffle_cases())
def test_gather_warp_shuffle(src_shape, indices_shape, axis, src_layout, indices_layout, tmp_path: pathlib.Path,
                             device):

    def prepare_kernel(src: torch.Tensor, axis: int, indices: torch.Tensor):
        output = torch.empty(indices.shape, dtype=src.dtype, device=src.device)
        compiled = gather_test_kernel.warmup(src, indices, output, axis, src.shape[0], src.shape[1], src.stride(0),
                                             src.stride(1), indices.shape[0], indices.shape[1], indices.stride(0),
                                             indices.stride(1), output.shape[0], output.shape[1], output.stride(0),
                                             output.stride(1), grid=(1, ))
        return output, compiled

    def inject_layout(ir, src: torch.Tensor, axis, indices: torch.Tensor, src_layout, idx_layout):
        ir = f"""
#src_layout = #ttg.{src_layout}
#idx_layout = #ttg.{idx_layout}
{ir}"""

        dtypes = {torch.int32: "i32", torch.float32: "f32", torch.int64: "i64", torch.float64: "f64"}

        src_spec = f"{src.shape[0]}x{src.shape[1]}x{dtypes[src.dtype]}"
        indices_spec = f"{indices.shape[0]}x{indices.shape[1]}x{dtypes[indices.dtype]}"
        output_spec = f"{indices.shape[0]}x{indices.shape[1]}x{dtypes[src.dtype]}"

        pat = r"(%[0-9]+) = tt.gather (%[0-9]+)\[(%[0-9]+)\] {axis = "
        pat += str(axis)
        pat += r" : i32, efficient_layout} : \(tensor\<"
        pat += src_spec
        pat += r", (#[a-z]+[0-9]+)\>, tensor\<"
        pat += indices_spec
        pat += r", (#[a-z]+[0-9]+)\>\) -> tensor\<"
        pat += output_spec
        pat += r", (#[a-z]+[0-9]+)\>"

        repl = r"""
    %src = ttg.convert_layout \2 : tensor<""" + src_spec + r""", \4> -> tensor<""" + src_spec + r""", #src_layout>
    %idx = ttg.convert_layout \3 : tensor<""" + indices_spec + r""", \5> -> tensor<""" + indices_spec + r""", #idx_layout>
    %out = tt.gather %src[%idx] {axis = """ + str(
            axis
        ) + r""" : i32, efficient_layout} : (tensor<""" + src_spec + r""", #src_layout>, tensor<""" + indices_spec + r""", #idx_layout>) -> tensor<""" + output_spec + r""", #idx_layout>
    \1 = ttg.convert_layout %out : tensor<""" + output_spec + r""", #idx_layout> -> tensor<""" + output_spec + r""", \6>"""
        return re.sub(pat, repl, ir)

    src = torch.randn(src_shape, device=device)
    indices = torch.randint(0, src.shape[axis], indices_shape, device=device)
    ref = torch.gather(src, axis, indices)

    output, compiled = prepare_kernel(src, axis, indices)
    ir = compiled.asm["ttgir"]
    ir = inject_layout(ir, src, axis, indices, src_layout, indices_layout)
    assert ir != compiled.asm["ttgir"]

    temp_file = tmp_path / "test_warp_gather.ttgir"
    temp_file.write_text(ir)

    kernel = triton.compile(str(temp_file))
    assert ("nvvm.shfl.sync.idx" in kernel.asm["llir"]) or ("llvm.amdgcn.ds.bpermute"
                                                            in kernel.asm["llir"]) or ("_Z17sub_group_shufflefj"
                                                                                       in kernel.asm["llir"])

    kernel[(1, 1, 1)](src, indices, output)

    torch.testing.assert_close(output, ref, rtol=0, atol=0)
