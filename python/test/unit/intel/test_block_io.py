import os
import itertools

import numpy as np
import pytest
import torch
import pathlib

import triton
from triton._internal_testing import is_xpu


@pytest.fixture(autouse=True)
def triton_block_io(monkeypatch):
    monkeypatch.setenv("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS", "1")
    yield


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

    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order):
        self.sz_per_thread = size_per_thread
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.order = order

    def __str__(self):
        return f"#ttg.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}}}>"


def warps_per_cta(layout):
    if isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent)
    else:
        return layout.warps_per_cta


layouts = [
    BlockedLayout([1, 1], [2, 16], [4, 1], [1, 0]),
    # DPAS layout
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[4, 4], rep_cluster=[2, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[4, 1], rep_cluster=[2, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[2, 2], rep_cluster=[1, 1]),
    # DotOp A
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
                          warps_per_cta=[1, 2], rep_cluster=[4, 1]), op_idx=0, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
                          warps_per_cta=[4, 2], rep_cluster=[2, 1]), op_idx=0, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
                          warps_per_cta=[4, 8], rep_cluster=[1, 1]), op_idx=0, k_width=2),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=0, k_width=1),
    # DotOp B
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
                          warps_per_cta=[2, 2], rep_cluster=[1, 1]), op_idx=1, k_width=1),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
                          warps_per_cta=[4, 4], rep_cluster=[2, 2]), op_idx=1, k_width=2),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
                          warps_per_cta=[8, 4], rep_cluster=[4, 4]), op_idx=1, k_width=4),
    DotOperandLayout(
        parent=DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
                          warps_per_cta=[4, 8], rep_cluster=[4, 1]), op_idx=1, k_width=1),
    # Slice layout
    SliceLayout(dim=1, parent=BlockedLayout([1, 4, 1], [2, 1, 16], [2, 1, 2], [2, 1, 0])),
]


@pytest.mark.parametrize("M, N", [[M, N] for M, N in itertools.product([64, 128], [64, 128])])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_block_io(M, N, dtype_str, layout, transpose, device, tmp_path: pathlib.Path):
    assert os.environ["TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"] == "1"
    warps = warps_per_cta(layout)
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    threads_per_warp = int(np.prod(threads_per_warp))

    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]

    support_block_io = triton.runtime.driver.active.get_current_target().arch['has_2d_block_io']

    block_io = "\"column_major\"" if transpose else "\"row_major\""

    load_ops = f"""
            %src_base = tt.splat %src : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %src_ptr = tt.addptr %src_base, {"%col_major_off" if transpose else "%row_major_off" } : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            %store_val = tt.load %src_ptr {{ttig.block_io = {block_io}}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            """
    store_ops = f"""
            %dst_base = tt.splat %dst : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %dst_ptr = tt.addptr %dst_base, %row_major_off : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            tt.store %dst_ptr, %store_val {{ttig.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            """

    ir = f"""
    #layout = {layout}
    module attributes {{{"ttig.support_2d_block_io," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
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

            %stride_M = arith.constant dense<{M}> : tensor<1x{N}xi32, #layout>
            %col_stride = arith.muli %5, %stride_M : tensor<1x{N}xi32, #layout>
            %8 = tt.broadcast %2 : tensor<{M}x1xi32, #layout> -> tensor<{M}x{N}xi32, #layout>
            %9 = tt.broadcast %col_stride : tensor<1x{N}xi32, #layout> -> tensor<{M}x{N}xi32, #layout>
            %col_major_off = arith.addi %8, %9 : tensor<{M}x{N}xi32, #layout>

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

    a = a.permute(1, 0).contiguous().permute(1, 0) if transpose else a

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)

    if support_block_io:
        if isinstance(layout, DotOperandLayout):
            if (layout.op_idx == 0 and layout.k_width == 2) and dtype_str == "float32":
                # The tile width is too large for block load/store
                return
        llir = kernel.asm["llir"]
        assert 'spirv_Subgroup2DBlockStoreINTEL' in llir or 'GenISA.LSC2DBlockWrite' in llir
        load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
        assert load_count > 0 or transpose


# The 2D block I/O tile covers only the inner two dimensions, so every leading
# dimension needs its own stride from the tensor descriptor. Re-deriving it from the
# 2D surface parameters steps a leading dimension by the wrong amount whenever its
# stride differs from `shapes[-2] * strides[-2]`, reading the wrong slice with no
# diagnostic (issue #7882). Only the rank-4 shape below has such a dimension; the
# dense rank-3 shapes agree with the re-derived value and act as controls.
@pytest.mark.parametrize("shape", [[64, 64, 32], [128, 128, 16], [4, 64, 64, 32]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("block_io", ["row_major", "column_major"])
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_block_io_nd(shape, dtype_str, block_io, device, tmp_path: pathlib.Path):
    rank = len(shape)
    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]
    torch_dtype = getattr(torch, dtype_str)
    support_block_io = triton.runtime.driver.active.get_current_target().arch['has_2d_block_io']

    # A column_major load reads along the tile's second-to-last dimension, so that
    # dimension has to vary fastest and, below 32 bits, pack to d32 per lane, or the
    # tile is rejected and never becomes a block load at all.
    transpose = block_io == "column_major"
    size_per_thread, order = [1] * rank, list(reversed(range(rank)))
    if transpose:
        size_per_thread[rank - 2] = 4 // torch_dtype.itemsize
        order[0], order[1] = order[1], order[0]
    layout = BlockedLayout(size_per_thread, [1, 2, 16] if rank == 3 else [1, 1, 2, 16],
                           [8, 4, 1] if rank == 3 else [4, 2, 4, 1], order)

    # A column_major load declares the descriptor's inner two dimensions swapped
    # relative to the loaded tile, so the two types differ and must not be merged. The
    # source is twice as deep and read at index `batch`, so that leading stride is both
    # folded into the base pointer and walked by the result layout.
    desc_shape = shape[:-2] + [shape[-1], shape[-2]] if transpose else list(shape)
    batch = desc_shape[0]
    src_shape = [2 * batch] + desc_shape[1:]

    def consts(name, values, mlir_ty):
        return "\n            ".join(f"%{name}{i} = arith.constant {v} : {mlir_ty}" for i, v in enumerate(values))

    def refs(name):
        return ", ".join(f"%{name}{i}" for i in range(rank))

    def dense_strides(s):
        return list(np.cumprod([1] + s[:0:-1]))[::-1]

    src_type = "x".join(str(s) for s in desc_shape) + f"x{ty}"
    dst_type = "x".join(str(s) for s in shape) + f"x{ty}"

    ir = f"""
    #layout = {layout}
    module attributes {{{"ttig.support_2d_block_io," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {int(np.prod(warps_per_cta(layout)))} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {int(np.prod(layout.threads_per_warp))} : i32}} {{
        tt.func public @block_store(%src: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %dst: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            {consts("src_dim", src_shape, "i32")}
            {consts("src_str", dense_strides(src_shape), "i64")}
            {consts("src_off", [batch] + [0] * (rank - 1), "i32")}
            {consts("dst_dim", shape, "i32")}
            {consts("dst_str", dense_strides(shape), "i64")}
            {consts("dst_off", [0] * rank, "i32")}

            %src_desc = tt.make_tensor_descriptor %src, [{refs("src_dim")}], [{refs("src_str")}] : !tt.ptr<{ty}>, !tt.tensordesc<{src_type}>
            %val = tt.descriptor_load %src_desc[{refs("src_off")}] {{ttig.block_io = "{block_io}", padding = 1 : i32}} : !tt.tensordesc<{src_type}> -> tensor<{dst_type}, #layout>

            %dst_desc = tt.make_tensor_descriptor %dst, [{refs("dst_dim")}], [{refs("dst_str")}] : !tt.ptr<{ty}>, !tt.tensordesc<{dst_type}>
            tt.descriptor_store %dst_desc[{refs("dst_off")}], %val {{ttig.block_io = "row_major"}} : !tt.tensordesc<{dst_type}>, tensor<{dst_type}, #layout>

            tt.return
        }}
    }}
    """

    if torch_dtype.is_floating_point:
        a = torch.randn(src_shape, dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=src_shape, dtype=torch_dtype, device=device)

    x = torch.empty(shape, dtype=torch_dtype, device=device)

    temp_file = tmp_path / "test_block_io_nd.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    tile = a[batch:2 * batch]
    assert torch.equal(tile.transpose(-2, -1) if transpose else tile, x)

    if support_block_io:
        llir = kernel.asm["llir"]
        load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
        assert load_count > 0
