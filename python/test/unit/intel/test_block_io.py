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
@pytest.mark.parametrize("load_block_ptr, store_block_ptr", [(True, True), (False, False), (True, False),
                                                             (False, True)])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_block_io(M, N, dtype_str, layout, load_block_ptr, store_block_ptr, transpose, device, tmp_path: pathlib.Path):
    assert os.environ["TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"] == "1"
    warps = warps_per_cta(layout)
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    threads_per_warp = int(np.prod(threads_per_warp))

    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]

    support_block_io = triton.runtime.driver.active.get_current_target().arch['has_2d_block_io']

    block_io = "\"column_major\"" if transpose else "\"row_major\""

    strides = "[%c1_i64, %M_i64]" if transpose else "[%N_i64, %c1_i64]"

    if load_block_ptr:
        load_ops = f"""
            %src_ptr = tt.make_tensor_ptr %src, [%M_i64, %N_i64], {strides}, [%c0_i32, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #layout>>
            %store_val = tt.load %src_ptr {{ttig.block_io = {block_io}, boundaryCheck = array<i32: 0, 1>, padding = 1 : i32}} : !tt.ptr<tensor<{M}x{N}x{ty}, #layout>>
            """
    else:
        load_ops = f"""
            %src_base = tt.splat %src : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
            %src_ptr = tt.addptr %src_base, {"%col_major_off" if transpose else "%row_major_off" } : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>, tensor<{M}x{N}xi32, #layout>
            %store_val = tt.load %src_ptr {{ttig.block_io = {block_io}}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #layout>
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


@pytest.mark.parametrize("shape", [[64, 64, 32], [128, 128, 16], [4, 64, 64, 32]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_block_io_nd(shape, dtype_str, transpose, device, tmp_path: pathlib.Path):
    # Determine rank and shape
    rank = len(shape)

    if rank == 3:
        layout = BlockedLayout([1, 1, 1], [1, 2, 16], [8, 4, 1], [2, 1, 0])
    else:
        layout = BlockedLayout([1, 1, 1, 1], [1, 1, 2, 16], [4, 2, 4, 1], [3, 2, 1, 0])

    # Generate IR constants for shape and strides
    shapes_ir = "\n".join([f"%dim{i}_i64 = arith.constant {s} : i64" for i, s in enumerate(shape)])
    strides_row_major = [shape[-2] * shape[-1], shape[-1], 1]
    strides_col_major = [shape[-2] * shape[-1], 1, shape[-2]]
    for s in reversed(shape[1:-2]):
        strides_row_major.insert(0, strides_row_major[0] * s)
        strides_col_major.insert(0, strides_col_major[0] * s)
    strides_row_major_ir = "\n".join(
        [f"%stride_row_major_{i}_i64 = arith.constant {s} : i64" for i, s in enumerate(strides_row_major)])
    strides_col_major_ir = "\n".join(
        [f"%stride_col_major_{i}_i64 = arith.constant {s} : i64" for i, s in enumerate(strides_col_major)])
    strides_row_major_list = [f"%stride_row_major_{i}_i64" for i in range(rank)]
    strides_col_major_list = [f"%stride_col_major_{i}_i64" for i in range(rank)]

    ty = {"float32": "f32", "float16": "f16", "bfloat16": "i16", "int8": "i8"}[dtype_str]
    support_block_io = triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)

    # Build tensor type string
    tensor_type = "x".join(str(s) for s in shape) + f"x{ty}"

    # Build order for make_tensor_ptr (reverse order for IR)
    order = ", ".join(str(i) for i in reversed(range(rank)))

    # Build boundaryCheck array
    boundary_check = ", ".join(str(i) for i in range(rank))

    block_io = "\"column_major\"" if transpose else "\"row_major\""

    ir = f"""
    #layout = {layout}
    module attributes {{{"ttig.support_2d_block_io," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {int(np.prod(warps_per_cta(layout)))} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {int(np.prod(layout.threads_per_warp))} : i32}} {{
        tt.func public @block_store(%src: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %dst: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            {shapes_ir}
            {strides_row_major_ir}
            {strides_col_major_ir}
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            %src_ptr = tt.make_tensor_ptr %src, [{", ".join([f"%dim{i}_i64" for i in range(rank)])}], {"[" + ", ".join(strides_col_major_list if transpose else strides_row_major_list) + "]"}, [{", ".join(["%c0_i32"]*rank)}] {{order = array<i32: {order}>}} : <tensor<{tensor_type}, #layout>>
            %store_val = tt.load %src_ptr {{ttig.block_io = {block_io}, boundaryCheck = array<i32: {boundary_check}>, padding = 1 : i32}} : !tt.ptr<tensor<{tensor_type}, #layout>>

            %dst_ptr = tt.make_tensor_ptr %dst, [{", ".join([f"%dim{i}_i64" for i in range(rank)])}], {"[" + ", ".join(strides_row_major_list) + "]"}, [{", ".join(["%c0_i32"]*rank)}] {{order = array<i32: {order}>}} : <tensor<{tensor_type}, #layout>>
            tt.store %dst_ptr, %store_val {{ttig.block_io = "row_major", boundaryCheck = array<i32: {boundary_check}>}} : !tt.ptr<tensor<{tensor_type}, #layout>>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn(shape, dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=shape, dtype=torch_dtype, device=device)

    x = torch.empty_like(a)

    temp_file = tmp_path / "test_block_io_nd.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    if transpose:
        perm = list(range(rank))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        a = a.permute(*perm).contiguous().permute(*perm)

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)

    if support_block_io:
        llir = kernel.asm["llir"]
        load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
        assert load_count > 0 or transpose


@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
def test_block_io_4d_blocked(device, tmp_path: pathlib.Path):
    """Test block IO with a rank-4 blocked layout.

    Uses the specific layout:
      #blocked = #ttg.blocked<{sizePerThread = [1, 1, 4, 1],
                               threadsPerWarp = [1, 4, 1, 8],
                               warpsPerCTA = [4, 1, 1, 2],
                               order = [2, 1, 3, 0]}>
    and tensor<4x4x4x16xi32> to verify that 4D block IO loads and stores
    round-trip correctly.
    """
    layout = BlockedLayout([1, 1, 4, 1], [1, 4, 1, 8], [4, 1, 1, 2], [2, 1, 3, 0])
    num_warps = int(np.prod(warps_per_cta(layout)))
    threads_per_warp = int(np.prod(layout.threads_per_warp))
    support_block_io = triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)

    ir = f"""
    #layout = {layout}
    module attributes {{{"ttig.support_2d_block_io," if support_block_io else ""} "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @block_io_4d(%src: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %dst: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
            %dim0 = arith.constant 4 : i64
            %dim1 = arith.constant 4 : i64
            %dim2 = arith.constant 4 : i64
            %dim3 = arith.constant 16 : i64
            %stride0 = arith.constant 256 : i64
            %stride1 = arith.constant 64 : i64
            %stride2 = arith.constant 16 : i64
            %stride3 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            %src_ptr = tt.make_tensor_ptr %src, [%dim0, %dim1, %dim2, %dim3], [%stride0, %stride1, %stride2, %stride3], [%c0_i32, %c0_i32, %c0_i32, %c0_i32] {{order = array<i32: 3, 2, 1, 0>}} : <tensor<4x4x4x16xi32, #layout>>
            %val = tt.load %src_ptr {{ttig.block_io = "row_major", boundaryCheck = array<i32: 0, 1, 2, 3>, padding = 1 : i32}} : !tt.ptr<tensor<4x4x4x16xi32, #layout>>

            %dst_ptr = tt.make_tensor_ptr %dst, [%dim0, %dim1, %dim2, %dim3], [%stride0, %stride1, %stride2, %stride3], [%c0_i32, %c0_i32, %c0_i32, %c0_i32] {{order = array<i32: 3, 2, 1, 0>}} : <tensor<4x4x4x16xi32, #layout>>
            tt.store %dst_ptr, %val {{ttig.block_io = "row_major", boundaryCheck = array<i32: 0, 1, 2, 3>}} : !tt.ptr<tensor<4x4x4x16xi32, #layout>>

            tt.return
        }}
    }}
    """

    a = torch.randint(low=-127, high=128, size=[4, 4, 4, 16], dtype=torch.int32, device=device)
    x = torch.empty_like(a)

    temp_file = tmp_path / "test_block_io_4d_blocked.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)

    if support_block_io:
        llir = kernel.asm["llir"]
        load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
        assert load_count > 0
