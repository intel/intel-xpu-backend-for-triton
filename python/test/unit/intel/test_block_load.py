import itertools

import numpy as np
import pytest
import torch
import pathlib

import triton
from triton._internal_testing import is_xpu


@pytest.mark.parametrize("M, N", [[256, 64], [256, 32], [128, 32], [128, 16], [128, 8], [64, 64], [64, 32], [32, 32]])
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
        return f"#triton_intel_gpu.dpas<{{repeatCount={self.repeatCount}, systolicDepth={self.systolic_depth}, executionSize = {self.execution_size}, opsPerChan = {self.ops_per_chan}, threadsPerWarp = {self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, repCluster={self.rep_cluster}}}>"


def warps_per_cta(layout):
    return layout.warps_per_cta


@pytest.mark.parametrize("M, N", [[256, 64], [256, 32], [128, 32], [64, 64], [64, 32], [32, 32]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not torch.xpu.get_device_capability()['has_subgroup_2d_block_io'],
                   reason="Block loads not supported on this architecture")
def test_block_load_dpas_layout(M, N, dtype_str, transpose, device, tmp_path: pathlib.Path):
    # modify the layouts to ensure the correct OCL/SPIRV intrinsic is called for each datatype
    if dtype_str == "int8":
        A_width = 2
        B_width = 4
        layouts = "#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 2], A = [8, 32], B = [32, 32], C = [8, 32]}>"
    elif dtype_str == "float32":
        A_width = 1
        B_width = 1
        layouts = "#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>"
    else:
        A_width = 1
        B_width = 2
        layouts = "#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>"

    block_io = "\"column_major\"" if transpose else "\"row_major\""

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = layouts + f"""
    module attributes {{triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_dpas_layout(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32
            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix
            %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %2 = tt.load %1 {{boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %3 = tt.make_tensor_ptr %arg1, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            tt.store %3, %2 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>

            // B matrix
            %4 = tt.make_tensor_ptr %arg2, [%N_i64, %M_i64], {"[%c1_i64, %N_i64]" if transpose else "[%M_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %5 = tt.load %4 {{boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = {block_io} }} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %6 = tt.make_tensor_ptr %arg3, [%N_i64, %M_i64], {"[%c1_i64, %N_i64]" if transpose else "[%M_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            tt.store %6, %5 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
        b = torch.randn((N, M), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)
        b = torch.randint(low=-127, high=128, size=(N, M), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)
    y = torch.empty_like(b.T if transpose else b)

    temp_file = tmp_path / "test_block_load_dpas_layout.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x, b, y)
    #import pdb; pdb.set_trace()
    assert torch.equal(a, x) and torch.equal(b.T if transpose else b, y)


layouts = [
    # Layout for Xe2 and Xe2+
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=32,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[8, 4], rep_cluster=[1, 1]),
    # Layout for Xe
]


@pytest.mark.parametrize("M, N", [[M, N] for M, N in itertools.product([32, 64, 128, 256], [32, 64, 128, 256])])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
def test_tensor_pointer_block_load(M, N, dtype_str, layout, device, tmp_path: pathlib.Path):

    warps = warps_per_cta(layout)
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    ops_per_chan = layout.ops_per_chan
    A_width = 1 if ops_per_chan == 1 else ops_per_chan // 2
    B_width = ops_per_chan

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    support_block_io = torch.xpu.get_device_capability()['has_subgroup_2d_block_io']

    ir = f"""
    #mma = {layout}
    #dot_a = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>
    #dot_b = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>
    module attributes {{triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, {"triton_intel_gpu.support_sg_2d_block," if support_block_io else ""} triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @tensor_pointer_block_load(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg6: i32 {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg7: i32 {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
            // A matrix
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_a}}>>
            %2 = tt.expand_dims %1 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_a}}>> -> tensor<{M}x1xi32, #dot_a>
            %3 = tt.splat %arg6 : i32 -> tensor<{M}x1xi32, #dot_a>
            %4 = arith.muli %2, %3 : tensor<{M}x1xi32, #dot_a>
            %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_a}}>>
            %6 = tt.expand_dims %5 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_a}}>> -> tensor<1x{N}xi32, #dot_a>
            %7 = tt.broadcast %4 : tensor<{M}x1xi32, #dot_a> -> tensor<{M}x{N}xi32, #dot_a>
            %8 = tt.broadcast %6 : tensor<1x{N}xi32, #dot_a> -> tensor<{M}x{N}xi32, #dot_a>
            %9 = arith.addi %7, %8 : tensor<{M}x{N}xi32, #dot_a>

            %10 = tt.splat %arg0 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %11 = tt.addptr %10, %9 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>, tensor<{M}x{N}xi32, #dot_a>
            %12 = tt.load %11 {{triton_intel_gpu.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %13 = tt.splat %arg1 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %14 = tt.addptr %13, %9 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>, tensor<{M}x{N}xi32, #dot_a>
            tt.store %14, %12 {{boundaryCheck = array<i32: 0, 1>}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>

            // B matrix
            %22 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_b}}>>
            %44 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_b}}>>
            %46 = tt.expand_dims %44 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_b}}>> -> tensor<{M}x1xi32, #dot_b>
            %48 = tt.splat %arg7 : i32 -> tensor<{M}x1xi32, #dot_b>
            %49 = arith.muli %46, %48 : tensor<{M}x1xi32, #dot_b>
            %50 = tt.expand_dims %22 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_b}}>> -> tensor<1x{N}xi32, #dot_b>
            %51 = tt.broadcast %49 : tensor<{M}x1xi32, #dot_b> -> tensor<{M}x{N}xi32, #dot_b>
            %52 = tt.broadcast %50 : tensor<1x{N}xi32, #dot_b> -> tensor<{M}x{N}xi32, #dot_b>
            %53 = arith.addi %51, %52 : tensor<{M}x{N}xi32, #dot_b>

            %54 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %55 = tt.addptr %54, %53 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>, tensor<{M}x{N}xi32, #dot_b>
            %56 = tt.load %55 {{triton_intel_gpu.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %57 = tt.splat %arg3 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %58 = tt.addptr %57, %53 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>, tensor<{M}x{N}xi32, #dot_b>
            tt.store %58, %56 {{boundaryCheck = array<i32: 0, 1>}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>

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

    temp_file = tmp_path / "test_tensor_pointer_block_load.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    if support_block_io:
        # assert '2d block io' in kernel.asm['llir']
        pass

    kernel[(1, 1, 1)](a, x, a.stride(0), a, y, a.stride(0))

    assert torch.equal(a, x) and torch.equal(a, y)
