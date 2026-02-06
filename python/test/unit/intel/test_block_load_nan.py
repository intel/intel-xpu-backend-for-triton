import pytest
import torch

import pathlib

import triton
from triton._internal_testing import is_xpu


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [128, 8], [64, 64], [64, 32], [32, 32], [16, 64]])
@pytest.mark.parametrize("dtype_str", ["float32"])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_dpas_layout(M, N, dtype_str, transpose, device, tmp_path: pathlib.Path):
    # modify the layouts to ensure the correct OCL/SPIRV intrinsic is called for each datatype
    if dtype_str == "int8":
        A_width = 2
        B_width = 4
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 2]}>"
        num_warps = 4
    elif dtype_str == "float32":
        A_width = 1
        B_width = 1
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
        num_warps = 32
    else:
        A_width = 1
        B_width = 2
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
        num_warps = 32

    block_io = "\"column_major\"" if transpose else "\"row_major\""

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ref_ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_dpas_layout(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32

            %Mload_i64 = arith.constant {M-1} : i64
            %Nload_i64 = arith.constant {N-1} : i64

            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix
            %1 = tt.make_tensor_ptr %arg0, [%Mload_i64, %Nload_i64], [%Nload_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %2 = tt.load %1 {{padding = 1 : i32, boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %3 = tt.make_tensor_ptr %arg1, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            tt.store %3, %2 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>

            // B matrix
            %4 = tt.make_tensor_ptr %arg2, [%Nload_i64, %Mload_i64], {"[%c1_i64, %Nload_i64]" if transpose else "[%Mload_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %5 = tt.load %4 {{padding = 1 : i32, boundaryCheck = array<i32: 0, 1>, ttig.block_io = {block_io} }} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %6 = tt.make_tensor_ptr %arg3, [%N_i64, %M_i64], {"[%c1_i64, %N_i64]" if transpose else "[%M_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            tt.store %6, %5 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>

            tt.return
        }}
    }}
    """

    ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_dpas_layout(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32

            %Mload_i64 = arith.constant {M-1} : i64
            %Nload_i64 = arith.constant {N-1} : i64

            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix
            %1 = tt.make_tensor_ptr %arg0, [%Mload_i64, %Nload_i64], [%Nload_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %2 = tt.load %1 {{padding = 2 : i32, boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %3 = tt.make_tensor_ptr %arg1, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            tt.store %3, %2 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>

            // B matrix
            %4 = tt.make_tensor_ptr %arg2, [%Nload_i64, %Mload_i64], {"[%c1_i64, %Nload_i64]" if transpose else "[%Mload_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %5 = tt.load %4 {{padding = 2 : i32, boundaryCheck = array<i32: 0, 1>, ttig.block_io = {block_io} }} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %6 = tt.make_tensor_ptr %arg3, [%N_i64, %M_i64], {"[%c1_i64, %N_i64]" if transpose else "[%M_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            tt.store %6, %5 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)

    a_ref = torch.ones((M, N), dtype=torch_dtype, device=device)
    b_ref = torch.ones((N, M), dtype=torch_dtype, device=device)
    x_ref = torch.empty_like(a_ref)
    y_ref = torch.empty_like(b_ref.T if transpose else b_ref)

    temp_file = tmp_path / "test_block_load_dpas_layout_ref.ttgir"
    temp_file.write_text(ref_ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a_ref, x_ref, b_ref, y_ref)

    x_ref[x_ref == 0] = float('nan')
    y_ref[y_ref == 0] = float('nan')

    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    b = torch.ones((N, M), dtype=torch_dtype, device=device)
    x = torch.empty_like(a)
    y = torch.empty_like(b.T if transpose else b)

    temp_file = tmp_path / "test_block_load_dpas_layout.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x, b, y)

    torch.set_printoptions(profile="full", precision=2, sci_mode=0, linewidth=200)
    print(y_ref.T if transpose else y_ref)
    print(y.T if transpose else y)

    assert torch.allclose(x_ref, x, equal_nan=True) and torch.allclose(y_ref.T if transpose else y_ref,
                                                                       y.T if transpose else y, equal_nan=True)
