import pytest
import torch

import pathlib

import triton
from triton._internal_testing import is_xpu


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [128, 8], [64, 64], [64, 32], [32, 32], [16, 64]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16"])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_dpas_layout(M, N, dtype_str, device, tmp_path: pathlib.Path):
    # modify the layouts to ensure the correct OCL/SPIRV intrinsic is called for each datatype
    if dtype_str == "float32":
        A_width = 1
        B_width = 1
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
        num_warps = 32
    else:
        A_width = 1
        B_width = 2
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
        num_warps = 32

    block_io = "\"row_major\""

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_dpas_layout(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32

            %Mload_i32 = arith.constant {M-1} : i32
            %Nload_i32 = arith.constant {N-1} : i32
            %Mload_i64 = arith.constant {M-1} : i64
            %Nload_i64 = arith.constant {N-1} : i64

            %M_i32 = arith.constant {M} : i32
            %N_i32 = arith.constant {N} : i32
            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix
            %1 = tt.make_tensor_descriptor %arg0, [%Mload_i32, %Nload_i32], [%Nload_i64, %c1_i64] {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %2 = tt.descriptor_load %1[%0, %c0_i32] {{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}} : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>> -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %3 = tt.make_tensor_descriptor %arg1, [%M_i32, %N_i32], [%N_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            tt.descriptor_store %3[%0, %c0_i32], %2 : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}> >, tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            // B matrix
            %4 = tt.make_tensor_descriptor %arg2, [%Nload_i32, %Mload_i32], [%Mload_i64, %c1_i64] {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %5 = tt.descriptor_load %4[%c0_i32, %0] {{ttig.block_io = {block_io}, ttig.desc_padding = 2 : i32}} : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>> -> tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %6 = tt.make_tensor_descriptor %arg3, [%N_i32, %M_i32], [%M_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            tt.descriptor_store %6[%c0_i32, %0], %5 : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}> >, tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)

    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    b = torch.ones((N, M), dtype=torch_dtype, device=device)
    x = torch.empty_like(a)
    y = torch.empty_like(b)

    temp_file = tmp_path / "test_block_load_dpas_layout.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x, b, y)

    torch.set_printoptions(profile="full", precision=2, sci_mode=0, linewidth=200)

    # Build expected output explicitly: 1.0 for in-bounds elements, NaN for OOB.
    # The descriptor has shape (M-1, N-1) for A and (N-1, M-1) for B, loaded into
    # full (M, N) and (N, M) tiles respectively.
    # This avoids depending on the zero-padding reference path, which is unreliable
    # for packed fp16 (kWidth=2) operands where hardware boundary behaviour varies.
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = float('nan')  # OOB row
    x_expected[:, N - 1:] = float('nan')  # OOB col

    y_expected = torch.ones((N, M), dtype=torch_dtype, device=device)
    y_expected[N - 1:, :] = float('nan')  # OOB row
    y_expected[:, M - 1:] = float('nan')  # OOB col

    assert torch.allclose(x_expected, x, equal_nan=True) and torch.allclose(y_expected, y, equal_nan=True)


@pytest.mark.parametrize("M, N", [[256, 64], [128, 16], [64, 64], [32, 32]])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_dpas_layout_pitch_rounding(M, N, device, tmp_path: pathlib.Path):
    """Test NaN-padded fp16 loads where pitch >= rounded base_width.

    Uses stride=[N, 1] (full tensor stride) so pitch = N*2 bytes.
    Shape is declared as (M-1, N-1), giving base_width = (N-1)*2 bytes.
    When (N-1)*2 is not 4-byte aligned, the pass rounds it up to ceil(…/4)*4.
    Since pitch = N*2 >= rounded, the 2D block load is used (not scalar).
    Verifies that in-bounds elements (cols 0..N-2) read as 1.0 and the OOB
    element (col N-1) reads as NaN — confirming correctness on Max 1100 where
    coarse-grained i32 OOB zeroing is only avoided with the rounded base_width.
    """
    A_width, B_width = 1, 2
    layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
    num_warps = 32
    ty = "f16"
    torch_dtype = torch.float16

    ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_pitch_rounding(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32

            %Mload_i32 = arith.constant {M-1} : i32
            %Nload_i32 = arith.constant {N-1} : i32
            %Mload_i64 = arith.constant {M-1} : i64
            %Nload_i64 = arith.constant {N-1} : i64

            %M_i32 = arith.constant {M} : i32
            %N_i32 = arith.constant {N} : i32
            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix: shape=(M-1, N-1), stride=(N, 1)  →  pitch=N*2 >= rounded
            %1 = tt.make_tensor_descriptor %arg0, [%Mload_i32, %Nload_i32], [%N_i64, %c1_i64] {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %2 = tt.descriptor_load %1[%0, %c0_i32] {{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}} : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>> -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %3 = tt.make_tensor_descriptor %arg1, [%M_i32, %N_i32], [%N_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            tt.descriptor_store %3[%0, %c0_i32], %2 : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}> >, tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            // B matrix: shape=(N-1, M-1), stride=(M, 1)  →  pitch=M*2 >= rounded
            %4 = tt.make_tensor_descriptor %arg2, [%Nload_i32, %Mload_i32], [%M_i64, %c1_i64] {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %5 = tt.descriptor_load %4[%c0_i32, %0] {{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}} : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>> -> tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %6 = tt.make_tensor_descriptor %arg3, [%N_i32, %M_i32], [%M_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            tt.descriptor_store %6[%c0_i32, %0], %5 : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}> >, tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>

            tt.return
        }}
    }}
    """

    # Input tensors: shape [M, N] with actual stride [N, 1].
    # The descriptor declares (M-1, N-1) with the FULL stride — so pitch = N*2.
    # When (N-1)*2 % 4 != 0 (e.g. N=16: 15*2=30, 30%4=2), the pass rounds
    # base_width up to 32 bytes.  pitch=32 >= 32 → 2D block load is used.
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    b = torch.ones((N, M), dtype=torch_dtype, device=device)
    x = torch.empty_like(a)
    y = torch.empty_like(b)

    temp_file = tmp_path / "test_block_load_pitch_rounding.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))
    kernel[(1, 1, 1)](a, x, b, y)

    torch.set_printoptions(profile="full", precision=2, sci_mode=0, linewidth=200)

    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = float('nan')  # OOB row
    x_expected[:, N - 1:] = float('nan')  # OOB col

    y_expected = torch.ones((N, M), dtype=torch_dtype, device=device)
    y_expected[N - 1:, :] = float('nan')  # OOB row
    y_expected[:, M - 1:] = float('nan')  # OOB col

    assert torch.allclose(x_expected, x, equal_nan=True) and torch.allclose(y_expected, y, equal_nan=True)
