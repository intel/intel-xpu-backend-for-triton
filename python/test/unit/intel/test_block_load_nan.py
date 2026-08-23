import pytest
import torch

import pathlib

import triton
from triton._internal_testing import is_xpu, is_xpu_pvc


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
    # Col N-1 and row M-1 are the OOB padding slots; use a sentinel (99.0)
    # so a leaked padding value is unmistakably distinct from in-bounds 1.0.
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    a[:, N - 1] = 99.0  # OOB col padding slot — must not appear in output
    b = torch.ones((N, M), dtype=torch_dtype, device=device)
    b[:, M - 1] = 99.0  # OOB col padding slot for B
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


@pytest.mark.parametrize("M, N", [[256, 64], [128, 16], [64, 64], [32, 32]])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_dpas_layout_pitch_rounding_pad_zero(M, N, device, tmp_path: pathlib.Path):
    """Same as test_block_load_dpas_layout_pitch_rounding but with PAD_ZERO.

    Without the fix: hardware internally rounds base_width up, treating the OOB
    column as in-bounds and loading from memory (1.0) instead of zero-filling.
    With fix: OOB column correctly becomes 0.0 via software zero mask.
    """
    A_width, B_width = 1, 2
    layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
    num_warps = 32
    ty = "f16"
    torch_dtype = torch.float16

    ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_pitch_rounding_pad_zero(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32
            %Mload_i32 = arith.constant {M-1} : i32
            %Nload_i32 = arith.constant {N-1} : i32
            %M_i32 = arith.constant {M} : i32
            %N_i32 = arith.constant {N} : i32
            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A: shape=(M-1, N-1), stride=(N, 1) — pitch=N*2 >= rounded. padding=1 (PAD_ZERO)
            %1 = tt.make_tensor_descriptor %arg0, [%Mload_i32, %Nload_i32], [%N_i64, %c1_i64] {{padding = 1 : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %2 = tt.descriptor_load %1[%0, %c0_i32] {{ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}} : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>> -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %3 = tt.make_tensor_descriptor %arg1, [%M_i32, %N_i32], [%N_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            tt.descriptor_store %3[%0, %c0_i32], %2 : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}> >, tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            // B: shape=(N-1, M-1), stride=(M, 1). padding=1 (PAD_ZERO)
            %4 = tt.make_tensor_descriptor %arg2, [%Nload_i32, %Mload_i32], [%M_i64, %c1_i64] {{padding = 1 : i32}} : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %5 = tt.descriptor_load %4[%c0_i32, %0] {{ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}} : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>> -> tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %6 = tt.make_tensor_descriptor %arg3, [%N_i32, %M_i32], [%M_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            tt.descriptor_store %6[%c0_i32, %0], %5 : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}> >, tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            tt.return
        }}
    }}
    """

    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    a[:, N - 1] = 99.0  # sentinel: must NOT appear in output (should be 0.0)
    b = torch.ones((N, M), dtype=torch_dtype, device=device)
    b[:, M - 1] = 99.0  # sentinel for B
    x = torch.empty_like(a)
    y = torch.empty_like(b)

    temp_file = tmp_path / "test_block_load_pitch_rounding_pad_zero.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))
    kernel[(1, 1, 1)](a, x, b, y)

    torch.set_printoptions(profile="full", precision=2, sci_mode=0, linewidth=200)

    # In-bounds: 1.0. OOB row (M-1) and OOB col (N-1): 0.0 (PAD_ZERO fill).
    # Without fix: OOB col = 99.0 (sentinel leaked — hardware treated it as in-bounds).
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = 0.0  # OOB row -> zero fill
    x_expected[:, N - 1:] = 0.0  # OOB col -> zero fill (NOT 99.0 sentinel)

    y_expected = torch.ones((N, M), dtype=torch_dtype, device=device)
    y_expected[N - 1:, :] = 0.0  # OOB row
    y_expected[:, M - 1:] = 0.0  # OOB col

    assert torch.allclose(x_expected, x) and torch.allclose(
        y_expected, y), (f"PAD_ZERO col bound bug: col {N-1} = {x[:2, N-1].tolist()} "
                         f"(expected 0.0; sentinel 99.0 leaked — OOB element not zero-filled)")


@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_nan_mask_col_bound(device, tmp_path: pathlib.Path):
    """Reproducer for NaN mask column bound bug in pitch-aware rounding.

    When base_width is rounded up (e.g., K=31 fp16, pitch=32 elements → pitch=64
    bytes ≥ rounded(62)=64), the NaN mask must use the ORIGINAL inner shape (31)
    as its column bound, NOT the rounded base_width / elemBytes (32). Otherwise
    col N-1=31 is considered in-bounds and gets 99.0 (sentinel leaked from the
    OOB padding slot) instead of NaN.

    Key design choices:
    - Outer dimension (M-1) is a RUNTIME function arg: prevents getFoldedConstantValue
      from folding it, bypassing the overly-conservative scalar fallback check that
      fires when the outer shape is a compile-time odd constant.
    - N=32 gives pitch=64 bytes (the 2D block load minimum), allowing the 2D block
      load path to be taken.
    - Single-warp layout (warpsPerCTA=[1,1]) with K=32 (two K-reps): lane i covers
      cols i and i+16. Lane 15 covers col 31 = N-1 (the OOB boundary col).
    """
    M, N = 8, 32  # pitch = N*2 = 64 bytes (minimum for 2D block load)
    ty = "f16"
    torch_dtype = torch.float16
    # Two K-reps (K=32 = 2 * K_tile=16) in single warp; lane 15 -> cols 15 and 31.
    layouts = ("#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, "
               "opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>")
    num_warps = 1
    ir = (layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion,
                        ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io,
                        ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32,
                        "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu",
                        "ttg.threads-per-warp" = 16 : i32}} {{
        // %arg2 = M-1 (runtime): prevents scalar fallback from the outer-shape check.
        tt.func public @nan_mask_col_bound_test(
            %arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}},
            %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}},
            %arg2: i32) attributes {{noinline = false}} {{
            %c31_i32 = arith.constant {N - 1} : i32   // inner shape = 31 (foldable)
            %c8_i32  = arith.constant {M}     : i32
            %c32_i32 = arith.constant {N}     : i32
            %c32_i64 = arith.constant {N}     : i64   // stride = 32; pitch = 64 bytes
            %c1_i64  = arith.constant 1       : i64
            %c0_i32  = arith.constant 0       : i32
            // shape=[%arg2, 31], stride=[32, 1]
            // pitch=64 >= rounded(31*2=62)=64 -> pitch-aware rounding applies
            // %arg2 (M-1=7) is runtime -> outer-shape check cannot fold -> passes
            %desc = tt.make_tensor_descriptor %arg0, [%arg2, %c31_i32], [%c32_i64, %c1_i64]
                      {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty},
                      #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            %loaded = tt.descriptor_load %desc[%c0_i32, %c0_i32]
                        {{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}}
                      : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
                     -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            // Output uses full [M, N] shape so ALL elements (incl. row M-1, col N-1) are written.
            %out_desc = tt.make_tensor_descriptor %arg1, [%c8_i32, %c32_i32], [%c32_i64, %c1_i64]
                          : <{ty}>, !tt.tensordesc<{M}x{N}x{ty},
                          #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %loaded
                      : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>,
                        tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            tt.return
        }}
    }}
    """)
    # Col N-1 is the OOB padding slot — sentinel 99.0 distinguishes a leaked value from NaN.
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    a[:, N - 1] = 99.0
    x = torch.empty((M, N), dtype=torch_dtype, device=device)
    temp_file = tmp_path / "test_nan_mask_col_bound.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))
    kernel[(1, 1, 1)](a, x, M - 1)  # pass runtime M-1
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = float('nan')  # OOB row
    x_expected[:, N - 1:] = float('nan')  # OOB col (was 99.0 in input)
    torch.set_printoptions(profile="full", precision=2, sci_mode=False, linewidth=200)
    assert torch.allclose(
        x_expected, x, equal_nan=True), (f"NaN mask col bound bug: col {N - 1} = {x[:, N - 1].tolist()} "
                                         f"(expected NaN; mask used rounded bound 32 instead of original innerSh 31)")


@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
@pytest.mark.xfail(
    is_xpu_pvc(), reason="Issue 2: PVC applies OOB checks at i32 granularity. With a runtime column "
    "count the pitch check in LowerTo2DBlockLoad is bypassed, so a contiguous odd-K surface still "
    "has its last valid column zeroed by the coarse check. Needs a runtime pitch branch to fix.")
def test_block_load_nan_runtime_k_contiguous(device, tmp_path: pathlib.Path):
    """Test PAD_NAN with runtime K and a contiguous tensor (pitch = K*2 bytes).

    The descriptor column count is a runtime function argument, so
    getFoldedConstantValue returns null → the !descColCount branch is taken →
    the static pitch check is bypassed → ttig.2d_block_load is always emitted.

    In LoadStoreOpToLLVM, hwBaseWidth = roundUp(K*2, 4). For K=15 (contiguous,
    pitch=30): hwBaseWidth=32 > pitch=30, violating the hardware constraint.
    The NaN mask (built from the original K=15 boundary) correctly NaN-fills
    col 15 regardless.

    On PVC Max 1100 WITHOUT the hwBaseWidth rounding fix (hwBaseWidth=30=pitch):
      - Coarse i32 OOB check: last i32 word (bytes 28-31) is OOB → zeroes
        BOTH col 14 (valid!) and col 15 (OOB) → col 14 = 0.0 (WRONG).
    On PVC Max 1100 WITH the fix (hwBaseWidth=32): all 16 cols in hardware
      in-bounds, NaN mask handles col 15, col 14 = 1.0 (correct).
    On BMG (fine-grained OOB): fine-grained check handles each element
      individually — col 14 is always correct regardless of fix.

    This test PASSES on BMG (fine-grained OOB: only col 15 zeroed, then
    NaN-masked). On PVC Max 1100 (coarse i32 OOB), col 14 is zeroed by the
    hardware coarse check even after the fix (min(32,30)=30 → bytes 28-31
    still OOB at 4-byte granularity). This is Issue 2 — a known gap for
    contiguous runtime odd-K on Max 1100 — and is expected to fail there.
    """
    M, N = 8, 16  # single DPAS tile: [repeatCount=8, K_tile=16], one warp
    ty = "f16"
    torch_dtype = torch.float16
    layouts = ("#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, "
               "opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>")
    num_warps = 1
    ir = (layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion,
                        ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io,
                        ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32,
                        "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu",
                        "ttg.threads-per-warp" = 16 : i32}} {{
        // %arg2 = M-1 (runtime): bypasses outer-shape check.
        // %arg3 = K   (runtime): getFoldedConstantValue returns null →
        //                         !descColCount → continue → no pitch check.
        tt.func public @runtime_k_contiguous(
            %arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}},
            %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}},
            %arg2: i32,
            %arg3: i32) attributes {{noinline = false}} {{
            %k_i64   = arith.extsi %arg3 : i32 to i64  // stride = K (contiguous)
            %c8_i32  = arith.constant {M}     : i32
            %c16_i32 = arith.constant {N}     : i32
            %c16_i64 = arith.constant {N}     : i64
            %c1_i64  = arith.constant 1       : i64
            %c0_i32  = arith.constant 0       : i32

            // shape=[M-1, K], stride=[K, 1] — CONTIGUOUS pitch = K*2 bytes.
            // K=%arg3 is runtime → !descColCount → continue (no pitch check).
            // For K=15: pitch=30 < rounded(30)=32 → would be scalar if K were const.
            %desc = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%k_i64, %c1_i64]
                      {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty},
                      #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            %loaded = tt.descriptor_load %desc[%c0_i32, %c0_i32]
                        {{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}}
                      : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
                     -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            // Output uses full [M, N] so all elements including OOB ones are written.
            %out_desc = tt.make_tensor_descriptor %arg1, [%c8_i32, %c16_i32], [%c16_i64, %c1_i64]
                          : <{ty}>, !tt.tensordesc<{M}x{N}x{ty},
                          #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %loaded
                      : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>,
                        tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            tt.return
        }}
    }}
    """)
    K = 15  # odd → contiguous pitch = 30 bytes < rounded = 32 bytes
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    a[:, N - 1] = 99.0  # sentinel in OOB col — must NOT appear in output
    x = torch.empty((M, N), dtype=torch_dtype, device=device)
    temp_file = tmp_path / "test_runtime_k_contiguous.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))
    kernel[(1, 1, 1)](a, x, M - 1, K)  # pass both M-1 and K as runtime args

    # rows 0..M-2, cols 0..K-2: in-bounds → 1.0
    # col K-1 (col 14): last VALID column — must be 1.0, NOT 0.0.
    #   On Max 1100 without fix: base_width=K*2=30, coarse i32 OOB zeroes the
    #   last i32 word (bytes 28-31) → col 14 AND col 15 both zeroed → col 14 = 0.0.
    # col K (col 15): OOB → NaN (PAD_NAN)
    # row M-1: OOB row → NaN
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = float('nan')  # OOB row
    x_expected[:, K:] = float('nan')  # OOB cols (K..N-1)
    torch.set_printoptions(profile="full", precision=2, sci_mode=False, linewidth=200)
    assert torch.allclose(x_expected, x,
                          equal_nan=True), (f"Runtime K contiguous bug: col {K - 1} = {x[:, K - 1].tolist()} "
                                            f"(expected 1.0; on Max 1100 without fix: 0.0 due to coarse i32 OOB)")
