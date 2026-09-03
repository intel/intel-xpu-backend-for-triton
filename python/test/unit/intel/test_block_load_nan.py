import pytest
import torch

import pathlib

import triton
from triton._internal_testing import is_xpu, is_xpu_pvc


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [128, 8], [64, 64], [64, 32], [32, 32], [16, 64]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16"])
@pytest.mark.parametrize("padding_id, expected_oob", [(2, float('nan')), (1, 0.0)], ids=["pad_nan", "pad_zero"])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_dpas_layout(M, N, dtype_str, padding_id, expected_oob, device, tmp_path: pathlib.Path):
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
            %1 = tt.make_tensor_descriptor %arg0, [%Mload_i32, %Nload_i32], [%Nload_i64, %c1_i64] {{padding = {padding_id} : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %2 = tt.descriptor_load %1[%0, %c0_i32] {{ttig.block_io = "row_major", ttig.desc_padding = {padding_id} : i32}} : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>> -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %3 = tt.make_tensor_descriptor %arg1, [%M_i32, %N_i32], [%N_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            tt.descriptor_store %3[%0, %c0_i32], %2 : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}> >, tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            // B matrix
            %4 = tt.make_tensor_descriptor %arg2, [%Nload_i32, %Mload_i32], [%Mload_i64, %c1_i64] {{padding = {padding_id} : i32}} : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %5 = tt.descriptor_load %4[%c0_i32, %0] {{ttig.block_io = {block_io}, ttig.desc_padding = {padding_id} : i32}} : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>> -> tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
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

    # Build expected output explicitly: 1.0 for in-bounds elements, NaN for OOB
    # (PAD_NAN) or 0.0 (PAD_ZERO). The descriptor has shape (M-1, N-1) for A and
    # (N-1, M-1) for B, loaded into full (M, N) and (N, M) tiles respectively.
    # This avoids depending on the zero-padding reference path, which is unreliable
    # for packed fp16 (kWidth=2) operands where hardware boundary behaviour varies.
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = expected_oob  # OOB row
    x_expected[:, N - 1:] = expected_oob  # OOB col

    y_expected = torch.ones((N, M), dtype=torch_dtype, device=device)
    y_expected[N - 1:, :] = expected_oob  # OOB row
    y_expected[:, M - 1:] = expected_oob  # OOB col

    assert torch.allclose(x_expected, x, equal_nan=True) and torch.allclose(y_expected, y, equal_nan=True)


@pytest.mark.parametrize("M, N", [[256, 64], [128, 16], [64, 64], [32, 32]])
@pytest.mark.parametrize(
    "dtype_str, padding_id, expected_oob",
    [
        ("float16", 2, float('nan')),
        ("float16", 1, 0.0),
        # PAD_NAN excluded for int16: there is no NaN encoding for integers
        ("int16", 1, 0.0),
    ],
    ids=["float16-pad_nan", "float16-pad_zero", "int16-pad_zero"],
)
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_dpas_layout_pitch_rounding(M, N, dtype_str, padding_id, expected_oob, device,
                                               tmp_path: pathlib.Path):
    """Test pitch-aware base_width rounding, for PAD_NAN and PAD_ZERO, float16 and int16."""
    # 64 extra elements of row stride -> >=128 bytes of pitch headroom for the
    # 2-byte element types tested here, comfortably clearing the 65-byte
    # minPitch margin for any M, N in this parametrization.
    PITCH_PAD = 64
    A_width, B_width = 1, 2
    layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
    num_warps = 32
    ty = {"float16": "f16", "int16": "i16"}[dtype_str]
    torch_dtype = getattr(torch, dtype_str)

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
            %NPitch_i64 = arith.constant {N + PITCH_PAD} : i64
            %MPitch_i64 = arith.constant {M + PITCH_PAD} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix: shape=(M-1, N-1), stride=(N+PAD, 1) -> pitch=(N+PAD)*2 >= minPitch.
            // Output descriptor stays contiguous (stride=N): the store never
            // requests padding, so it has no pitch-headroom requirement.
            %1 = tt.make_tensor_descriptor %arg0, [%Mload_i32, %Nload_i32], [%NPitch_i64, %c1_i64] {{padding = {padding_id} : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %2 = tt.descriptor_load %1[%0, %c0_i32] {{ttig.block_io = "row_major", ttig.desc_padding = {padding_id} : i32}} : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>> -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            %3 = tt.make_tensor_descriptor %arg1, [%M_i32, %N_i32], [%N_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>
            tt.descriptor_store %3[%0, %c0_i32], %2 : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}> >, tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            // B matrix: shape=(N-1, M-1), stride=(M+PAD, 1) -> pitch=(M+PAD)*2 >= minPitch
            %4 = tt.make_tensor_descriptor %arg2, [%Nload_i32, %Mload_i32], [%MPitch_i64, %c1_i64] {{padding = {padding_id} : i32}} : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %5 = tt.descriptor_load %4[%c0_i32, %0] {{ttig.block_io = "row_major", ttig.desc_padding = {padding_id} : i32}} : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>> -> tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            %6 = tt.make_tensor_descriptor %arg3, [%N_i32, %M_i32], [%M_i64, %c1_i64] : <{ty}>, !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>
            tt.descriptor_store %6[%c0_i32, %0], %5 : !tt.tensordesc<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}> >, tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>

            tt.return
        }}
    }}
    """

    # A/B are allocated PITCH_PAD columns wider than their logical (M,N)/(N,M)
    # shape so the descriptor's widened stride still addresses real, owned
    # memory for every row -- the load's own tile width stays exactly N/M
    # (base_width rounds up to exactly the full tile for these even N/M
    # values), so the extra PITCH_PAD columns are never read, only skipped
    # over between rows. Outputs (x, y) stay the original, unpadded shape.
    a = torch.ones((M, N + PITCH_PAD), dtype=torch_dtype, device=device)
    a[:, N - 1] = 99.0  # OOB col padding slot — must not appear in output
    b = torch.ones((N, M + PITCH_PAD), dtype=torch_dtype, device=device)
    b[:, M - 1] = 99.0  # OOB col padding slot for B
    x = torch.empty((M, N), dtype=torch_dtype, device=device)
    y = torch.empty((N, M), dtype=torch_dtype, device=device)

    temp_file = tmp_path / "test_block_load_pitch_rounding.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))
    kernel[(1, 1, 1)](a, x, b, y)

    torch.set_printoptions(profile="full", precision=2, sci_mode=0, linewidth=200)

    # In-bounds: 1.0. OOB row/col: NaN (PAD_NAN) or 0.0 (PAD_ZERO).
    # Without the fix: OOB col leaks the 99.0 sentinel (hardware treated it as
    # in-bounds after base_width rounding, before the mask restored the fill).
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = expected_oob  # OOB row
    x_expected[:, N - 1:] = expected_oob  # OOB col

    y_expected = torch.ones((N, M), dtype=torch_dtype, device=device)
    y_expected[N - 1:, :] = expected_oob  # OOB row
    y_expected[:, M - 1:] = expected_oob  # OOB col

    assert torch.allclose(x_expected, x, equal_nan=True) and torch.allclose(
        y_expected, y, equal_nan=True), (f"padding_id={padding_id}: col {N-1} = {x[:2, N-1].tolist()} "
                                         f"(expected {expected_oob}; sentinel 99.0 leaked — OOB element not filled)")


@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_load_nan_mask_col_bound(device, tmp_path: pathlib.Path):
    """Reproducer for NaN mask column bound bug in pitch-aware rounding.

    When base_width is rounded up (e.g., K=31 fp16, pitch=96 elements ->
    pitch=192 bytes >= minPitch=rounded(62)+65=129), the NaN mask must use the
    ORIGINAL inner shape (31) as its column bound, NOT the rounded base_width
    / elemBytes (32). Otherwise col N-1=31 is considered in-bounds and gets
    99.0 (sentinel leaked from the OOB padding slot) instead of NaN.
    """
    M, N = 8, 32
    PITCH_PAD = 64  # stride = N + PITCH_PAD -> pitch = 96*2 = 192 bytes >= minPitch=129
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
            %c32_i64 = arith.constant {N}     : i64   // output stride: contiguous, no padding requested
            %cPitch_i64 = arith.constant {N + PITCH_PAD} : i64   // input stride: real pitch headroom
            %c1_i64  = arith.constant 1       : i64
            %c0_i32  = arith.constant 0       : i32
            // shape=[%arg2, 31], stride=[96, 1]
            // pitch=192 >= minPitch=rounded(31*2=62)+65=129 -> pitch-aware rounding applies
            // %arg2 (M-1=7) is runtime -> outer-shape check cannot fold -> passes
            %desc = tt.make_tensor_descriptor %arg0, [%arg2, %c31_i32], [%cPitch_i64, %c1_i64]
                      {{padding = 2 : i32}} : <{ty}>, !tt.tensordesc<{M}x{N}x{ty},
                      #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            %loaded = tt.descriptor_load %desc[%c0_i32, %c0_i32]
                        {{ttig.block_io = "row_major", ttig.desc_padding = 2 : i32}}
                      : !tt.tensordesc<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
                     -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 1}}>>
            // Output uses full [M, N] shape, contiguous stride, so ALL elements
            // (incl. row M-1, col N-1) are written; the store never requests
            // padding, so it has no pitch-headroom requirement of its own.
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
    # `a` is allocated PITCH_PAD columns wider than N so the descriptor's
    # widened stride still addresses real, owned memory (see
    # test_block_load_dpas_layout_pitch_rounding for the full rationale).
    a = torch.ones((M, N + PITCH_PAD), dtype=torch_dtype, device=device)
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
    is_xpu_pvc(), reason="PVC applies OOB checks at i32 granularity. With a runtime column "
    "count the pitch check in LowerTo2DBlockLoad is bypassed, so a contiguous odd-K surface still "
    "has its last valid column zeroed by the coarse check. Needs a runtime pitch branch to fix.")
def test_block_load_nan_runtime_k_contiguous(device, tmp_path: pathlib.Path):
    """Test PAD_NAN with runtime K and a contiguous tensor (pitch = K*2 bytes). """
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
    # col K-1 (col 14): last VALID column → 1.0, NOT 0.0.
    # col K (col 15): OOB → NaN (PAD_NAN)
    # row M-1: OOB row → NaN
    x_expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    x_expected[M - 1:, :] = float('nan')  # OOB row
    x_expected[:, K:] = float('nan')  # OOB cols (K..N-1)
    torch.set_printoptions(profile="full", precision=2, sci_mode=False, linewidth=200)
    assert torch.allclose(x_expected, x,
                          equal_nan=True), (f"Runtime K contiguous bug: col {K - 1} = {x[:, K - 1].tolist()} "
                                            f"(expected 1.0)")
