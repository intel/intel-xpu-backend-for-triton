import pytest
import torch
import pathlib

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@pytest.fixture(autouse=True, params=[False, True], ids=["branch-io", "predicated-io"])
def predicated_io(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_LOAD", "1")
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_STORE", "1")
    yield


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [64, 64], [64, 32], [32, 32], [16, 64], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_load_store(M, N, dtype_str, device, tmp_path: pathlib.Path):
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_load_store(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %stride_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c0_i32 = arith.constant 0 : i32

            %src_desc = tt.make_tensor_descriptor %arg0, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32]
                    : !tt.tensordesc<{M}x{N}x{ty}, #blocked> -> tensor<{M}x{N}x{ty}, #blocked>

            %dst_desc = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32], %data
                                : !tt.tensordesc<{M}x{N}x{ty}, #blocked>, tensor<{M}x{N}x{ty}, #blocked>

            tt.return
        }}
    }}
    """
    torch.manual_seed(42)

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)

    temp_file = tmp_path / "test_tdesc_load_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [64, 64], [64, 32], [32, 32], [16, 64], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_load_zero_padding(M, N, dtype_str, device, tmp_path: pathlib.Path):
    """Load a MxN block through a descriptor whose shape is (M-1)x(N-1).

    The last row and last column are out of bounds and must be zero-padded.
    Input is filled with ones so any zero in the output indicates padding.
    """
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_load_store_pad(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %stride_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %cM_minus1 = arith.constant {M - 1} : i32
            %cN_minus1 = arith.constant {N - 1} : i32
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c0_i32 = arith.constant 0 : i32

            // Source descriptor with shape (M-1)x(N-1) — last row/col out of bounds
            %src_desc = tt.make_tensor_descriptor %arg0, [%cM_minus1, %cN_minus1], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32]
                    : !tt.tensordesc<{M}x{N}x{ty}, #blocked> -> tensor<{M}x{N}x{ty}, #blocked>

            // Destination descriptor with full shape so we can store everything
            %dst_desc = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32], %data
                                : !tt.tensordesc<{M}x{N}x{ty}, #blocked>, tensor<{M}x{N}x{ty}, #blocked>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    x = torch.empty_like(a)

    temp_file = tmp_path / "test_tdesc_load_zero_padding.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)

    # Build expected: ones everywhere except last row and last column are zero-padded
    expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    expected[M - 1, :] = 0
    expected[:, N - 1] = 0

    assert torch.equal(x, expected)


@triton.jit
def grouped_mm_tma_kernel_nonk(
    a_ptr,
    b_ptr,
    c_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_ak,
    stride_am,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Non-K-major: A desc [G,K,M], B desc [G,K,N], MMA: dot(a.T, b)."""
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[G, K, M],
        strides=[stride_ag, stride_ak, stride_am],
        block_shape=[1, BLOCK_K, BLOCK_M],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[G, K, N],
        strides=[stride_bg, stride_bk, stride_bn],
        block_shape=[1, BLOCK_K, BLOCK_N],
    )

    for g in tl.range(G):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = a_desc.load([g, k, 0])
            # tl.device_print(" A1:", a)
            a = a.reshape(BLOCK_K, BLOCK_M)
            # tl.device_print(" A2:", a)
            # a = a_desc.load([g, k, 0])
            # a = tl.permute(a, (0, 2, 1)).reshape(BLOCK_M, BLOCK_K)
            b = b_desc.load([g, k, 0]).reshape(BLOCK_K, BLOCK_N)
            acc += tl.dot(a.T, b)

        c = acc.to(tl.bfloat16)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_base = c_ptr + g * stride_cg
        c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tma_correctness(device):
    torch.manual_seed(42)
    dtype = torch.float32

    # Parameters from the failing test case
    G, M, N, K = 2, 3, 7, 13
    M_padded = 8
    N_padded = 8
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
    grid = (1, )

    # A: created as randn(G, K, M_padded).transpose(-2,-1)[:,:M,:]
    # → shape [G, M, K], stride (K*M_padded, 1, M_padded) - M is innermost
    A2_full = torch.randn(G, K, M_padded, dtype=dtype, device=device)
    A2 = A2_full.transpose(-2, -1)[:, :M, :]  # shape (5,3,13), stride (K*M_padded, 1, M_padded)=(104, 1, 8)

    B2_full = torch.randn(G, K, N_padded, dtype=dtype, device=device)
    B2_view = B2_full.transpose(-2, -1)[:, :N, :]  # shape (5,7,13), stride (104, 1, 8)
    mat_b = B2_view.transpose(-2, -1)  # shape (5,13,7), stride (104, 8, 1)

    # Reference: A2 @ mat_b for each group
    ref2 = torch.bmm(A2, mat_b)

    C2_tma = torch.zeros(G, M, N, dtype=dtype, device=device)
    # Pass strides for [G, K, M] order for A and [G, K, N] order for B
    grouped_mm_tma_kernel_nonk[grid](
        A2,
        mat_b,
        C2_tma,
        G,
        M,
        N,
        K,
        A2.stride(0),
        A2.stride(2),
        A2.stride(1),  # stride_G, stride_K(=8), stride_M(=1)
        mat_b.stride(0),
        mat_b.stride(1),
        mat_b.stride(2),  # stride_G, stride_K(=8), stride_N(=1)
        C2_tma.stride(0),
        C2_tma.stride(1),
        C2_tma.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    torch.testing.assert_close(ref2, C2_tma, atol=1e-5, rtol=1e-5)
