"""End-to-end test for the StageLargeFMADotsViaSLM pass.

The shape `64x128x128 f16` matches the original ARL-S PTSS-overflow report
that motivated PR #7276.  Without the pass, FMA lowering fully unrolls K=128
and exceeds the 256 KB scratch space limit on ARL-S iGPU; with the pass,
operands are staged in SLM (~48 KB) and K is tiled at 32, dropping per-thread
live bytes well below the cliff.

The test does NOT itself enable / disable the pass (it's permanently in the
pipeline once the redesign lands).  It just compiles + runs the shape that
used to fail and verifies correctness against torch.matmul.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu, is_xpu_arl_s


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("M, N, K", [(64, 128, 128)])
def test_arl_s_64x128x128_no_ptss_overflow(M, N, K, device):
    """The ARL-S PTSS-overflow regression.  Pre-redesign this would fail
    IGC compilation with 587 KB scratch space allocated against a 256 KB cap;
    with the SLM-staging pass, K=128 is tiled at 32 and the per-thread live
    set stays bounded.
    """
    if not is_xpu_arl_s():
        pytest.skip("PTSS overflow only reproducible on ARL-S iGPU")

    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    c = torch.empty((M, N), dtype=torch.float16, device=device)

    grid = (1, 1)
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
        num_warps=4,
    )

    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    err = (c.float() - ref.float()).abs().max().item()
    # f16 GEMM tolerance: well below 1e-1 relative for these shapes.
    assert err < 1e-1, f"max abs error {err} exceeds 1e-1"


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("M, N, K, BLOCK_K",
                         [(64, 128, 256, 64),  # outer-loop matmul; inner K tile chosen by autotuner
                          (64, 64, 128, 64),  # smaller dot, pass should still trigger if pressure crosses threshold
                          ])
def test_correctness_outer_loop(M, N, K, BLOCK_K, device):
    """Correctness across multiple shapes — runs on any XPU; the pass is a
    no-op on DPAS-capable hardware (PVC, BMG, ARL-H Xe2)."""
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    c = torch.empty((M, N), dtype=torch.float16, device=device)

    grid = (1, 1)
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    ref = torch.matmul(a.float(), b.float()).to(torch.float16)
    err = (c.float() - ref.float()).abs().max().item()
    assert err < 1e-1, f"max abs error {err} exceeds 1e-1"
