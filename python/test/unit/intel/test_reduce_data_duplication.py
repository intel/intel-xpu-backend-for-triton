import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def small_m_gemm_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr, A_KM: tl.constexpr):
    # A is a (M, K) row-major matrix; with M == 1 it is equally a (K, M) one.
    pid_n = tl.program_id(0)
    b_desc = tl.make_tensor_descriptor(b_ptr, (K, N), (N, 1), (BK, BN))
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        if A_KM:  # load a [BK, BM] tile, then transpose
            a = tl.load(a_ptr + (k + rk)[:, None] + rm[None, :] * K, mask=(rm < M)[None, :], other=0.0).T
        else:  # load a [BM, BK] tile
            a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :], mask=(rm < M)[:, None], other=0.0)
        acc = tl.dot(a, b_desc.load([k, pid_n * BN]), acc)
    tl.make_tensor_descriptor(c_ptr, (M, N), (N, 1), (BM, BN)).store([0, pid_n * BN], acc)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize("a_km", [True, False])
def test_small_m_gemm_a_tile_not_staged(a_km, device):
    # A small-M A tile loaded in the loop through a tensor of pointers is converted to the DPAS operand layout on
    # every iteration. ReduceDataDuplication must leave that conversion to the generic lowering rather than staging it
    # through a K-contiguous buffer the operand reads one element at a time (#8890).
    if not triton.runtime.driver.active.get_current_target().arch["has_subgroup_matrix_multiply_accumulate"]:
        pytest.skip("Needs DPAS")
    M, N, K = 1, 1024, 4096
    BM, BN, BK = 8, 64, 256
    torch.manual_seed(0)
    a = torch.rand((M, K), device=device, dtype=torch.bfloat16)
    b = torch.rand((K, N), device=device, dtype=torch.bfloat16)
    c = torch.zeros((M, N), device=device, dtype=torch.float32)
    kernel = small_m_gemm_kernel[(N // BN, )](a, b, c, M, N, K, BM, BN, BK, a_km, num_warps=4, num_stages=2,
                                              grf_mode="256")
    torch.testing.assert_close(c, torch.mm(a, b, out_dtype=torch.float32), atol=1e-2, rtol=1e-3)
    assert "ttg.local_alloc" not in kernel.asm["ttgir"]
