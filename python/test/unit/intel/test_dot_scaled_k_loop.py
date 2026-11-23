"""A GEMM K loop built on a native tt.dot_scaled, with the scales loaded in the loop (issue #8931).

On targets with the block scaled DPAS such a loop ran at 24-42% of the same loop built on tt.dot:
it was not pipelined (no 2D block prefetch of the A and B tiles), and each scale load was converted
from its blocked layout to the scale layout of the DPAS through SLM, between two work-group barriers
per conversion, in every iteration.

The scales are random on purpose: with the usual all-ones scales a scale that reaches the wrong row
or group of the DPAS goes unnoticed.
"""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _scaled_gemm(A, B, A_scale, B_scale, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BN, BK])
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_g = tl.arange(0, BK // 32)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = a_desc.load([pid_m * BM, k])
        b = b_desc.load([pid_n * BN, k])
        a_sc = tl.load(A_scale + offs_m[:, None] * (K // 32) + (k // 32 + offs_g)[None, :])
        b_sc = tl.load(B_scale + offs_n[:, None] * (K // 32) + (k // 32 + offs_g)[None, :])
        acc = tl.dot_scaled(a, a_sc, "e4m3", b.T, b_sc, "e4m3", acc)
    c_desc = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BM, BN])
    c_desc.store([pid_m * BM, pid_n * BN], acc)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("BM, BN, BK, num_warps, num_stages", [(128, 128, 64, 16, 3), (128, 128, 128, 16, 2),
                                                               (256, 256, 64, 32, 3)])
def test_dot_scaled_k_loop(BM, BN, BK, num_warps, num_stages, device):
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch.get("has_subgroup_scaled_matrix_multiply_accumulate", False):
        pytest.skip("target has no block scaled DPAS")

    M = N = 512
    K = 1024
    torch.manual_seed(0)
    a = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device=device).to(torch.float8_e4m3fn)
    a_scale = torch.randint(124, 131, (M, K // 32), device=device, dtype=torch.uint8)
    b_scale = torch.randint(124, 131, (N, K // 32), device=device, dtype=torch.uint8)
    c = torch.zeros(M, N, device=device, dtype=torch.float32)

    kernel = _scaled_gemm[(M // BM, N // BN)](a, b, a_scale, b_scale, c, M, N, K, BM=BM, BN=BN, BK=BK,
                                              num_warps=num_warps, num_stages=num_stages)

    a_ref = a.float() * torch.exp2(a_scale.float() - 127).repeat_interleave(32, dim=1)
    b_ref = b.float() * torch.exp2(b_scale.float() - 127).repeat_interleave(32, dim=1)
    ref = a_ref @ b_ref.T
    assert ((c - ref).norm() / ref.norm()).item() < 1e-5

    ttgir, llir = kernel.asm["ttgir"], kernel.asm["llir"]
    if "ScaledMatrixMultiplyAccumulateINTEL" not in llir:
        pytest.skip("tt.dot_scaled is not lowered to the block scaled DPAS on this target")
    # The K loop is pipelined: the A and B tiles are prefetched.
    assert "ttig.descriptor_prefetch" in ttgir
    # The scales are loaded in the layout the DPAS takes them in: no conversion in the loop, so no
    # round trip through SLM and no barrier.
    loop = ttgir[ttgir.index("scf.for"):]
    loop = loop[:loop.index("scf.yield")]
    assert not re.search(r"ttg\.convert_layout", loop), loop
    assert not re.search(r"call[^\n]*barrier", llir)
