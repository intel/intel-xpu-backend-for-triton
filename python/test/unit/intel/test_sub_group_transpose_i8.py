"""Regression tests for issue #8749: 8-bit sub-group transposes through local memory.

A `ttg.convert_layout` whose source holds one 16-byte row per lane (coalesced loads of
16-byte-aligned `uint8` rows) and whose destination holds one column per lane is lowered
as a sub-group transpose through SLM. With 1-byte elements the padded SLM rows were 17
bytes long, so the per-lane 16-byte vector loads were not dword aligned while carrying
LLVM's implied alignment, and lane `l` got its rows shifted by `l % 4`.

The kernel below (the reporter's) hits that conversion on its unpacked fp8 A operand whenever a
warp owns 16 rows of A and the dot runs at sub-group size 16; the other configurations take the
generic shared-memory conversion and act as controls.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

# fp4 (E2M1) code -> value; every product of two of these is exact in fp32.
E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]


@triton.jit
def _e2m1_to_e4m3_bits(code):
    code = code.to(tl.int32)
    mag = code & 7
    bits = tl.where(mag == 0, 0, tl.where(mag == 1, 0x30, (mag + 12) << 2))
    return bits | ((code >> 3) << 7)


@triton.jit
def _packed_dot_kernel(A, B, C, M, N, K, stride_a_m, stride_a_k, stride_b_n, stride_b_k, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, BF16: tl.constexpr):
    # The kernel of issue #8749: A [M, K/2] and B [N, K/2] hold packed E2M1 nibbles that are
    # unpacked to fp8 (e4m3) in registers right before the dot. The unpacked A operand
    # reaches the dot through an 8-bit sub-group transpose.
    num_n = tl.cdiv(N, BN)
    pid_m = tl.program_id(0) // num_n
    pid_n = tl.program_id(0) % num_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK // 2)
    a_ptrs = A + rm[:, None] * stride_a_m + rk[None, :] * stride_a_k
    b_ptrs = B + rn[None, :] * stride_b_n + rk[:, None] * stride_b_k
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        w = tl.load(b_ptrs)
        aq = tl.interleave(_e2m1_to_e4m3_bits(a & 0xF), _e2m1_to_e4m3_bits(a >> 4))
        aq = aq.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        wq = tl.trans(tl.interleave(tl.trans(_e2m1_to_e4m3_bits(w & 0xF)), tl.trans(_e2m1_to_e4m3_bits(w >> 4))))
        wq = wq.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        if BF16:
            acc += tl.dot(aq.to(tl.bfloat16), wq.to(tl.bfloat16))
        else:
            acc += tl.dot(aq, wq)
        a_ptrs += BK // 2 * stride_a_k
        b_ptrs += BK // 2 * stride_b_k
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def _unpack(p):
    """[R, C] packed uint8 -> [R, 2C] fp32 values, low nibble first."""
    table = torch.tensor(E2M1)
    p = p.cpu()
    return torch.stack([table[(p & 0xF).long()], table[(p >> 4).long()]], -1).flatten(-2)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize("BM, num_warps", [(128, 8), (128, 4), (64, 8), (64, 4), (256, 16)])
@pytest.mark.parametrize("bf16", [False, True])
def test_packed_e2m1_fp8_dot(BM, num_warps, bf16, device):
    M, N, K = 256, 128, 256
    BN, BK = 32, 32
    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    c = torch.zeros(M, N, device=device)
    grid = (triton.cdiv(M, BM) * (N // BN), )
    _packed_dot_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), BM, BN, BK, bf16,
                             num_warps=num_warps, num_stages=2)
    ref = _unpack(a) @ _unpack(b).T
    torch.testing.assert_close(c.cpu(), ref, atol=0, rtol=0)
