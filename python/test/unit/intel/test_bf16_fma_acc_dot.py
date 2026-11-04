"""FMA-path dots with a bf16 accumulator accumulate in fp32 and round once.

On the FMA path, a bf16 accumulator used to be rounded after every
multiply-add, and IGC lowers a bf16 multiply-add to a `mad` with a bf16
destination, which rounds toward zero
(intel/intel-xpu-backend-for-triton#8466). Such dots now accumulate in fp32
and round the result to bf16 once: they must match, bit for bit, the same
kernel written with an fp32 accumulator and an explicit cast to bf16.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _fma_dot_bf16_acc(a_ptr, b_ptr, c_ptr, ieee_ptr, N: tl.constexpr, K: tl.constexpr, BLOCK_K: tl.constexpr,
                      BF16_ACC: tl.constexpr):
    offs = tl.arange(0, N)
    rk = tl.arange(0, BLOCK_K)
    c_ptrs = c_ptr + offs[:, None] * N + offs[None, :]
    acc = tl.load(c_ptrs)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * N + offs[None, :])
        if BF16_ACC:
            acc = tl.dot(a, b, acc)
        else:
            acc = tl.dot(a, b, acc.to(tl.float32)).to(tl.bfloat16)
    tl.store(c_ptrs, acc)
    # An IEEE f32 dot cannot use DPAS, and it demotes every dot in the function
    # to the FMA path.
    x = (offs[:, None] + offs[None, :]).to(tl.float32)
    tl.store(ieee_ptr + offs[:, None] * N + offs[None, :], tl.dot(x, x, input_precision="ieee"))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_dtype, b_dtype", [(torch.bfloat16, torch.bfloat16), (torch.float8_e5m2, torch.float8_e5m2),
                                              (torch.float8_e4m3fn, torch.float8_e4m3fn),
                                              (torch.float8_e4m3fn, torch.float8_e5m2)])
def test_fma_dot_bf16_acc(a_dtype, b_dtype, device):
    N, K, BLOCK_K = 64, 128, 32
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = (torch.randn((N, K), generator=generator) / 2).to(a_dtype)
    b = (torch.randn((K, N), generator=generator) / 2).to(b_dtype)
    c = torch.randn((N, N), generator=generator).to(torch.bfloat16)

    def run(bf16_acc):
        out = c.to(device)
        ieee_out = torch.empty((N, N), dtype=torch.float32, device=device)
        kernel = _fma_dot_bf16_acc[(1, )](a.to(device), b.to(device), out, ieee_out, N, K, BLOCK_K, BF16_ACC=bf16_acc)
        return out, kernel

    out, kernel = run(bf16_acc=True)
    assert "#ttig.dpas" not in kernel.asm["ttgir"], "kernel no longer exercises the FMA path"
    ref_out, _ = run(bf16_acc=False)
    assert torch.equal(out, ref_out), "bf16 accumulator result differs from the fp32 accumulator kernel"

    expected = c.float()
    for k in range(0, K, BLOCK_K):
        expected = (expected + a[:, k:k + BLOCK_K].float() @ b[k:k + BLOCK_K, :].float()).to(torch.bfloat16).float()
    torch.testing.assert_close(out.float().cpu(), expected, atol=0.1, rtol=2e-2)
