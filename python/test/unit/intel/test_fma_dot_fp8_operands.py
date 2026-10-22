"""FMA-path dots with fp8 operands widen them before the layout conversion to the dot operand layout.

The widened operands must match, bit for bit, the same kernel written with explicit casts of the loaded fp8 values,
and fp32-accumulator results must match an fp64 reference. The 128x128 tile with 8 warps used to give wrong results
for e4m3 x e5m2 -> fp32 when the operands were widened in the dot operand layout.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _fma_dot_fp8(a_ptr, b_ptr, c_ptr, ieee_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, SOURCE_CAST: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    acc = tl.load(c_ptrs)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :])
        if SOURCE_CAST:
            # What the compiler does: a 16-bit float accumulator widens fp8 to fp16, and a bf16 accumulator
            # accumulates in fp32 and rounds the result once.
            if c_ptr.dtype.element_ty == tl.float16:
                acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)
            else:
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc.to(tl.float32),
                             input_precision="ieee").to(c_ptr.dtype.element_ty)
        else:
            acc = tl.dot(a, b, acc)
    tl.store(c_ptrs, acc)
    # An IEEE f32 dot cannot use DPAS, and it demotes every dot in the function to the FMA path.
    r = tl.arange(0, 16)
    x = (r[:, None] + r[None, :]).to(tl.float32)
    tl.store(ieee_ptr + r[:, None] * 16 + r[None, :], tl.dot(x, x, input_precision="ieee"))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_dtype, b_dtype", [
    (torch.float8_e5m2, torch.float8_e5m2),
    (torch.float8_e4m3fn, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float8_e5m2),
    (torch.float8_e5m2, torch.float8_e4m3fn),
])
@pytest.mark.parametrize("acc_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("BM, BN, num_warps", [(128, 128, 8), (64, 64, 4)])
def test_fma_dot_fp8_operands(a_dtype, b_dtype, acc_dtype, BM, BN, num_warps, device):
    M = N = K = 512
    BK = 32
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = (torch.randn((M, K), generator=generator) / 4).to(a_dtype).to(device)
    b = (torch.randn((K, N), generator=generator) / 4).to(b_dtype).to(device)
    ieee = torch.empty((16, 16), dtype=torch.float32, device=device)

    def run(source_cast):
        out = torch.zeros((M, N), dtype=acc_dtype, device=device)
        kernel = _fma_dot_fp8[(M // BM, N // BN)](a, b, out, ieee, M, N, K, BM, BN, BK, SOURCE_CAST=source_cast,
                                                  num_warps=num_warps)
        return out, kernel

    out, kernel = run(source_cast=False)
    assert "#ttig.dpas" not in kernel.asm["ttgir"], "kernel no longer exercises the FMA path"
    ref_out, _ = run(source_cast=True)
    assert torch.equal(out, ref_out), "compiler-widened fp8 operands differ from explicit casts"
    if acc_dtype == torch.float32:
        expected = (a.double() @ b.double()).float()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
