"""`tl.fma` must keep a separately rounded addend rounded.

`tl.fma(d, mul_rn(g, g), x * c)` has to match eager: `x * c` is rounded to f32
first and only the fma rounds once. Lowering `tl.fma` to a contractable
`fmul`/`fadd` pair lets IGC fuse the *addend* multiply instead of ours when the
pair reaches it with the plain multiply first (LLVM canonicalizes the operand
order), so the result is 1 ulp off. The pair is therefore emitted after the
LLVM optimizer, multiply first (`expandFmaToContractPair`).

Reproducer from https://github.com/intel/intel-xpu-backend-for-triton/pull/8753.
"""

import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fma_rounded_addend(x, g, o, XBLOCK: tl.constexpr):
    i = tl.arange(0, XBLOCK)
    m = i < 4
    a = tl.load(x + i, m)
    b = tl.load(g + i, m)
    t2 = a * tl.full([1], 0.1, tl.float32)  # rounded to f32
    t4 = libdevice.mul_rn(b, b)
    d = tl.full([1], 0.9, tl.float32)
    tl.store(o + i, tl.fma(d, t4, t2), m)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
def test_fma_keeps_separately_rounded_addend():
    x = torch.tensor([1.0000001192092896, 0.3, 1e-7, 1e-30], device="xpu")
    g = torch.tensor([0.01, 0.02, 0.001, 0.0001], device="xpu")
    o = torch.empty(4, device="xpu")
    _fma_rounded_addend[(1, )](x, g, o, 1024)

    # f32 reference: round `x * 0.1` and `g * g`, then a single-rounding fma.
    xn, gn = x.cpu().numpy(), g.cpu().numpy()
    addend = (xn * np.float32(0.1)).astype(np.float32)
    prod = (gn * gn).astype(np.float32)
    expected = (prod.astype(np.float64) * np.float64(np.float32(0.9)) + addend.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(o.cpu().numpy(), expected)
