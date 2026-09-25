"""End-to-end coverage for INT8 dots that do not use DPAS (issue #7854).

`DPASAnalysis::canUseDPAS` is function-wide: one dot that DPAS cannot handle --
an `input_precision="ieee"` f32 dot, for instance -- demotes *every* dot in the
same function to the FMA path, including an INT8 dot of a perfectly DPAS-able
shape. That INT8 dot used to be silently wrong: the FMA lowering emitted a
`mul`/`add` chain, and IGC folded it back into a `dp4a` whose two operands it
had reordered independently, computing sum_j a[permA(j)] * b[permB(j)].

There was no INT8 coverage of the FMA path before this test; the only
end-to-end FMA dot test is f16 (`test_stage_large_fma_dots_via_slm.py`).
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _int8_dot_with_ieee_dot(a_ptr, b_ptr, c_ptr, ieee_ptr, BLOCK: tl.constexpr):
    """The INT8 dot is DPAS-able on its own; the IEEE dot below is what pushes
    the whole function onto FMA."""
    offs = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs[:, None] * BLOCK + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * BLOCK + offs[None, :])
    c_ptrs = c_ptr + offs[:, None] * BLOCK + offs[None, :]
    tl.store(c_ptrs, tl.dot(a, b, acc=tl.load(c_ptrs), out_dtype=tl.int32))

    x = (offs[:, None] + offs[None, :]).to(tl.float32)
    tl.store(ieee_ptr + offs[:, None] * BLOCK + offs[None, :], tl.dot(x, x, input_precision="ieee"))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
# K < 32 is rejected outright for INT8 by the `min_dot_size` guard in
# `third_party/intel/backend/compiler.py`, so 32 is the smallest usable block.
@pytest.mark.parametrize("BLOCK", [32, 64])
def test_int8_dot_demoted_to_fma(BLOCK, device):
    generator = torch.Generator(device="cpu").manual_seed(17)
    a = torch.randint(-4, 5, (BLOCK, BLOCK), dtype=torch.int8, generator=generator)
    b = torch.randint(-4, 5, (BLOCK, BLOCK), dtype=torch.int8, generator=generator)
    c = torch.randint(-16, 17, (BLOCK, BLOCK), dtype=torch.int32, generator=generator)
    expected = a.to(torch.int32) @ b.to(torch.int32) + c

    acc = c.to(device)
    ieee_out = torch.empty((BLOCK, BLOCK), dtype=torch.float32, device=device)
    kernel = _int8_dot_with_ieee_dot[(1, )](a.to(device), b.to(device), acc, ieee_out, BLOCK=BLOCK, num_warps=4)

    # The point of the test is the FMA path, so fail loudly if a pipeline change
    # makes this shape use DPAS instead and the coverage silently disappears.
    assert "#ttig.dpas" not in kernel.asm["ttgir"], "kernel no longer exercises the FMA path"
    torch.testing.assert_close(acc.cpu(), expected)
