# The reduction kernel below is kept byte-for-byte identical to the inductor
# reproducer, which assigns several intermediates that are never read. Suppress
# ruff's unused-variable check file-wide so the kernel stays unchanged.
# ruff: noqa: F841
"""Regression test for the CRI ``ocloc`` segfault on a persistent reduction.

This reuses the minimal ``ocloc`` crash reproducer verbatim: a pure
PyTorch + Triton kernel with all ``torch._inductor`` machinery stripped
out. The kernel is the persistent reduction inductor generates for::

    logical_not(any(logical_not(x)))   ==   all(x)

over a bool tensor of shape ``[4, 129]`` (516 elements flattened).

The ``triton_helpers.any`` reduction is inlined here as ``_any`` so the
emitted Triton IR stays as close as possible to the inductor-generated
kernel that crashed ``ocloc``.

Compiling this kernel on the CRI simulator crashed ``ocloc`` with a
segmentation fault (exit code ``-11``). This test guards against that
regression by compiling and launching the kernel and checking the result
matches ``torch.all``. It runs wherever an Intel XPU device is available
(the crash manifests on the CRI ``ocloc`` path) and passes harmlessly on
other architectures.

Reproducer: https://github.com/user-attachments/files/29886027/repro_ocloc_crash.py
Issue: https://github.com/intel/intel-xpu-backend-for-triton/issues/7542
"""

import pytest
import torch

import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)


@triton.jit
def _any_combine(a, b):
    return a | b


@triton.jit
def _any(a, dim):
    return tl.reduce(a, dim, _any_combine)


@triton.jit
def triton_per_fused_all_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1
    r0_numel = 516
    R0_BLOCK: tl.constexpr = 1024
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.int1)
    tmp1 = tmp0 == 0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, False)
    tmp5 = _any(tmp4, 1)[:, None].to(tl.int1)
    tmp6 = tmp5 == 0
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp6, None)


def test_ocloc_reduction_crash(device):
    """Compiling this persistent reduction crashed ``ocloc`` (SIGSEGV, exit -11)."""
    x = torch.rand((4, 129), device=device) > 0.5  # bool tensor [4, 129] -> 516 elems
    out = torch.empty((), device=device, dtype=torch.bool)

    xnumel = 1
    r0_numel = 516
    grid = (1, )

    triton_per_fused_all_0[grid](out, x, xnumel, r0_numel, XBLOCK=1)
    torch.xpu.synchronize()

    expected = torch.all(x)
    assert out.item() == expected.item(), f"mismatch: triton={out.item()} expected={expected.item()}"
