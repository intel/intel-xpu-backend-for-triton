"""Regression tests for the Intel scan lowering of blocked layouts.

TritonIntelGPUToLLVM/ScanOpToLLVM.cpp keeps core's scan lowering from before
triton-lang/triton#12184, including the fix from triton-lang/triton#11990 in
`BlockedScanLayout::get`. The reland of #12184 removed the regression tests of
that fix from python/test/gluon/test_lowerings.py; these are its one-CTA cases.
"""
import pytest
import torch

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


@gluon.jit
def _combine(a, b):
    return a + b


@gluon.jit
def scan_kernel(x_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr, axis: ttgl.constexpr):
    x_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
    x_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
    x = ttgl.load(x_ptr + x_offs_m * N + x_offs_n)
    y = ttgl.associative_scan(x, axis=axis, combine_fn=_combine)
    ttgl.store(z_ptr + x_offs_m * N + x_offs_n, y)


@pytest.mark.parametrize("M, N, src_layout, axis", [
    pytest.param(2, 2, ttgl.BlockedLayout([2, 1], [THREADS_PER_WARP, 1], [4, 1], [1, 0]), 0, id="registers"),
    pytest.param(2, 4 * THREADS_PER_WARP, ttgl.BlockedLayout([1, 2], [1, THREADS_PER_WARP], [1, 4], [0, 1]), 1,
                 id="multiwarp"),
])
def test_scan_blocked_shape_clipping(M, N, src_layout, axis, device):
    x = torch.arange(1, M * N + 1, dtype=torch.int32, device=device).reshape(M, N)
    y = torch.empty_like(x)
    scan_kernel[(1, )](x, y, M, N, src_layout, axis, num_warps=4)
    torch.testing.assert_close(y, x.cumsum(axis, dtype=torch.int32))
