"""End-to-end tests for merging a host-side TensorDescriptor with a device-side one.

A `tt.make_tensor_descriptor` that shares a value with an untraceable descriptor
falls back to pointer form. The merge point decides how the fallback is reached:
an `arith.select` exposes both sides at one op, while an `scf.if` spreads them
over the two `scf.yield`s, which the per-op grouping used to miss.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.tools.tensor_descriptor import TensorDescriptor

M = N = 64
BM = BN = 32


@triton.jit
def _merge_kernel(host_desc, b_ptr, out_ptr, cond_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr,
                  MAKE_IN_BRANCH: tl.constexpr):
    # Loaded from memory so the `if` stays a runtime branch.
    cond = tl.load(cond_ptr) != 0
    if MAKE_IN_BRANCH:
        # Survives canonicalization as an scf.if, so the merge goes through a yield.
        if cond:
            d = host_desc
            x = tl.zeros((BM, BN), dtype=tl.float16)
        else:
            d = tl.make_tensor_descriptor(b_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
            x = d.load([0, 0])
    else:
        # Canonicalized to an arith.select.
        local = tl.make_tensor_descriptor(b_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
        x = local.load([0, 0])
        if cond:
            d = host_desc
        else:
            d = local
    y = d.load([0, 0])
    offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(out_ptr + offs, x + 2 * y)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific descriptor rewrite")
@pytest.mark.parametrize("make_in_branch", [False, True], ids=["select", "scf_if"])
def test_merge_with_host_descriptor(make_in_branch, device, with_allocator):
    a = torch.randn((M, N), device=device, dtype=torch.float16)
    b = torch.randn((M, N), device=device, dtype=torch.float16)
    host_desc = TensorDescriptor.from_tensor(a, block_shape=[BM, BN])
    a_tile, b_tile = a[:BM, :BN], b[:BM, :BN]
    for cond in (1, 0):
        out = torch.empty((BM, BN), device=device, dtype=torch.float16)
        cond_t = torch.full((1, ), cond, device=device, dtype=torch.int32)
        _merge_kernel[(1, )](host_desc, b, out, cond_t, M, N, BM, BN, make_in_branch)
        if cond:
            expected = 2 * a_tile if make_in_branch else b_tile + 2 * a_tile
        else:
            expected = 3 * b_tile
        torch.testing.assert_close(out, expected)


@triton.jit(noinline=True)
def _load_tile(desc, off):
    return desc.load([off, 0])


@triton.jit
def _noinline_kernel(b_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    d = tl.make_tensor_descriptor(b_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    x = d.load([0, 0])
    y = _load_tile(d, BM)
    offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(out_ptr + offs, x + 2 * y)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific descriptor rewrite")
def test_descriptor_passed_to_noinline(device, with_allocator):
    b = torch.randn((M, N), device=device, dtype=torch.float16)
    out = torch.empty((BM, BN), device=device, dtype=torch.float16)
    _noinline_kernel[(1, )](b, out, M, N, BM, BN)
    torch.testing.assert_close(out, b[:BM, :BN] + 2 * b[BM:2 * BM, :BN])
