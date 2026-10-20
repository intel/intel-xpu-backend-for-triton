"""PrepareIfCombining merges loops that RemoveMasks versions with the same guard when a
reduction and a store sit between them (#8404).

The looped kernels mirror the loop and gap structure Inductor emits for large reductions. Their masked
loads use the canonical `offs < K - k * BLOCK` form with a `tl.cdiv(K, BLOCK)` trip count, and every loop
re-reads its input, since RemoveMasks does not version a store-only loop."""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

pytestmark = pytest.mark.skipif(not is_xpu(), reason="PrepareIfCombining and RemoveMasks are Intel XPU passes")


@triton.jit
def _two_masked_loops(X, Y, Out, K, BLOCK_K: tl.constexpr):
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        acc += tl.load(X + k * BLOCK_K + offs, mask=offs < K - k * BLOCK_K, other=0.0)
    total = tl.sum(acc, axis=0)
    tl.store(Out, total)
    acc2 = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        acc2 += tl.load(Y + k * BLOCK_K + offs, mask=offs < K - k * BLOCK_K, other=0.0) * total
    tl.store(Out + 1, tl.sum(acc2, axis=0))


# The guard is `K % BLOCK_K == 0 && K > BLOCK_K`: 256 takes the unmasked branch, 200 the masked one.
@pytest.mark.parametrize("K", [256, 200])
def test_same_guard_loops_merge(K, device):
    BLOCK_K = 64
    x = torch.randn(K, device=device)
    y = torch.randn(K, device=device)
    out = torch.empty(2, device=device)
    kernel = _two_masked_loops[(1, )](x, y, out, K, BLOCK_K=BLOCK_K)
    ttir = kernel.asm["ttir"]
    assert ttir.count("scf.if") == 1, ttir
    assert ttir.count("scf.for") == 4, ttir
    total = x.sum()
    torch.testing.assert_close(out[0], total, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1], (y * total).sum(), rtol=1e-4, atol=1e-4)


@triton.jit
def _looped_softmax(X, Y, K, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = X + row * K
    acc_max = tl.full([BLOCK], float("-inf"), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK)):
        x = tl.load(base + k * BLOCK + offs, mask=offs < K - k * BLOCK, other=float("-inf"))
        acc_max = tl.maximum(acc_max, x)
    row_max = tl.max(acc_max, axis=0)
    acc_sum = tl.zeros([BLOCK], tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK)):
        # Padded lanes load -inf, so they contribute exp(-inf) = 0.
        x = tl.load(base + k * BLOCK + offs, mask=offs < K - k * BLOCK, other=float("-inf"))
        acc_sum += tl.exp(x - row_max)
    row_sum = tl.sum(acc_sum, axis=0)
    for k in range(0, tl.cdiv(K, BLOCK)):
        mask = offs < K - k * BLOCK
        x = tl.load(base + k * BLOCK + offs, mask=mask, other=float("-inf"))
        tl.store(Y + row * K + k * BLOCK + offs, tl.exp(x - row_max) / row_sum, mask=mask)


@triton.jit
def _looped_layer_norm(X, Y, Mean, Rstd, K, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = X + row * K
    acc = tl.zeros([BLOCK], tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK)):
        acc += tl.load(base + k * BLOCK + offs, mask=offs < K - k * BLOCK, other=0.0)
    mean = tl.sum(acc, axis=0) / K
    tl.store(Mean + row, mean)
    acc_sq = tl.zeros([BLOCK], tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK)):
        mask = offs < K - k * BLOCK
        x = tl.load(base + k * BLOCK + offs, mask=mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)  # padded lanes must not add (0 - mean)^2
        acc_sq += centered * centered
    rstd = 1.0 / tl.sqrt(tl.sum(acc_sq, axis=0) / K + eps)
    tl.store(Rstd + row, rstd)
    for k in range(0, tl.cdiv(K, BLOCK)):
        mask = offs < K - k * BLOCK
        x = tl.load(base + k * BLOCK + offs, mask=mask, other=0.0)
        tl.store(Y + row * K + k * BLOCK + offs, (x - mean) * rstd, mask=mask)


@triton.jit
def _looped_rms_norm(X, Y, Rstd, K, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = X + row * K
    acc = tl.zeros([BLOCK], tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK)):
        x = tl.load(base + k * BLOCK + offs, mask=offs < K - k * BLOCK, other=0.0)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / K + eps)
    tl.store(Rstd + row, rstd)
    for k in range(0, tl.cdiv(K, BLOCK)):
        mask = offs < K - k * BLOCK
        x = tl.load(base + k * BLOCK + offs, mask=mask, other=0.0)
        tl.store(Y + row * K + k * BLOCK + offs, x * rstd, mask=mask)


@triton.jit
def _two_loops_different_bounds(X, Y, Out, K, K2, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK)):
        acc += tl.load(X + k * BLOCK + offs, mask=offs < K - k * BLOCK, other=0.0)
    total = tl.sum(acc, axis=0)
    tl.store(Out, total)
    acc2 = tl.zeros([BLOCK], tl.float32)
    for k in range(0, tl.cdiv(K2, BLOCK)):
        acc2 += tl.load(Y + k * BLOCK + offs, mask=offs < K2 - k * BLOCK, other=0.0) * total
    tl.store(Out + 1, tl.sum(acc2, axis=0))


ROWS = 3
BLOCK = 128


def _assert_merged(kernel, num_loops):
    # These kernels contain no other scf.if, so the raw counts are the versioning counts: one
    # if holding the unmasked and the masked copy of every loop.
    ttir = kernel.asm["ttir"]
    assert ttir.count("scf.if") == 1, ttir
    assert ttir.count("scf.for") == 2 * num_loops, ttir


# 512 is a multiple of BLOCK (the unmasked branch runs), 500 leaves a tail (the masked one runs).
@pytest.mark.parametrize("K", [512, 500])
def test_looped_softmax(K, device):
    x = torch.randn(ROWS, K, device=device)
    y = torch.empty_like(x)
    kernel = _looped_softmax[(ROWS, )](x, y, K, BLOCK=BLOCK)
    _assert_merged(kernel, 3)
    torch.testing.assert_close(y, torch.softmax(x, dim=-1), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("K", [512, 500])
def test_looped_layer_norm(K, device):
    eps = 1e-5
    x = torch.randn(ROWS, K, device=device)
    y = torch.empty_like(x)
    mean = torch.empty(ROWS, device=device)
    rstd = torch.empty(ROWS, device=device)
    kernel = _looped_layer_norm[(ROWS, )](x, y, mean, rstd, K, eps, BLOCK=BLOCK)
    _assert_merged(kernel, 3)
    torch.testing.assert_close(mean, x.mean(dim=-1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rstd, torch.rsqrt(x.var(dim=-1, unbiased=False) + eps), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y, torch.nn.functional.layer_norm(x, (K, ), eps=eps), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("K", [512, 500])
def test_looped_rms_norm(K, device):
    eps = 1e-5
    x = torch.randn(ROWS, K, device=device)
    y = torch.empty_like(x)
    rstd = torch.empty(ROWS, device=device)
    kernel = _looped_rms_norm[(ROWS, )](x, y, rstd, K, eps, BLOCK=BLOCK)
    _assert_merged(kernel, 2)
    ref_rstd = torch.rsqrt((x * x).mean(dim=-1) + eps)
    torch.testing.assert_close(rstd, ref_rstd, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y, x * ref_rstd[:, None], rtol=1e-4, atol=1e-4)


# The two guards test different values (`K` and `K2`), so the ifs must stay apart.
def test_different_guards_do_not_merge(device):
    K, K2 = 256, 192
    x = torch.randn(K, device=device)
    y = torch.randn(K2, device=device)
    out = torch.empty(2, device=device)
    kernel = _two_loops_different_bounds[(1, )](x, y, out, K, K2, BLOCK=64)
    ttir = kernel.asm["ttir"]
    assert ttir.count("scf.if") == 2, ttir
    assert ttir.count("scf.for") == 4, ttir
    total = x.sum()
    torch.testing.assert_close(out[0], total, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1], y.sum() * total, rtol=1e-4, atol=1e-4)
