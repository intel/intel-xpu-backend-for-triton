"""Tests for the `narrow_broadcast_loads` XPUOptions field: a masked row-broadcast bias load in a GEMM epilogue is
loaded once as 1xN and broadcast, with every value, including masked-off ones, unchanged."""

import re

import pytest
import torch

import triton
import triton.language as tl

BM, BK = 64, 32  # BK differs from BM and BN, so the bias tile is the only BM x BN load


@triton.jit
def bias_gemm(a_ptr, b_ptr, bias_ptr, c_ptr, dbg_ptr, second_ptr, M, N, K, grid_n, OTHER: tl.constexpr,
              SECOND_USE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0) // grid_n, tl.program_id(0) % grid_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    bias = tl.load(bias_ptr + rn[None, :], mask=mask, other=OTHER)
    tl.store(dbg_ptr + rm[:, None] * (grid_n * BN) + rn[None, :], bias)  # unmasked: shows other in masked-off lanes
    if SECOND_USE:
        tl.store(second_ptr + rm[:, None] * (grid_n * BN) + rn[None, :], bias, mask=mask)  # a raw use, same mask
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], (acc + bias).to(c_ptr.dtype.element_ty), mask=mask)


def _load_counts(ttgir, bn):
    wide = len(re.findall(rf"tt\.load [^\n]*tensor<{BM}x{bn}x!tt\.ptr", ttgir))
    return wide, len(re.findall(rf"tt\.load [^\n]*tensor<1x{bn}x!tt\.ptr", ttgir))


# shape (M, N, K), extra programs past the grid, other, dtype, second use, threads per warp, BN
CASES = [((256, 512, 128), 0, None, torch.float16, False, 16, 128),
         ((200, 500, 128), 2, 1.5, torch.float16, False, 16, 128),
         ((196, 512, 128), 0, 0.0, torch.float16, False, 16, 128),
         ((200, 500, 128), 2, 1.5, torch.bfloat16, False, 16, 128),
         ((256, 512, 128), 0, 1.5, torch.float16, True, 16, 128),
         ((200, 500, 128), 2, None, torch.float16, True, 16, 128),
         ((196, 512, 128), 2, 0.0, torch.bfloat16, True, 16, 128),
         ((200, 500, 128), 2, 1.5, torch.float16, False, 32, 128),
         ((196, 512, 128), 0, None, torch.float16, True, 32, 128),
         ((200, 512, 128), 0, 1.5, torch.float16, False, 16, 256),  # BN == 16 warps * 16 lanes: the pass must not fire
         ]


@pytest.mark.parametrize("shape, extra, other, dtype, second, tpw, bn", CASES)
def test_narrow_broadcast_loads(shape, extra, other, dtype, second, tpw, bn, device):
    M, N, K = shape
    sizes = triton.runtime.driver.active.utils.get_device_properties(torch.xpu.current_device())["sub_group_sizes"]
    if tpw not in sizes:
        pytest.skip(f"threads_per_warp={tpw} is not in the device's sub-group sizes {sizes}")
    torch.manual_seed(0)
    a = (torch.randn((M, K), device=device) * 0.1).to(dtype)
    b = (torch.randn((K, N), device=device) * 0.1).to(dtype)
    bias = torch.randn((N, ), device=device).to(dtype)
    grid_n = triton.cdiv(N, bn)
    programs = triton.cdiv(M, BM) * grid_n + extra
    dbg_rows = triton.cdiv(programs, grid_n) * BM
    fires = bn == 128
    results = {}
    for narrow in (True, False):
        c = torch.zeros((M, N), dtype=dtype, device=device)
        dbg = torch.full((dbg_rows, grid_n * bn), -9.0, dtype=dtype, device=device)
        sec = torch.full_like(dbg, -7.0)
        kernel = bias_gemm[(programs, )](a, b, bias, c, dbg, sec, M, N, K, grid_n, OTHER=other, SECOND_USE=second,
                                         BM=BM, BN=bn, BK=BK, num_warps=16, warp_size=tpw, sub_32_dpas=tpw == 32,
                                         narrow_broadcast_loads=narrow)
        assert kernel.metadata.threads_per_warp == tpw
        assert _load_counts(kernel.asm["ttir"], bn) == (1, 0)
        wide, narrow_loads = _load_counts(kernel.asm["ttgir"], bn)
        # RemoveLayoutConversions may clone the cheap 1xN load per consumer layout.
        assert (wide == 0 and narrow_loads >= 1) if narrow and fires else (wide, narrow_loads) == (1, 0)
        assert "ttig.dpas" in kernel.asm["ttgir"]
        results[narrow] = (c, dbg, sec)

    ref = (a.float() @ b.float() + bias.float()).to(dtype)
    torch.testing.assert_close(results[True][0], ref, atol=1e-2, rtol=1e-2)
    for on, off in zip(results[True], results[False]):
        assert torch.equal(on, off)  # the rewrite changes no stored value

    # Unmasked debug store: every launched tile holds the bias where the mask is true and `other` (zero if absent)
    # elsewhere; tiles past the grid's last launched program keep the prefill.
    fill = 0.0 if other is None else other
    expect_dbg = torch.full_like(dbg, -9.0)
    for p in range(programs):
        tm, tn = divmod(p, grid_n)
        expect_dbg[tm * BM:(tm + 1) * BM, tn * bn:(tn + 1) * bn] = fill
    expect_dbg[:M, :N] = bias
    assert torch.equal(results[True][1], expect_dbg)
    # Second use: the raw bias under the result's mask, the sentinel everywhere else.
    expect_sec = torch.full_like(sec, -7.0)
    if second:
        expect_sec[:M, :N] = bias
    assert torch.equal(results[True][2], expect_sec)
