"""A load masked by nothing but a per-dimension boundary check, with other=0,
is served by the 2D block message's own `base_height`/`base_width` clamps
instead of per-element predication (see ttgi::matchBoundaryCheckMask)."""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _masked_matmul(a_ptr, b_ptr, c_ptr, M, N, K, stride_am: tl.constexpr, stride_bk: tl.constexpr,
                   stride_cm: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _masked_matmul_bt(a_ptr, b_ptr, c_ptr, M, N, K, stride_am: tl.constexpr, stride_bn: tl.constexpr,
                      stride_cm: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # B is stored (N, K), so its tile is read column major: the bound on the
        # memory-contiguous dim is K rather than N.
        b = tl.load(b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Every dimension is ragged, so each tail tile relies on the clamps. K stays a
# multiple of 16 to keep the row pitch 16-byte aligned, and of 2 so the width
# granularity f16 needs is provable. It also stays >= 32, as a narrower K would
# put the transposed B tile's row pitch below the 64-byte hardware minimum.
@pytest.mark.parametrize("M, N, K", [(1022, 1008, 240), (256, 256, 256), (65, 66, 48)])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_boundary_check_mask_matmul(M, N, K, transpose_b, device):
    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((N, K) if transpose_b else (K, N), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    kernel = _masked_matmul_bt if transpose_b else _masked_matmul
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    compiled = kernel[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), BLOCK_M, BLOCK_N, BLOCK_K,
                            num_warps=8)

    ref = a.float() @ (b.float().T if transpose_b else b.float())
    torch.testing.assert_close(c.float(), ref, atol=1e-1, rtol=1e-2)

    # Both operand loads must reach the hardware clamps: a per-element fallback
    # would leave no 2D block read at all.
    llir = compiled.asm["llir"]
    load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
    assert load_count > 0, "masked operand loads did not become 2D block loads"


# A column bound that leaves the surface narrower than the 64-byte minimum the
# message nominally wants. A is a 16-column view of a 64-column buffer, so its
# row pitch (128 bytes) clears the minimum while its width (32 bytes) does not,
# and the clamp still has to zero-fill the tile's remaining 16 columns.
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_boundary_check_mask_narrow_surface(device):
    torch.manual_seed(0)
    M, N, K, K_PAD = 1022, 1008, 16, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    a_full = torch.randn((M, K_PAD), device=device, dtype=torch.float16)
    a = a_full[:, :K]
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    compiled = _masked_matmul[grid](a, b, c, M, N, K, K_PAD, b.stride(0), c.stride(0), BLOCK_M, BLOCK_N, BLOCK_K,
                                    num_warps=8)

    # The padding columns must not reach the accumulator.
    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=1e-1, rtol=1e-2)
    llir = compiled.asm["llir"]
    load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
    assert load_count > 0, "masked operand loads did not become 2D block loads"


@triton.jit
def _halo_load(a_ptr, out_ptr, M, N, stride_am: tl.constexpr, HALO: tl.constexpr, LOWER: tl.constexpr,
               ON_ROW: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    if ON_ROW:
        offs_m = pid * BLOCK_M - HALO + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
    else:
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid * BLOCK_N - HALO + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if LOWER:
        mask = mask & ((offs_m[:, None] >= 0) if ON_ROW else (offs_n[None, :] >= 0))
    x = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_n[None, :], mask=mask, other=0.0)
    # The dot puts the load in a DPAS operand layout, where it can become a 2D
    # block load.
    acc = tl.dot(x, tl.full((BLOCK_N, 16), 1.0, tl.float16))
    om = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(out_ptr + om[:, None] * 16 + tl.arange(0, 16)[None, :], acc)


# An index that starts before the tensor, which the surface clamps read like
# any other: a `>= 0` conjunct must only be dropped where it is vacuous. In the
# column case the tiles also sweep every width the lowering has to legalise:
# full (program 0), 8 valid columns, i.e. under the 64-byte minimum
# (program 1), and none at all (programs 2 and 3).
@pytest.mark.parametrize("on_row", [True, False])
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
def test_boundary_check_mask_negative_index(on_row, lower, device):
    torch.manual_seed(0)
    BLOCK_M, BLOCK_N, HALO = 64, 32, 8
    M, N, PAD = 256, 32, 64
    # Real memory around the tensor, filled with a marker any read of it shows.
    buf = torch.full((M + 2 * PAD, N + 2 * PAD), 100.0, device=device, dtype=torch.float16)
    a = buf[PAD:PAD + M, PAD:PAD + N]
    a.copy_(torch.randn((M, N), device=device, dtype=torch.float16))
    out = torch.empty((M, 16), device=device, dtype=torch.float32)
    compiled = _halo_load[(M // BLOCK_M, )](a, out, M, N, a.stride(0), HALO, lower, on_row, BLOCK_M, BLOCK_N,
                                            num_warps=4, num_stages=1)

    ref = torch.empty_like(out)
    for p in range(M // BLOCK_M):
        rows = torch.arange(BLOCK_M, device=device) + p * BLOCK_M
        cols = torch.arange(BLOCK_N, device=device)
        if on_row:
            rows = rows - HALO
        else:
            cols = cols + p * BLOCK_N - HALO
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        if lower:
            mask &= (rows[:, None] >= 0) if on_row else (cols[None, :] >= 0)
        # Clamped only to keep the gather in range; the clamped elements are
        # masked out.
        tile = buf[(rows + PAD).clamp(0, buf.shape[0] - 1)][:, (cols + PAD).clamp(0, buf.shape[1] - 1)].float()
        ref[p * BLOCK_M:(p + 1) * BLOCK_M] = torch.where(mask, tile, 0.0).sum(1, keepdim=True).expand(-1, 16)
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)

    # Without a lower bound the mask is absorbed whatever the base's sign, so
    # the clamps produced the result above. Only the column case shows it: a
    # column-varying mask that is not absorbed gets no 2D block load at all,
    # while a row-varying one still gets a masked one.
    if not on_row and not lower and triton.runtime.driver.active.get_current_target().arch['has_2d_block_io']:
        llir = compiled.asm["llir"]
        load_count = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
        assert load_count > 0, "boundary-check mask was not absorbed into a 2D block load"
