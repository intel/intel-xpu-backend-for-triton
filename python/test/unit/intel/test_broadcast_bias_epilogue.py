"""A GEMM epilogue that adds a broadcast bias (addmm with bias strides (0, 1)) through a full BLOCK_M x BLOCK_N
pointer tensor with repeated addresses. The bias load feeds the DPAS accumulator, so it must not be a layout anchor:
it is loaded in the DPAS encoding and the epilogue needs no layout conversion."""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _addmm_broadcast_bias_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                                 stride_bias_m: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                 BLOCK_K: tl.constexpr, DESC_STORE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, (M, K), (stride_am, 1), (BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(b_ptr, (K, N), (stride_bk, 1), (BLOCK_K, BLOCK_N))
    row = pid_m * BLOCK_M
    col = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for off_k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([row, off_k]), b_desc.load([off_k, col]), acc)

    rm = row + tl.arange(0, BLOCK_M)
    rn = col + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # stride_bias_m is 0: every row of the pointer tensor repeats the same BLOCK_N addresses.
    bias = tl.load(bias_ptr + rm[:, None] * stride_bias_m + rn[None, :], mask=mask, other=0,
                   eviction_policy="evict_last").to(tl.float32)
    result = (acc + bias).to(c_ptr.dtype.element_ty)
    if DESC_STORE:
        c_desc = tl.make_tensor_descriptor(c_ptr, (M, N), (stride_cm, 1), (BLOCK_M, BLOCK_N))
        c_desc.store([row, col], result)
    else:
        tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], result, mask=mask)


@triton.jit
def _addmm_windowed_bias_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_bk, stride_cm, row_lo, row_hi,
                                n_valid, stride_bias_m: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr, HAS_OTHER: tl.constexpr, OTHER: tl.constexpr):
    # The bias rows are masked to the window [row_lo, row_hi), so the messages of one shared broadcast-row load have
    # different predicates (false then true at row_lo, true then false at row_hi), and the whole tile is stored
    # without a mask: every masked-off element must hold `other` (0 when there is none).
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, (M, K), (stride_am, 1), (BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(b_ptr, (K, N), (stride_bk, 1), (BLOCK_K, BLOCK_N))
    row = pid_m * BLOCK_M
    col = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for off_k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([row, off_k]), b_desc.load([off_k, col]), acc)

    rm = row + tl.arange(0, BLOCK_M)
    rn = col + tl.arange(0, BLOCK_N)
    mask = ((rm >= row_lo) & (rm < row_hi))[:, None] & (rn < n_valid)[None, :]
    ptrs = bias_ptr + rm[:, None] * stride_bias_m + rn[None, :]
    if HAS_OTHER:
        bias = tl.load(ptrs, mask=mask, other=OTHER, eviction_policy="evict_last")
    else:
        bias = tl.load(ptrs, mask=mask, eviction_policy="evict_last")
    result = (acc + bias.to(tl.float32)).to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], result)


def _has_dpas_and_2d_block_io():
    # Evaluated at collection time: on other backends the target's arch is not a dict.
    if not is_xpu():
        return False
    arch = triton.runtime.driver.active.get_current_target().arch
    return arch.get('has_2d_block_io', False) and arch.get('has_subgroup_matrix_multiply_accumulate', False)


def _bias_load_feeds_dot_in_dpas(ttgir):
    # The bias is the only tt.load; it must produce the dot's (DPAS) encoding, with no layout conversion left.
    dot_enc = re.search(r"= tt\.dot .* -> tensor<[^,]+, (#\w+)>", ttgir).group(1)
    load_enc = re.search(r"= tt\.load .* : tensor<[^,]+, (#\w+)>", ttgir).group(1)
    assert load_enc == dot_enc, f"bias loaded in {load_enc}, dot result in {dot_enc}"
    assert "ttg.convert_layout" not in ttgir


@pytest.mark.skipif(not _has_dpas_and_2d_block_io(), reason="the bias is loaded in the DPAS encoding")
@pytest.mark.parametrize("desc_store", [False, True])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, num_warps", [
    (64, 64, 32, 4),
    (64, 128, 32, 4),
    (64, 128, 32, 8),
    (128, 128, 32, 8),
    (64, 128, 64, 16),
])
# Full tiles only, partial M and N tiles, and partial N tiles only.
@pytest.mark.parametrize("M, N, K", [(256, 256, 128), (65, 129, 64), (200, 300, 96), (256, 300, 96)])
def test_broadcast_bias_epilogue(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, desc_store, device, with_allocator):
    torch.manual_seed(0)
    # Descriptors need 16-byte aligned row pitches: pad B and C to a multiple of 8 columns.
    n_pitch = triton.cdiv(N, 8) * 8
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, n_pitch), dtype=torch.float16, device=device)[:, :N]
    c = torch.empty((M, n_pitch), dtype=torch.float16, device=device)[:, :N]
    bias = torch.randn((N, ), dtype=torch.float16, device=device).expand(M, N)
    assert bias.stride() == (0, 1)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    kernel = _addmm_broadcast_bias_kernel[grid](a, b, c, bias, M, N, K, a.stride(0), b.stride(0), c.stride(0),
                                                stride_bias_m=bias.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                BLOCK_K=BLOCK_K, DESC_STORE=desc_store, num_warps=num_warps,
                                                num_stages=2)

    torch.testing.assert_close(c.float(), torch.addmm(bias, a, b).float(), atol=5e-2, rtol=1e-2)
    # M and N are runtime values here. With an M that is not a multiple of 16 the row mask changes within a block load
    # message, and with an N that is not one the column mask does too, so the bias is not a row-major 2D block load
    # whose messages share one load: costed by its distinct addresses, it may stay an anchor.
    ttgir = kernel.asm["ttgir"]
    if M % 16 == 0 and N % 16 == 0:
        _bias_load_feeds_dot_in_dpas(ttgir)
    else:
        assert 'ttig.block_io = "row_major"' not in re.search(r"= tt\.load .*", ttgir).group(0)


@pytest.mark.skipif(not _has_dpas_and_2d_block_io(), reason="the bias is loaded in the DPAS encoding")
@pytest.mark.parametrize("other", [None, 0.0, 3.0])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, num_warps", [
    (64, 64, 32, 4),
    (64, 128, 32, 8),
    (128, 128, 32, 8),
    (64, 128, 64, 16),
])
def test_broadcast_bias_masked_rows(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, other, device, with_allocator):
    M, N, K = 256, 256, 64
    # Rows below row_lo and from row_hi on are masked off inside a tile, and so are the last columns. All bounds are
    # multiples of 16: the column bound keeps the bias a 2D block load (an unaligned one makes it a gather), and the
    # row bounds keep the mask constant over each block load message, so the messages that share one broadcast-row
    # load can have different predicates (with other row bounds each message is a single row and nothing is shared).
    # They do where a warp's rows span a bound: in (64, 128, 64, 16) each warp holds 16 contiguous rows, so its
    # messages always share a predicate; that config covers a shared load whose members are all on or all off.
    row_lo, row_hi, n_valid = 16, 208, N - 16
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    c = torch.empty((M, N), dtype=torch.float16, device=device)
    bias_row = torch.randn((N, ), dtype=torch.float16, device=device)
    bias = bias_row.expand(M, N)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    kernel = _addmm_windowed_bias_kernel[grid](a, b, c, bias, M, N, K, a.stride(0), b.stride(0), c.stride(0), row_lo,
                                               row_hi, n_valid, stride_bias_m=bias.stride(0), BLOCK_M=BLOCK_M,
                                               BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, HAS_OTHER=other is not None,
                                               OTHER=0.0 if other is None else other, num_warps=num_warps, num_stages=2)

    rows = torch.arange(M, device=device)[:, None]
    cols = torch.arange(N, device=device)[None, :]
    valid = (rows >= row_lo) & (rows < row_hi) & (cols < n_valid)
    expected_bias = torch.where(valid, bias.float(), 0.0 if other is None else other)
    ref = (a.float() @ b.float() + expected_bias).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=5e-2, rtol=1e-2)
    ttgir = kernel.asm["ttgir"]
    _bias_load_feeds_dot_in_dpas(ttgir)
    # A 2D block load: in the DPAS encoding its messages that differ only in their rows share one load.
    assert 'ttig.block_io = "row_major"' in re.search(r"= tt\.load .*", ttgir).group(0)
