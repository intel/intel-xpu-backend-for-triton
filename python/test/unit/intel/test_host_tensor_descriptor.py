"""End-to-end tests for host-side TensorDescriptor on Intel XPU backend.

Verifies that host-side TensorDescriptor objects (created on the host and passed
as kernel arguments) reach the efficient 2D block I/O hardware path, producing
the same results and codegen as device-side tl.make_tensor_descriptor, and that
collapsing a rank-3 descriptor load into a rank-2 one keeps the bounds the
rank-3 form had (device-side descriptors included).
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.tools.tensor_descriptor import TensorDescriptor


def _has_2d_block_io():
    """Check if current device supports 2D block I/O."""
    return triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)


@triton.jit
def _matmul_kernel(a_desc_or_ptr, b_desc_or_ptr, c_ptr, M, N, K, stride_am: tl.constexpr, stride_ak: tl.constexpr,
                   stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Simple matmul kernel that works with both host and device descriptors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if isinstance(a_desc_or_ptr, tl.tensor_descriptor):
        a_desc = a_desc_or_ptr
    else:
        a_desc = tl.make_tensor_descriptor(a_desc_or_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                                           block_shape=[BLOCK_M, BLOCK_K])
    if isinstance(b_desc_or_ptr, tl.tensor_descriptor):
        b_desc = b_desc_or_ptr
    else:
        b_desc = tl.make_tensor_descriptor(b_desc_or_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                           block_shape=[BLOCK_K, BLOCK_N])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k])
        b = b_desc.load([k, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@pytest.mark.parametrize("M, N, K", [[128, 128, 64], [64, 64, 32]])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_host_descriptor_matmul_2d_block_io(M, N, K, dtype, device):
    """Host-side TensorDescriptor matmul produces correct results and 2D block loads."""
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)

    # Device-side path.
    c_device = torch.empty((M, N), dtype=torch.float32, device=device)
    grid = (M // BLOCK_M, N // BLOCK_N)
    kernel_device = _matmul_kernel[grid](
        a,
        b,
        c_device,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_device.stride(0),
        c_device.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Host-side path.
    c_host = torch.empty((M, N), dtype=torch.float32, device=device)
    a_desc = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    kernel_host = _matmul_kernel[grid](
        a_desc,
        b_desc,
        c_host,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_host.stride(0),
        c_host.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Correctness: both paths match reference.
    ref = (a.to(torch.float32) @ b.to(torch.float32))
    torch.testing.assert_close(c_device, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c_host, ref, rtol=1e-2, atol=1e-2)

    # Codegen: both paths must generate the same number of 2D block loads.
    device_llir = kernel_device.asm["llir"]
    host_llir = kernel_host.asm["llir"]
    device_loads = device_llir.count('spirv_Subgroup2DBlockLoad') + device_llir.count('GenISA.LSC2DBlockRead')
    host_loads = host_llir.count('spirv_Subgroup2DBlockLoad') + host_llir.count('GenISA.LSC2DBlockRead')
    assert device_loads > 0, "device-side path: no 2D block loads found"
    assert host_loads == device_loads, \
        f"host has {host_loads} 2D block loads, expected {device_loads} (same as device)"


@triton.jit
def _one_head_tile_kernel(a_desc, b_ptr, c_ptr, head, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    a = a_desc.load([0, head, 0]).reshape(BLOCK_M, BLOCK_D)
    offs_n = tl.arange(0, N)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_D)[:, None] * N + offs_n[None, :])
    tl.store(c_ptr + tl.arange(0, BLOCK_M)[:, None] * N + offs_n[None, :], tl.dot(a, b))


# One-head tile of a (TOKENS, HEADS, HEAD_DIM) tensor. The (BLOCK_M, 1, BLOCK_D) block
# spans dimensions 0 and 2, so 2D block I/O needs the unit middle dimension collapsed.
# Collapsing is only correct while the tile fits one head: past that it would read the
# next head instead of zero-padding (issues/7679, issues/7464).
@pytest.mark.parametrize("HEAD_DIM, collapse", [(128, True), (104, False)])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_host_descriptor_rank3_unit_middle_dim(HEAD_DIM, collapse, device):
    TOKENS, HEADS, N = 256, 4, 64
    BLOCK_M, BLOCK_D, HEAD = 64, 128, 1

    torch.manual_seed(42)
    a = torch.randn((TOKENS, HEADS, HEAD_DIM), dtype=torch.float16, device=device)
    b = torch.randn((BLOCK_D, N), dtype=torch.float16, device=device)
    c = torch.empty((BLOCK_M, N), dtype=torch.float32, device=device)

    desc = TensorDescriptor(a, list(a.shape), list(a.stride()), [BLOCK_M, 1, BLOCK_D])
    kernel = _one_head_tile_kernel[(1, )](desc, b, c, HEAD, N, BLOCK_M, BLOCK_D)

    tile = torch.zeros((BLOCK_M, BLOCK_D), dtype=torch.float16, device=device)
    tile[:, :HEAD_DIM] = a[:BLOCK_M, HEAD, :]
    torch.testing.assert_close(c, tile.to(torch.float32) @ b.to(torch.float32), rtol=1e-2, atol=1e-2)

    # Check the descriptor rank, not `tt.reshape`: a rank-reducing descriptor_load also
    # leaves no reshape behind, yet nothing was collapsed.
    ttir = kernel.asm["ttir"]
    assert (f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" in ttir) != collapse
    if collapse:
        llir = kernel.asm["llir"]
        assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _identity(BLOCK: tl.constexpr):
    """Identity matrix, built in registers so it adds no 2D-eligible load."""
    i = tl.arange(0, BLOCK)
    return (i[:, None] == i[None, :]).to(tl.float16)


@triton.jit
def _outer_tile_kernel(a_ptr, out_ptr, row_off, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
                       BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[B, R, C], strides=[R * C, C, 1], block_shape=[1, BLOCK_R, BLOCK_C])
    a = desc.load([0, row_off, 0]).reshape(BLOCK_R, BLOCK_C)
    # The dot is what makes the reshape a fusion candidate; the identity operand
    # keeps the result equal to the loaded tile.
    acc = tl.dot(a, _identity(BLOCK_C))
    offs = tl.arange(0, BLOCK_R)[:, None] * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tl.store(out_ptr + offs, acc)


# Unit *outermost* dimension: a (B, R, C) tile whose rows run past R must pad,
# not read the next slice. Fusion bounds the merged dimension only, so the
# merged extent has to be this load's own (issues/8001).
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_rank3_unit_outer_dim_pads(device, with_allocator):
    B, R, C, BLOCK_R, BLOCK_C, ROW_OFF = 2, 32, 64, 32, 64, 16

    torch.manual_seed(42)
    a = torch.randn((B, R, C), dtype=torch.float16, device=device)
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _outer_tile_kernel[(1, )](a, out, ROW_OFF, B, R, C, BLOCK_R, BLOCK_C)

    tile = torch.zeros((BLOCK_R, BLOCK_C), dtype=torch.float16, device=device)
    tile[:R - ROW_OFF] = a[0, ROW_OFF:, :]
    torch.testing.assert_close(out, tile.to(torch.float32))

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _head_cols_kernel(a_desc, out_ptr, head, col_off, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    a = a_desc.load([0, head, col_off]).reshape(BLOCK_M, BLOCK_D)
    acc = tl.dot(a, _identity(BLOCK_D))
    offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptr + offs, acc)


# Unit *middle* dimension, columns running past the end of one head: those
# columns must pad rather than read the next head. Complements
# test_host_descriptor_rank3_unit_middle_dim, which loads a head from column 0.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_host_descriptor_rank3_unit_middle_dim_pads_columns(device):
    TOKENS, HEADS, HEAD_DIM = 256, 4, 128
    BLOCK_M, BLOCK_D, HEAD, COL_OFF = 64, 64, 1, 96

    torch.manual_seed(42)
    a = torch.randn((TOKENS, HEADS, HEAD_DIM), dtype=torch.float16, device=device)
    out = torch.full((BLOCK_M, BLOCK_D), -1.0, dtype=torch.float32, device=device)
    desc = TensorDescriptor(a, list(a.shape), list(a.stride()), [BLOCK_M, 1, BLOCK_D])
    kernel = _head_cols_kernel[(1, )](desc, out, HEAD, COL_OFF, BLOCK_M, BLOCK_D)

    tile = torch.zeros((BLOCK_M, BLOCK_D), dtype=torch.float16, device=device)
    tile[:, :HEAD_DIM - COL_OFF] = a[:BLOCK_M, HEAD, COL_OFF:]
    torch.testing.assert_close(out, tile.to(torch.float32))

    assert f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _neg_stride_kernel(a_ptr, out_ptr, batch, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr, BLOCK_R: tl.constexpr,
                       BLOCK_C: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[B, R, C], strides=[-(R * C), C, 1],
                                     block_shape=[1, BLOCK_R, BLOCK_C])
    a = desc.load([batch, 0, 0]).reshape(BLOCK_R, BLOCK_C)
    acc = tl.dot(a, _identity(BLOCK_C))
    offs = tl.arange(0, BLOCK_R)[:, None] * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tl.store(out_ptr + offs, acc)


# A negative outermost stride must not fuse: the collapse divides the strides,
# and doing that unsigned made a negative stride look like a huge positive
# ratio. The base sits at the last slice so the negative stride walks backwards
# inside the tensor rather than before the allocation.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_negative_outer_stride_declines(device, with_allocator):
    B, R, C, BLOCK_R, BLOCK_C, BATCH = 4, 32, 64, 32, 64, 1

    torch.manual_seed(42)
    a = torch.randn((B, R, C), dtype=torch.float16, device=device)
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _neg_stride_kernel[(1, )](a[B - 1], out, BATCH, B, R, C, BLOCK_R, BLOCK_C)

    torch.testing.assert_close(out, a[B - 1 - BATCH, :BLOCK_R, :].to(torch.float32))
    # Declined, so the rank-3 descriptor survives. Check the descriptor rank and
    # not `tt.reshape`: a rank-reducing descriptor_load leaves no reshape either.
    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" in kernel.asm["ttir"], "should not fuse"


@triton.jit
def _runtime_empty_kernel(a_ptr, out_ptr, heads, col_off, TOKENS: tl.constexpr, HEAD_DIM: tl.constexpr,
                          S0: tl.constexpr, S1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[TOKENS, heads, HEAD_DIM], strides=[S0, S1, 1],
                                     block_shape=[BLOCK_M, 1, BLOCK_D])
    a = desc.load([0, 0, col_off]).reshape(BLOCK_M, BLOCK_D)
    acc = tl.dot(a, _identity(BLOCK_D))
    offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptr + offs, acc)


# A collapsed dimension that is empty only at runtime. The extent is forced to
# zero, which pads on the generic path. Device-side because a host
# TensorDescriptor asserts every shape positive and cannot express it.
#
# `S1 < HEAD_DIM` makes the heads overlap, which nothing rejects. That is what
# makes this test bite: for a non-overlapping descriptor the old extent was
# already <= 0 here, so only the overlapping form distinguishes the two.
#
# `col_off` is a runtime stride-one index that cannot be proven divisible, so
# fusion still happens while `block_io` is withheld and the load stays on the
# generic path. Both are asserted: an unfused rank-3 load pads on an empty
# dimension by itself, and the 2D path encodes a zero extent as `0 - 1`.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_empty_collapsed_dim_pads(device, with_allocator):
    TOKENS, HEAD_DIM, S0, S1, BLOCK_M, BLOCK_D, COL_OFF = 64, 128, 512, 64, 64, 64, 3

    torch.manual_seed(42)
    # Shifted away from zero so padding cannot be mistaken for real data.
    a = torch.randn(TOKENS * S0, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_M, BLOCK_D), -1.0, dtype=torch.float32, device=device)
    kernel = _runtime_empty_kernel[(1, )](a, out, 0, COL_OFF, TOKENS, HEAD_DIM, S0, S1, BLOCK_M, BLOCK_D)

    torch.testing.assert_close(out, torch.zeros_like(out))

    assert f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') == 0, \
        "must stay on the generic path: the 2D path encodes a zero extent as 0 - 1"
