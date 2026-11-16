"""End-to-end tests for host-side TensorDescriptor on Intel XPU backend.

Verifies that host-side TensorDescriptor objects (created on the host and passed
as kernel arguments) reach the efficient 2D block I/O hardware path, producing
the same results and codegen as device-side tl.make_tensor_descriptor, that
collapsing a rank-3 descriptor load into a rank-2 one keeps the bounds the
rank-3 form had (device-side descriptors included), and that the collapse still
happens when elementwise ops or a permute sit between the load and its reshape.
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
# The collapse rebases the descriptor onto the head, so a tile wider than the head
# (104 < 128) pads the rest of its columns instead of reading the next head
# (issues/7679, issues/7464).
@pytest.mark.parametrize("HEAD_DIM", [128, 104])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_host_descriptor_rank3_unit_middle_dim(HEAD_DIM, device):
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
    assert f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" not in kernel.asm["ttir"], "not fused"
    # `b` is a 2D-eligible tl.load as well, so count nothing: check the
    # descriptor load itself kept 2D block I/O.
    loads = [l for l in kernel.asm["ttgir"].splitlines() if "tt.descriptor_load" in l]
    assert loads and all("ttig.block_io" in l for l in loads), loads


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
# not read the next slice. The rebased descriptor keeps R as its row extent, so
# those rows pad (issues/8001).
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
def _scaled_qk_kernel(q_ptr, k_ptr, out_ptr, head, S: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr):
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[2, S, D], strides=[S * D, D, 1], block_shape=[1, BM, D])
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[2, S, D], strides=[S * D, D, 1], block_shape=[1, BN, D])
    scale = 0.125  # A local float is an fp32 tensor: Q gets the extf -> mulf -> truncf chain.
    q = (q_desc.load([head, 0, 0]) * scale).to(tl.bfloat16).reshape(BM, D)
    k = tl.permute(k_desc.load([head, 0, 0]), [0, 2, 1]).reshape(D, BN)
    offs = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(out_ptr + offs, tl.dot(q, k))


# The #8511 attention prologue: Q is scaled and K permuted between the rank-3
# load and its reshape, so both fuse only once the reshape is hoisted to the load.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_rank3_reshape_hoisted_to_load(device, with_allocator):
    S, D, BM, BN, HEAD = 128, 64, 128, 32, 1

    torch.manual_seed(42)
    q = torch.randn((2, S, D), dtype=torch.bfloat16, device=device)
    k = torch.randn((2, S, D), dtype=torch.bfloat16, device=device)
    out = torch.empty((BM, BN), dtype=torch.float32, device=device)
    kernel = _scaled_qk_kernel[(1, )](q, k, out, HEAD, S, D, BM, BN)

    ref = (q[HEAD, :BM].float() * 0.125).to(torch.bfloat16).float() @ k[HEAD, :BN].float().T
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    assert "!tt.tensordesc<1x" not in kernel.asm["ttir"], "not fused"
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


# A negative outermost stride (issues/8001, case 2). Merging the slices divided
# by it and read the block as padding, so it had to decline; the collapse now
# only offsets the base by it. The base sits at the last slice so the negative
# stride walks backwards inside the tensor rather than before the allocation.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_negative_outer_stride(device, with_allocator):
    B, R, C, BLOCK_R, BLOCK_C, BATCH = 4, 32, 64, 32, 64, 1

    torch.manual_seed(42)
    a = torch.randn((B, R, C), dtype=torch.float16, device=device)
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _neg_stride_kernel[(1, )](a[B - 1], out, BATCH, B, R, C, BLOCK_R, BLOCK_C)

    torch.testing.assert_close(out, a[B - 1 - BATCH, :BLOCK_R, :].to(torch.float32))
    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _runtime_empty_kernel(a_ptr, out_ptr, heads, col_off, TOKENS: tl.constexpr, HEAD_DIM: tl.constexpr,
                          S0: tl.constexpr, S1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                          PADDING: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[TOKENS, heads, HEAD_DIM], strides=[S0, S1, 1],
                                     block_shape=[BLOCK_M, 1, BLOCK_D], padding_option=PADDING)
    a = desc.load([0, 0, col_off]).reshape(BLOCK_M, BLOCK_D)
    acc = tl.dot(a, _identity(BLOCK_D))
    offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptr + offs, acc)


# A collapsed dimension that is empty only at runtime, collapsing the *middle*
# dimension. The rank-3 load is pure padding; the fused one keeps the base where
# it is and the guard forces the column index past the head width, which is what
# makes it pad. Device-side because a host TensorDescriptor asserts every shape
# positive and cannot express it. `S1 < HEAD_DIM` makes the heads overlap, which
# nothing rejects.
#
# `col_off` selects the lowering path, and both must pad. It is the stride-one
# index: 0 is provably divisible so `block_io` is granted and the load is a 2D
# block read, which pads by reading past `base_width`; 3 is not, so `block_io` is
# withheld and the generic path pads by predication. `block_io` is asserted in
# both directions, because a silent demotion to a gather would leave the path
# this padding is about untested.
@pytest.mark.parametrize("padding", ["zero", "nan"])
@pytest.mark.parametrize("col_off, block_io", [(0, True), (3, False)])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_empty_collapsed_dim_pads(device, with_allocator, col_off, block_io, padding):
    TOKENS, HEAD_DIM, S0, S1, BLOCK_M, BLOCK_D = 64, 128, 512, 64, 64, 64

    torch.manual_seed(42)
    # Shifted away from zero so padding cannot be mistaken for real data.
    a = torch.randn(TOKENS * S0, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_M, BLOCK_D), -1.0, dtype=torch.float32, device=device)
    kernel = _runtime_empty_kernel[(1, )](a, out, 0, col_off, TOKENS, HEAD_DIM, S0, S1, BLOCK_M, BLOCK_D, padding)

    if padding == "nan":
        assert torch.isnan(out).all(), "padding must keep the descriptor's padding value"
    else:
        torch.testing.assert_close(out, torch.zeros_like(out))

    assert f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    loads = llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')
    assert (loads > 0) == block_io, f"expected block_io={block_io}, got {loads} 2D block loads"


@triton.jit
def _runtime_empty_outer_kernel(a_ptr, out_ptr, batches, R: tl.constexpr, C: tl.constexpr, S0: tl.constexpr,
                                BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, PADDING: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[batches, R, C], strides=[S0, C, 1],
                                     block_shape=[1, BLOCK_R, BLOCK_C], padding_option=PADDING)
    a = desc.load([0, 0, 0]).reshape(BLOCK_R, BLOCK_C)
    acc = tl.dot(a, _identity(BLOCK_C))
    offs = tl.arange(0, BLOCK_R)[:, None] * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tl.store(out_ptr + offs, acc)


# The outermost-collapse counterpart: the guard forces the row index past `R`,
# so the out-of-range read goes out of bounds in a different direction - past
# `base_height` here, past `base_width` above.
@pytest.mark.parametrize("padding", ["zero", "nan"])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_empty_outer_dim_pads(device, with_allocator, padding):
    R, C, S0, BLOCK_R, BLOCK_C = 32, 64, 1024, 32, 64

    torch.manual_seed(42)
    a = torch.randn(8192, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _runtime_empty_outer_kernel[(1, )](a, out, 0, R, C, S0, BLOCK_R, BLOCK_C, padding)

    if padding == "nan":
        assert torch.isnan(out).all(), "padding must keep the descriptor's padding value"
    else:
        torch.testing.assert_close(out, torch.zeros_like(out))

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _runtime_empty_merged_kernel(a_ptr, out_ptr, n, batch, B: tl.constexpr, C: tl.constexpr, S0: tl.constexpr,
                                 BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    # `n * BLOCK_R` is a runtime extent, so the fusion emits its empty-dimension
    # floor.
    desc = tl.make_tensor_descriptor(a_ptr, shape=[B, n * BLOCK_R, C], strides=[S0, C, 1],
                                     block_shape=[1, BLOCK_R, BLOCK_C])
    a = desc.load([batch, 0, 0]).reshape(BLOCK_R, BLOCK_C)
    acc = tl.dot(a, _identity(BLOCK_C))
    offs = tl.arange(0, BLOCK_R)[:, None] * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tl.store(out_ptr + offs, acc)


# The dimension after the collapsed one empty at runtime (`n == 0`), with the
# collapsed one not. The rank-2 view's row extent is then 0, which the 2D block
# message encodes as `0 - 1`: a 16MB surface that reads real data. The fusion
# declares one block of rows instead, and the guard keys on the empty dimension
# too, or an in-range `batch` would read row 0 of that block. `batch` covers both
# halves of the guard and the in-range case; `n == 2` checks that the extra
# runtime terms still read the right slice (1 would be specialized to a
# `constexpr`).
@pytest.mark.parametrize("batch", [-1, 0, 3])
@pytest.mark.parametrize("n", [0, 2])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_empty_merged_dim_pads(device, with_allocator, n, batch):
    B, C, S0, BLOCK_R, BLOCK_C = 4, 64, 4096, 32, 64

    torch.manual_seed(42)
    a = torch.randn((B + 1) * S0, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    # S0 elements in, so that a negative index escaping the guard reads inside the
    # allocation and fails the comparison rather than faulting.
    kernel = _runtime_empty_merged_kernel[(1, )](a[S0:], out, n, batch, B, C, S0, BLOCK_R, BLOCK_C)

    tile = torch.zeros((BLOCK_R, BLOCK_C), dtype=torch.float16, device=device)
    if n > 0 and 0 <= batch < B:
        start = (batch + 1) * S0
        tile = a[start:start + BLOCK_R * BLOCK_C].reshape(BLOCK_R, BLOCK_C)
    torch.testing.assert_close(out, tile.to(torch.float32))

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _runtime_empty_merged_middle_kernel(a_ptr, out_ptr, n, head, TOKENS: tl.constexpr, HEADS: tl.constexpr,
                                        S0: tl.constexpr, S1: tl.constexpr, BLOCK_M: tl.constexpr,
                                        BLOCK_D: tl.constexpr):
    # `n * BLOCK_D` is a runtime head width, so the fusion emits its
    # empty-dimension floor - a surface width on this branch.
    desc = tl.make_tensor_descriptor(a_ptr, shape=[TOKENS, HEADS, n * BLOCK_D], strides=[S0, S1, 1],
                                     block_shape=[BLOCK_M, 1, BLOCK_D])
    a = desc.load([0, head, 0]).reshape(BLOCK_M, BLOCK_D)
    acc = tl.dot(a, _identity(BLOCK_D))
    offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptr + offs, acc)


# The middle-collapse counterpart: the head width is empty at runtime (`n == 0`),
# so the floor is a surface *width* and the forced index runs past `base_width`
# rather than `base_height`. Both heads are in range, so only the guard's
# empty-dimension term makes the load pad; `n == 2` checks that the extra runtime
# terms still read the right head.
@pytest.mark.parametrize("head", [0, 3])
@pytest.mark.parametrize("n", [0, 2])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_empty_merged_middle_dim_pads(device, with_allocator, n, head):
    TOKENS, HEADS, BLOCK_M, BLOCK_D = 64, 4, 64, 64
    S1 = 2 * BLOCK_D
    S0 = HEADS * S1

    torch.manual_seed(42)
    a = torch.randn(TOKENS * S0, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_M, BLOCK_D), -1.0, dtype=torch.float32, device=device)
    kernel = _runtime_empty_merged_middle_kernel[(1, )](a, out, n, head, TOKENS, HEADS, S0, S1, BLOCK_M, BLOCK_D)

    tile = torch.zeros((BLOCK_M, BLOCK_D), dtype=torch.float16, device=device)
    if n > 0:
        tile = torch.as_strided(a, (TOKENS, HEADS, S1), (S0, S1, 1))[:BLOCK_M, head, :BLOCK_D]
    torch.testing.assert_close(out, tile.to(torch.float32))

    assert f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _overlap_outer_kernel(a_ptr, out_ptr, batch, B: tl.constexpr, R: tl.constexpr, C: tl.constexpr, S0: tl.constexpr,
                          BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, PADDING: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[B, R, C], strides=[S0, C, 1], block_shape=[1, BLOCK_R, BLOCK_C],
                                     padding_option=PADDING)
    a = desc.load([batch, 0, 0]).reshape(BLOCK_R, BLOCK_C)
    acc = tl.dot(a, _identity(BLOCK_C))
    offs = tl.arange(0, BLOCK_R)[:, None] * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tl.store(out_ptr + offs, acc)


# An out-of-range collapsed index (4 of 4) clamps the base to the last slice, so
# without the guard the load would read that slice. The rank-3 form pads the
# whole tile, so the fused form must too (#8070). `S0 < R * C` makes the slices
# overlap, which nothing rejects.
#
# `batch` is a runtime argument on purpose. A `tl.constexpr` would still catch the
# wrong values - the guard folds to the extent, as @fuseGuardsCollapsedIndex-
# AtExtent shows - but it would not cover emitting the guard itself.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_overlapping_strides_out_of_range_outer_pads(device, with_allocator):
    B, R, C, S0, BLOCK_R, BLOCK_C = 4, 32, 64, 1024, 32, 64

    torch.manual_seed(42)
    # Shifted away from zero so padding cannot be mistaken for real data.
    a = torch.randn(8192, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _overlap_outer_kernel[(1, )](a, out, B, B, R, C, S0, BLOCK_R, BLOCK_C, "zero")

    torch.testing.assert_close(out, torch.zeros_like(out))

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


# The same, with `padding_option="nan"`: forcing the index out of range is only
# correct if the padding *value* still comes from the descriptor. `test_block_io.
# py::test_block_io_nd_out_of_bounds` covers the 2D path's NaN masks from
# hand-written TTGIR; this reaches them through the fusion.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_out_of_range_collapsed_index_pads_nan(device, with_allocator):
    B, R, C, S0, BLOCK_R, BLOCK_C = 4, 32, 64, 1024, 32, 64

    torch.manual_seed(42)
    a = torch.randn(8192, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _overlap_outer_kernel[(1, )](a, out, B, B, R, C, S0, BLOCK_R, BLOCK_C, "nan")

    assert torch.isnan(out).all(), "padding must keep the descriptor's padding value"

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    # Without this a gather would satisfy the check above and leave the NaN masks
    # this test is here for untested.
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


# A *negative* collapsed index, exercising the guard's `sge` half - the
# out-of-range cases above only reach the `slt` half. The clamp keeps the base
# at slice 0, so without the guard the load would read that slice.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_negative_collapsed_index_pads(device, with_allocator):
    B, R, C, S0, BLOCK_R, BLOCK_C = 4, 32, 64, 1024, 32, 64

    torch.manual_seed(42)
    a = torch.randn(8192, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    # S0 elements in: a multiple of 8 f16 elements, so the base stays 16-byte
    # aligned as `make_tensor_descriptor` requires.
    kernel = _overlap_outer_kernel[(1, )](a[S0:], out, -1, B, R, C, S0, BLOCK_R, BLOCK_C, "zero")

    torch.testing.assert_close(out, torch.zeros_like(out))

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"


# The guard's *true* branch: an emitted guard must still read the slice it was
# asked for. Every other in-range case in the suite folds it away, so this is the
# only one that would catch an inverted `inRange`.
#
# `BATCH` is 2, not 1: `specialize.cc` turns an integer argument equal to 1 into a
# `constexpr`, which would fold the guard and make this a duplicate of the lit
# case. The assertion on the emitted comparisons keeps that from creeping back.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_runtime_in_range_collapsed_index_reads_slice(device, with_allocator):
    B, R, C, S0, BLOCK_R, BLOCK_C, BATCH = 4, 32, 64, 1024, 32, 64, 2

    torch.manual_seed(42)
    a = torch.randn(8192, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_R, BLOCK_C), -1.0, dtype=torch.float32, device=device)
    kernel = _overlap_outer_kernel[(1, )](a, out, BATCH, B, R, C, S0, BLOCK_R, BLOCK_C, "zero")

    # `BLOCK_R == R` and `BLOCK_C == C`, so the slice is contiguous from `BATCH * S0`.
    tile = a[BATCH * S0:BATCH * S0 + BLOCK_R * BLOCK_C].reshape(BLOCK_R, BLOCK_C)
    torch.testing.assert_close(out, tile.to(torch.float32))

    assert f"!tt.tensordesc<1x{BLOCK_R}x{BLOCK_C}x" not in kernel.asm["ttir"], "not fused"
    ttir = kernel.asm["ttir"]
    assert "arith.cmpi sge" in ttir and "arith.cmpi slt" in ttir, "guard folded away"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _overlap_middle_kernel(a_ptr, out_ptr, head, TOKENS: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                           S0: tl.constexpr, S1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    desc = tl.make_tensor_descriptor(a_ptr, shape=[TOKENS, HEADS, HEAD_DIM], strides=[S0, S1, 1],
                                     block_shape=[BLOCK_M, 1, BLOCK_D])
    a = desc.load([0, head, 0]).reshape(BLOCK_M, BLOCK_D)
    acc = tl.dot(a, _identity(BLOCK_D))
    offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptr + offs, acc)


# The middle-collapse counterpart: here the guarded index becomes surface *X*,
# the one `satisfies2DBlockReadAlignment` queries, so this is the case whose
# `block_io` depends on `isDivisible` seeing through the guard's `muli`/`addi`.
# A failure here with the outer case passing localises that immediately.
#
# Head 4 of 4 clamps the base to head 3, so without the guard the load would read
# it. `S1 < HEAD_DIM` overlaps the heads, which nothing rejects.
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_overlapping_strides_out_of_range_middle_pads(device, with_allocator):
    TOKENS, HEADS, HEAD_DIM, S0, S1, BLOCK_M, BLOCK_D = 64, 4, 128, 512, 64, 64, 64

    torch.manual_seed(42)
    a = torch.randn(TOKENS * S0, dtype=torch.float16, device=device) + 1.0
    out = torch.full((BLOCK_M, BLOCK_D), -1.0, dtype=torch.float32, device=device)
    kernel = _overlap_middle_kernel[(1, )](a, out, HEADS, TOKENS, HEADS, HEAD_DIM, S0, S1, BLOCK_M, BLOCK_D)

    torch.testing.assert_close(out, torch.zeros_like(out))

    assert f"!tt.tensordesc<{BLOCK_M}x1x{BLOCK_D}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _head_tiles_kernel(a, c_ptr, tokens, heads, stride_t, stride_h, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    # One (BLOCK_M, BLOCK_D) tile per head, from a host descriptor or one built here.
    # The identity operand leaves the descriptor load the only 2D-eligible one.
    if isinstance(a, tl.tensor_descriptor):
        desc = a
    else:
        desc = tl.make_tensor_descriptor(a, shape=[tokens, heads, BLOCK_D], strides=[stride_t, stride_h, 1],
                                         block_shape=[BLOCK_M, 1, BLOCK_D])
    head = tl.program_id(0)
    tile = desc.load([0, head, 0]).reshape(BLOCK_M, BLOCK_D)
    offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    tl.store(c_ptr + head * BLOCK_M * BLOCK_D + offs, tl.dot(tile, _identity(BLOCK_D)))


def _check_head_tiles(base, shape, strides, ref, host, device):
    """Load the first BLOCK_M rows of every head of the (tokens, heads, head_dim)
    tensor at `base` with `strides`, and compare against `ref`, the same tensor as
    torch sees it (torch has no negative strides)."""
    tokens, heads, head_dim = shape
    BLOCK_M = 64
    c = torch.full((heads, BLOCK_M, head_dim), float("nan"), dtype=torch.float32, device=device)
    if host:
        a = TensorDescriptor(base, list(shape), list(strides), [BLOCK_M, 1, head_dim])
    else:
        a = base
    kernel = _head_tiles_kernel[(heads, )](a, c, tokens, heads, strides[0], strides[1], BLOCK_M, head_dim)

    ref = ref[:BLOCK_M].transpose(0, 1).float()
    wrong = [h for h in range(heads) if not torch.equal(c[h], ref[h])]
    assert not wrong, f"wrong heads: {wrong}"
    assert f"!tt.tensordesc<{BLOCK_M}x1x{head_dim}x" not in kernel.asm["ttir"], "not fused"
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


# A runtime negative strides[1] (issues/8072): heads stored in reverse, the base
# at the last one. Merging the head into the column divided by the stride and
# fused whatever its sign, so every head but the first landed at a negative
# column and read as zeros. A host descriptor and one built from kernel
# arguments both carry the stride as a runtime value.
@pytest.mark.parametrize("host", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_negative_middle_stride(host, device, with_allocator):
    TOKENS, HEADS, HEAD_DIM = 128, 4, 128
    a = torch.randn((TOKENS, HEADS, HEAD_DIM), dtype=torch.float16, device=device)
    _check_head_tiles(a[:, -1], a.shape, [HEADS * HEAD_DIM, -HEAD_DIM, 1], a.flip(1), host, device)


# A (heads, tokens, head_dim) tensor read as (tokens, heads, head_dim), e.g. BHSD
# storage seen as BSHD, has strides [head_dim, tokens * head_dim, 1]. Merging put
# head h at column h * tokens * head_dim of one surface, whose width passed the
# 24-bit field from head 128 on (128 * 512 * 128 * 2 bytes == 16 MiB), so those
# heads read as zeros.
@pytest.mark.parametrize("host", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_wide_middle_stride(host, device, with_allocator):
    HEADS, TOKENS, HEAD_DIM = 130, 512, 128
    storage = torch.randn((HEADS, TOKENS, HEAD_DIM), dtype=torch.float16, device=device)
    view = storage.transpose(0, 1)
    _check_head_tiles(storage, view.shape, view.stride(), view, host, device)


@triton.jit
def _runtime_strides_kernel(a_ptr, out_ptr, idx, stride0, stride1, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                            OUTER: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    # A unit outer ([1, R, C]) or middle ([R, 1, C]) block, collapsed at `idx`.
    if OUTER:
        desc = tl.make_tensor_descriptor(a_ptr, shape=[S0, S1, S2], strides=[stride0, stride1, 1],
                                         block_shape=[1, BLOCK_R, BLOCK_C])
        a = desc.load([idx, 0, 0]).reshape(BLOCK_R, BLOCK_C)
    else:
        desc = tl.make_tensor_descriptor(a_ptr, shape=[S0, S1, S2], strides=[stride0, stride1, 1],
                                         block_shape=[BLOCK_R, 1, BLOCK_C])
        a = desc.load([0, idx, 0]).reshape(BLOCK_R, BLOCK_C)
    acc = tl.dot(a, _identity(BLOCK_C))
    offs = tl.arange(0, BLOCK_R)[:, None] * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    tl.store(out_ptr + offs, acc)


# A runtime collapsed stride that is a 16-byte multiple, as `make_tensor_descriptor`
# requires, but 8 f16 elements past a multiple of 16, so it gets no divisibility
# hint. Folding the collapsed index into the base must not leave the base provably
# 2-byte aligned only, which keeps the load off 2D block I/O: the unfused outer load
# had 2D block I/O, and the middle one gets it as well. Whether the load fuses is
# not the point here, so only its values and its 2D block loads are checked; the
# identity operand leaves it the only 2D-eligible load.
@pytest.mark.parametrize("outer", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_descriptor_unhinted_collapsed_stride_block_io(outer, device, with_allocator):
    N, R, C, IDX = 4, 32, 64, 2
    if outer:  # (N, R, C), slices R * C + 8 elements apart
        shape, strides = (N, R, C), (R * C + 8, C, 1)
    else:  # (R, N, C), heads C + 8 elements apart
        shape, strides = (R, N, C), (N * (C + 8), C + 8, 1)

    torch.manual_seed(42)
    a = torch.randn(shape[0] * strides[0], dtype=torch.float16, device=device)
    out = torch.full((R, C), -1.0, dtype=torch.float32, device=device)
    kernel = _runtime_strides_kernel[(1, )](a, out, IDX, strides[0], strides[1], *shape, outer, R, C)

    view = torch.as_strided(a, shape, strides)
    tile = view[IDX] if outer else view[:, IDX]
    torch.testing.assert_close(out, tile.to(torch.float32))
    llir = kernel.asm["llir"]
    assert llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead') > 0


@triton.jit
def _bool_copy_kernel(src_desc, dst_desc):
    dst_desc.store([0, 0], src_desc.load([0, 0]))


# Bool descriptors are bytes in memory: nonzero loads as True, stores write 0x01 (#8670).
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
def test_host_descriptor_bool_copy(device):
    raw = torch.ones((1, 32), dtype=torch.uint8, device=device)  # hidden columns 20-31 are nonzero
    raw[0, :20] = torch.tensor([0, 1, 2, 1] * 5, dtype=torch.uint8)  # 0x02 is nonzero with the low bit clear
    src = raw.view(torch.bool)[:, :20]  # shape [1, 20], stride [32, 1]: meets the 16-byte stride rule
    dst = torch.full((1, 32), 0xAA, dtype=torch.uint8, device=device)  # poison, so the fill lanes must be written
    kernel = _bool_copy_kernel[(1, )](TensorDescriptor.from_tensor(src, [1, 32]),
                                      TensorDescriptor.from_tensor(dst.view(torch.bool), [1, 32]))

    expected = torch.zeros_like(raw)
    expected[0, :20] = (raw[0, :20] != 0).to(torch.uint8)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    ttgir = kernel.asm["ttgir"]
    assert "tt.descriptor_load" in ttgir and "tt.descriptor_store" in ttgir, "expected descriptor-native ops"
    assert "ttig.block_io" not in ttgir, "expected the generic descriptor path"
