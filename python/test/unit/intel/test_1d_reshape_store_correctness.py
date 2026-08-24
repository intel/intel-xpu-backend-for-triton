"""Correctness tests for 1D-to-2D block I/O reshaping in MaterializeBlockPointer.

Tests for:
1. Strided stores using the Inductor pattern (reshape1DStridedStore)
2. Strided loads using the Inductor pattern (reshape1DStridedLoad)

Both patterns use:  addr = (xindex % W) + (xindex // W) * S

See: https://github.com/intel/intel-xpu-backend-for-triton/issues/6532
"""

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
from triton._internal_testing import is_xpu, numpy_random, to_triton, to_numpy


@triton.jit
def strided_store_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    W: tl.constexpr,
    S: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """Inductor-style strided store: addr = (xindex % W) + (xindex // W) * S."""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # Load from contiguous input
    val = tl.load(in_ptr + xindex, mask=xmask)

    # Simple computation to avoid being optimized away
    val = val * 0.5

    # Strided store address computation (Inductor pattern)
    col = xindex % W
    row = xindex // W
    out_offset = col + row * S

    # Splat-true mask: this is what triggers the 1D reshape optimization
    mask = tl.full([XBLOCK], True, tl.int1)

    tl.store(out_ptr + out_offset, val, mask=mask)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "W, S, XBLOCK, num_warps, dtype_str",
    [
        # H = XBLOCK / W = 1: this IS the case that exercises
        # `reshape1DStridedStore` (it requires H == 1 per the TODO in
        # MaterializeBlockPointer.cpp).  num_warps must be 1 so that
        # H / num_warps >= 1.
        (32, 96, 32, 1, "float16"),
        (32, 128, 32, 1, "float16"),
        (32, 192, 32, 1, "float16"),
        # H > 1: baseline functional correctness only — the store reshape
        # optimization is currently disabled for H != 1, so these cases do
        # not exercise the optimized path but verify the fallback gather
        # store remains correct.
        (32, 96, 1024, 4, "float16"),
        (32, 128, 1024, 4, "float16"),
        (32, 192, 1024, 4, "float16"),
    ],
    ids=[
        "H1_W32_S96_f16",
        "H1_W32_S128_f16",
        "H1_W32_S192_f16",
        "H32_W32_S96_f16_fallback",
        "H32_W32_S128_f16_fallback",
        "H32_W32_S192_f16_fallback",
    ],
)
def test_1d_reshape_strided_store(W, S, XBLOCK, num_warps, dtype_str, device):
    """Test 1D-to-2D block store reshape and fallback produce correct results.

    With H = XBLOCK / W == 1, the Inductor-style strided store is lowered
    via `reshape1DStridedStore` to a 2D block store.  With H > 1, the
    current implementation rejects the reshape (TODO: hardware transpose
    unsupported) and this case exercises the gather-store fallback.
    """
    num_rows = 1024
    xnumel = W * num_rows  # total elements

    # Generate reproducible input data
    rs = RandomState(17)
    x_np = numpy_random((xnumel, ), dtype_str=dtype_str, rs=rs)

    # Compute reference output with numpy.
    # The output buffer has shape [num_rows, S] (stride S between rows).
    # Each xindex stores val*0.5 at col=(xindex%W), row=(xindex//W).
    x_scaled = (x_np * np.float16(0.5)).reshape(num_rows, W)
    out_ref = np.zeros((num_rows, S), dtype=x_np.dtype)
    out_ref[:, :W] = x_scaled

    # Convert to device tensors
    x_tri = to_triton(x_np, device=device)
    out_tri = torch.zeros(num_rows * S, dtype=x_tri.dtype, device=device)

    # Launch kernel
    grid = (xnumel + XBLOCK - 1) // XBLOCK
    strided_store_kernel[(grid, )](
        x_tri,
        out_tri,
        xnumel,
        W=W,
        S=S,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
    )

    # Compare: reshape output to [num_rows, S] and check the first W columns
    # of each row (the rest should remain zero)
    out_actual = to_numpy(out_tri).reshape(num_rows, S)

    # Check the stored values (first W columns of each row)
    np.testing.assert_allclose(
        out_actual[:, :W],
        out_ref[:, :W],
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Strided store mismatch for W={W}, S={S}",
    )

    # Check that the remaining columns are untouched (zero)
    np.testing.assert_allclose(
        out_actual[:, W:],
        np.zeros((num_rows, S - W), dtype=out_ref.dtype),
        rtol=0,
        atol=0,
        err_msg=f"Non-zero values outside stored region for W={W}, S={S}",
    )


@triton.jit
def strided_load_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    W: tl.constexpr,
    S: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    """Inductor-style strided load: addr = (xindex % W) + (xindex // W) * S."""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # Strided load address computation (Inductor pattern)
    col = xindex % W
    row = xindex // W
    in_offset = col + row * S

    # Splat-true mask: triggers the 1D reshape optimization
    mask = tl.full([XBLOCK], True, tl.int1)

    val = tl.load(in_ptr + in_offset, mask=mask)

    # Simple computation to avoid being optimized away
    val = val * 2.0

    # Contiguous store
    tl.store(out_ptr + xindex, val, mask=xmask)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "W, S, dtype_str",
    [
        (32, 96, "float16"),
        (32, 128, "float16"),
        (32, 192, "float16"),
        # W < threadsPerWarp (16 < 32) — regression test for issue #6738.
        # Previously crashed in make_llir with "expensive view not supported
        # on reshape op" because the load encoding was constructed with
        # threadsPerWarp=[1, 32] which replicated data across lanes.
        (16, 96, "float16"),
        (16, 128, "float16"),
    ],
    ids=["W32_S96_f16", "W32_S128_f16", "W32_S192_f16", "W16_S96_f16", "W16_S128_f16"],
)
def test_1d_reshape_strided_load(W, S, dtype_str, device):
    """Test 1D-to-2D block load reshape produces correct results.

    The kernel does a strided gather load from a padded 2D surface,
    multiplies by 2, and stores contiguously. We compare against numpy.
    """
    XBLOCK = 1024
    num_rows = 1024
    xnumel = W * num_rows
    num_warps = 4

    rs = RandomState(17)
    # Create padded 2D input surface [num_rows, S]
    in_full = numpy_random((num_rows, S), dtype_str=dtype_str, rs=rs)
    # Reference: read first W columns of each row, multiply by 2
    in_values = in_full[:, :W].flatten()
    out_ref = in_values * np.dtype(dtype_str).type(2.0)

    # Device tensors
    in_tri = to_triton(in_full.flatten(), device=device)
    out_tri = torch.zeros(xnumel, dtype=getattr(torch, dtype_str), device=device)

    grid = (xnumel + XBLOCK - 1) // XBLOCK
    strided_load_kernel[(grid, )](
        in_tri,
        out_tri,
        xnumel,
        W=W,
        S=S,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
    )

    np.testing.assert_allclose(
        to_numpy(out_tri),
        out_ref,
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Strided load mismatch for W={W}, S={S}",
    )


# ---------------------------------------------------------------------------
# Regression tests for the W > threadsPerWarp correctness bug.
#
# When W > threadsPerWarp the hand-built BlockIOTileSizeInfo hardcoded
# numElemPerPackedVal=1 / tileWidth=W, which mis-described the tile to the
# hardware (lane l was given cols l and l+tpw, while it held cols 2l and
# 2l+1). The shared getBlockIOTileSize helper packs the adjacent per-lane
# columns into a wider element, producing the correct tile.
#
# On BMG (threadsPerWarp=32), the only reachable W>tpw case via
# matchStridedPattern is 8-bit elements with W=64 (the payload restriction
# caps 16-bit at W=32 = tpw). The test uses int8 to hit this path.
# ---------------------------------------------------------------------------
@triton.jit
def _wgt_store_kernel(
    src_ptr,
    dst_ptr,
    W: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-instance strided store.  BLOCK = W*H, H <= maxPerWarpHeight=8.

    The offset MUST be computed as a single expression before adding to the
    pointer.  'dst_ptr + off_a + off_b' generates two chained addptr ops;
    matchStridedPattern requires one addptr(splat, addi(remui, muli)).
    """
    i = tl.arange(0, BLOCK)
    v = tl.load(src_ptr + i)
    offset = (i % W) + (i // W) * S
    mask = tl.full([BLOCK], True, tl.int1)
    tl.store(dst_ptr + offset, v, mask=mask)


@triton.jit
def _wgt_load_kernel(
    src_ptr,
    dst_ptr,
    W: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-instance strided load.  BLOCK = W*H, H <= maxPerWarpHeight=32."""
    i = tl.arange(0, BLOCK)
    offset = (i % W) + (i // W) * S
    mask = tl.full([BLOCK], True, tl.int1)
    v = tl.load(src_ptr + offset, mask=mask)
    tl.store(dst_ptr + i, v)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "W, S, H, dtype_str",
    [
        # i8 / W=64 > tpw=32 / H=1 (store is gated at H=1 anyway).
        # BLOCK=W*H=64. Single instance, pure arange — matchStridedPattern fires.
        # Before the fix: tile_width=64, elem_size=8 → lane l stores to
        #   dst[l] and dst[l+32] but holds cols 2l and 2l+1 → wrong columns.
        # After the fix:  tile_width=32, elem_size=16 → packed correctly.
        (64, 128, 1, "int8"),
        (64, 192, 1, "int8"),
    ],
    ids=["H1_W64_S128_i8_wgt_tpw", "H1_W64_S192_i8_wgt_tpw"],
)
def test_1d_reshape_strided_store_w_gt_tpw(W, S, H, dtype_str, device):
    """Regression test: 1D strided store with W > threadsPerWarp must produce correct data."""
    BLOCK = W * H
    rs = RandomState(17)
    x_np = numpy_random((BLOCK, ), dtype_str=dtype_str, rs=rs)

    out_size = (H - 1) * S + W
    out_ref = np.zeros(out_size, dtype=x_np.dtype)
    for idx in range(BLOCK):
        out_ref[idx % W + (idx // W) * S] = x_np[idx]

    x_tri = to_triton(x_np, device=device)
    out_tri = torch.zeros(out_size, dtype=x_tri.dtype, device=device)

    _wgt_store_kernel[(1, )](x_tri, out_tri, W=W, S=S, BLOCK=BLOCK, num_warps=1)

    np.testing.assert_array_equal(
        to_numpy(out_tri),
        out_ref,
        err_msg=f"Store mismatch W={W} S={S} H={H} (W > threadsPerWarp bug)",
    )


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "W, S, H, dtype_str",
    [
        # i8 / W=64 > tpw=32 / H=8 (load allows up to H=32*numWarps=32).
        # BLOCK=W*H=512. Single instance.
        # Before the fix: tile_width=64, elem_size=8 tile_height=8 →
        #   each lane reads the wrong columns (interleaved instead of adjacent).
        # After the fix:  tile_width=32, elem_size=16 → packed correctly.
        (64, 128, 8, "int8"),
        (64, 192, 8, "int8"),
    ],
    ids=["H8_W64_S128_i8_wgt_tpw", "H8_W64_S192_i8_wgt_tpw"],
)
def test_1d_reshape_strided_load_w_gt_tpw(W, S, H, dtype_str, device):
    """Regression test: 1D strided load with W > threadsPerWarp must read correct data."""
    BLOCK = W * H
    rs = RandomState(17)
    in_full = numpy_random((H, S), dtype_str=dtype_str, rs=rs)
    out_ref = np.array([in_full[i // W, i % W] for i in range(BLOCK)], dtype=in_full.dtype)

    in_tri = to_triton(in_full.flatten(), device=device)
    out_tri = torch.zeros(BLOCK, dtype=in_tri.dtype, device=device)

    _wgt_load_kernel[(1, )](in_tri, out_tri, W=W, S=S, BLOCK=BLOCK, num_warps=1)

    np.testing.assert_array_equal(
        to_numpy(out_tri),
        out_ref,
        err_msg=f"Load mismatch W={W} S={S} H={H} (W > threadsPerWarp bug)",
    )
