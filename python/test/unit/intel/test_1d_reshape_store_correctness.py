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


def _has_block_store(llir):
    """True if the kernel emitted a 2D block store message."""
    return 'spirv_Subgroup2DBlockStoreINTEL' in llir or 'GenISA.LSC2DBlockWrite' in llir


def _has_block_load(llir):
    """True if the kernel emitted a 2D block load message."""
    return 'spirv_Subgroup2DBlockLoadINTEL' in llir or 'GenISA.LSC2DBlockRead' in llir


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "W, S, XBLOCK, num_warps, expect_block_store, dtype_str",
    [
        # H = XBLOCK / W = 1: a single contiguous row.  num_warps must be 1 so
        # that H / num_warps >= 1.
        (32, 96, 32, 1, True, "float16"),
        (32, 128, 32, 1, True, "float16"),
        (32, 192, 32, 1, True, "float16"),
        # H > 1: the multi-row case.  Per-warp height is H/num_warps = 8, the
        # store hardware maximum, so one 2D block store replaces 32 rows of
        # scatter.  This is the case the `H != 1` guard used to block.
        (32, 96, 1024, 4, True, "float16"),
        (32, 128, 1024, 4, True, "float16"),
        (32, 192, 1024, 4, True, "float16"),
        # W < threadsPerWarp (16 < 32): must fall back.  A [1, tpw] encoding on
        # a dimension of size W < tpw is replicated and cannot be legalized
        # (same reason as the load-side bail-out for issue #6738).
        (16, 96, 512, 4, False, "float16"),
    ],
    ids=[
        "H1_W32_S96_f16",
        "H1_W32_S128_f16",
        "H1_W32_S192_f16",
        "H32_W32_S96_f16",
        "H32_W32_S128_f16",
        "H32_W32_S192_f16",
        "H32_W16_S96_f16_narrow_fallback",
    ],
)
def test_1d_reshape_strided_store(W, S, XBLOCK, num_warps, expect_block_store, dtype_str, device):
    """Test 1D-to-2D block store reshape and fallback produce correct results.

    The Inductor-style strided store is lowered via `reshape1DStridedStore` to a
    2D block store, for H == 1 and for H > 1 alike: the value is reshaped to the
    natural 2D encoding and then converted into the hardware delivery encoding
    (lane k owns column k, registers stack rows).  `expect_block_store` pins
    whether the optimization fired — without it a silently-skipped reshape would
    still pass the numeric checks below.
    """
    # On devices without 2D block IO the optimization is not emitted; override
    # the parametrized expectation so the numeric checks still run.
    if not triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False):
        expect_block_store = False

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
    kernel = strided_store_kernel[(grid, )](
        x_tri,
        out_tri,
        xnumel,
        W=W,
        S=S,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
    )

    # Pin whether the 1D->2D reshape actually fired.  The numeric assertions
    # below pass either way, so without this a regression that silently skips
    # the optimization would go unnoticed.
    llir = kernel.asm["llir"]
    assert _has_block_store(llir) == expect_block_store, (
        f"expected block store: {expect_block_store}, got {not expect_block_store} "
        f"for W={W}, S={S}, XBLOCK={XBLOCK}, num_warps={num_warps}")

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
        # i8 / W=64 > tpw=32 / H=1.
        # BLOCK=W*H=64. Single instance, pure arange — matchStridedPattern fires.
        # Before the fix: tile_width=64, elem_size=8 → lane l stores to
        #   dst[l] and dst[l+32] but holds cols 2l and 2l+1 → wrong columns.
        # After the fix:  tile_width=32, elem_size=16 → packed correctly.
        (64, 128, 1, "int8"),
        (64, 192, 1, "int8"),
        # Same, with H = 8 = maxPerWarpHeight for stores.  This combines the
        # W > tpw packing with the multi-row geometry unblocked by lifting the
        # `H != 1` guard: sizePerThread = [8, 2], so each lane holds 2 adjacent
        # columns of all 8 rows and the packed tile is 32 wide by 8 high.
        (64, 128, 8, "int8"),
        (64, 192, 8, "int8"),
    ],
    ids=[
        "H1_W64_S128_i8_wgt_tpw",
        "H1_W64_S192_i8_wgt_tpw",
        "H8_W64_S128_i8_wgt_tpw",
        "H8_W64_S192_i8_wgt_tpw",
    ],
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

    kernel = _wgt_store_kernel[(1, )](x_tri, out_tri, W=W, S=S, BLOCK=BLOCK, num_warps=1)

    assert _has_block_store(kernel.asm["llir"]), \
        f"1D->2D store reshape did not fire for W={W} S={S} H={H}"

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


# ---------------------------------------------------------------------------
# Regression tests for the wrong per-warp row base in reshape1DStridedLoad.
#
# See: https://github.com/intel/intel-xpu-backend-for-triton/issues/7918
#
# The pointer and mask used to be reshaped with `allow_reorder
# efficient_layout`, which only retypes the tensor and moves no data, so the
# registers still held pointers in 1D order while the type claimed the 2D
# block load's delivery encoding.  The lowering reads each warp's base address
# out of that tensor, so warp w started at the row the *1D* layout put there
# rather than at row perWarpH*w.
#
# The bug is invisible unless the 1D source layout needs more than one
# register repetition per lane, i.e.
#
#     reps1D = XBLOCK / (sizePerThread1D * threadsPerWarp * num_warps) > 1
#
# because at reps1D == 1 the two layouts happen to agree on the warp base.
# That is why the tests above miss it: `test_1d_reshape_strided_load` uses
# XBLOCK=1024 with num_warps=4, giving 1024/(8*32*4) = 1, and the
# `_w_gt_tpw` tests use num_warps=1, where no per-warp base exists at all.
# Every config below has reps1D >= 2 and num_warps >= 2.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "W, S, XBLOCK, num_warps, dtype_str",
    [
        # The shape reported in #7918: reps1D = 4096/(8*32*4) = 4, per-warp
        # height 32 (the load hardware maximum).  Warp w read row 8*w instead
        # of 32*w, so 3072 of every 4096 elements came back wrong.
        (32, 96, 4096, 4, "float16"),
        (32, 128, 4096, 4, "float16"),
        (32, 192, 4096, 4, "float16"),
        # Fewer warps, same reps1D = 2048/(8*32*2) = 4: per-warp height is
        # still 32 but there are only two warps to get wrong.
        (32, 96, 2048, 2, "float16"),
        # More warps, reps1D = 4096/(8*32*8) = 2 and per-warp height 16.
        # Exercises a per-warp height below the hardware maximum.
        (32, 96, 4096, 8, "float16"),
    ],
    ids=[
        "W32_S96_X4096_w4_f16",
        "W32_S128_X4096_w4_f16",
        "W32_S192_X4096_w4_f16",
        "W32_S96_X2048_w2_f16",
        "W32_S96_X4096_w8_f16",
    ],
)
def test_1d_reshape_strided_load_multi_warp_base(W, S, XBLOCK, num_warps, dtype_str, device):
    """Every warp must read from its own row base after the 1D->2D load reshape.

    Uses the same Inductor-style gather as `test_1d_reshape_strided_load` but
    with reps1D >= 2, the regime in which a wrong per-warp base is observable.
    Four program instances are launched so the per-program offset is exercised
    alongside the per-warp base.
    """
    # Enough rows for four program instances.
    rows_per_block = XBLOCK // W
    num_rows = 4 * rows_per_block
    xnumel = W * num_rows

    rs = RandomState(17)
    # Padded 2D input surface [num_rows, S]; the kernel reads its first W columns.
    in_full = numpy_random((num_rows, S), dtype_str=dtype_str, rs=rs)
    out_ref = in_full[:, :W].flatten() * np.dtype(dtype_str).type(2.0)

    in_tri = to_triton(in_full.flatten(), device=device)
    out_tri = torch.zeros(xnumel, dtype=getattr(torch, dtype_str), device=device)

    grid = (xnumel + XBLOCK - 1) // XBLOCK
    kernel = strided_load_kernel[(grid, )](
        in_tri,
        out_tri,
        xnumel,
        W=W,
        S=S,
        XBLOCK=XBLOCK,
        num_warps=num_warps,
    )

    # Pin that the 1D->2D reshape actually fired.  Without this the test would
    # keep passing if the optimization were disabled, and would no longer guard
    # anything.  Skip the check on devices that have no 2D block I/O.
    if triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False):
        assert _has_block_load(kernel.asm["llir"]), \
            f"1D->2D load reshape did not fire for W={W} S={S} XBLOCK={XBLOCK} num_warps={num_warps}"

    np.testing.assert_allclose(
        to_numpy(out_tri),
        out_ref,
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"Strided load mismatch for W={W}, S={S}, XBLOCK={XBLOCK}, num_warps={num_warps}",
    )
