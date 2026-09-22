import pytest
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri
from triton.backends.intel.compiler import (XPUBackend, accepts_default_grf, extract_spill_size_from_zebin,
                                            min_spill_slots_for_rebuild, spill_slots_per_lane)


def test_empty_kernel(device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    x = to_triton(numpy_random(SIZE, dtype_str="bfloat16"), device=device, dst_type="bfloat16")
    kernel[(1, )](x, SIZE=SIZE, num_warps=4, generate_native_code=True)


@pytest.mark.parametrize(
    "spill_size, threads_per_warp, is_lts, accepted",
    [
        # No spill: nothing to rebuild for, on either driver line.
        (0, 32, False, True),
        (0, 32, True, True),
        # 64 B is 0 slots/lane at SIMD32. Rolling accepts it; LTS must NOT, or
        # the byte-wise rule degenerates into the slot-wise one (issue #8106).
        (64, 32, False, True),
        (64, 32, True, False),
        # The rolling boundary is 1024 B/hardware-thread at every SIMD width. This
        # triple is the only check of that width invariance (issue #8077).
        (960, 32, False, True),  # 7 slots/lane
        (1024, 32, False, False),  # 8 slots/lane -- first rebuild
        (960, 16, False, True),  # 15 slots/lane
        (1024, 16, False, False),  # 16 slots/lane -- same bytes, half the width
        (544, 8, False, True),  # 17 slots/lane -- SIMD8 reversal, `slots > 16` rebuilt here
        (1024, 8, False, False),  # 32 slots/lane
        # Unknown width falls back to raw bytes on both sides. Only meaningful as a
        # pair: an erroneous 8-slot threshold also rejects 1024, but accepts 960.
        (960, 0, False, True),
        (1024, 0, False, False),
        # Past the boundary, including a real dead-zone size from the #8077 census.
        (1408, 32, False, False),  # 11 slots/lane
        (2048, 32, False, False),  # 16 slots/lane
        (2176, 32, False, False),  # 17 slots/lane
        (1088, 16, False, False),  # 17 slots/lane at SIMD16
        # LTS rebuilds for any spill regardless of magnitude or width.
        (2048, 32, True, False),
        (1024, 16, True, False),
    ],
)
def test_accepts_default_grf(spill_size, threads_per_warp, is_lts, accepted):
    assert accepts_default_grf(spill_size, threads_per_warp, is_lts) is accepted


@pytest.mark.xfail(is_xpu_cri(), reason="unable to get spill_size")
def test_auto_large_grf(device, tmp_path):
    SIZE = 2048

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        x = tl.arange(0, SIZE)
        y = tl.sort(x, descending=True)
        tl.store(X + x, y)

    x = to_triton(numpy_random(SIZE, dtype_str="float32"), device=device, dst_type="float32")
    # Triton XPU chooses large GRF mode when the spill frame, normalized to
    # dword-equivalents per lane, reaches the reported-`n_spills` budget.
    k = kernel[(1, )](x, SIZE=SIZE, num_warps=1, generate_native_code=True, grf_mode='default')
    if "-cl-intel-256-GRF-per-thread" in k.metadata.build_flags:
        return  # the rebuild fired, which is the whole assertion

    # Read the outcome before the predicate: `make_zebin` keeps the *adopted*
    # binary, so a successful rebuild lowers `k.kernel`'s spill below the gate and
    # asking the predicate first would skip exactly when it should assert. Reaching
    # here means no rebuild happened, so this spill is the one the gate acted on.
    zebin = tmp_path / "kernel.zebin"
    zebin.write_bytes(k.kernel)
    spill_size = extract_spill_size_from_zebin(str(zebin))
    spill_slots = spill_slots_per_lane(spill_size, k.metadata.threads_per_warp)
    # The gate differs by driver line (issue #8106), so ask the predicate the
    # backend itself uses rather than re-deriving the rolling rule here.
    is_lts = XPUBackend.is_lts(k.metadata.target.arch.get("driver_version"))
    threshold = ("any spill"
                 if is_lts else f"{min_spill_slots_for_rebuild(k.metadata.threads_per_warp)} dword-equivalents/lane")
    observed = f"{spill_slots} dword-equivalents/lane from {spill_size} B/thread, rebuild at {threshold}"
    if not accepts_default_grf(spill_size, k.metadata.threads_per_warp, is_lts):
        pytest.fail(f"Gate should have rebuilt at large GRF but did not ({observed}).")
    pytest.skip(f"Kernel did not spill up to the threshold ({observed}); auto-large-GRF path was not "
                "exercised. Consider increasing SIZE.")
