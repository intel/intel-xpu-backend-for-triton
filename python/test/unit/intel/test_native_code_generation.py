import pytest
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri
from triton.backends.intel.compiler import (MAX_REG_SPILL_SLOTS_PER_LANE, XPUBackend, accepts_default_grf,
                                            extract_spill_size_from_zebin, spill_slots_per_lane)


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
        # At and just past the rolling limit, both SIMD widths.
        (2048, 32, False, True),  # 16 slots/lane -- at the limit
        (2176, 32, False, False),  # 17 slots/lane -- rebuild
        (1024, 16, False, True),  # 16 slots/lane at SIMD16
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
    # dword-equivalents per lane, exceeds the reported-`n_spills` budget.
    k = kernel[(1, )](x, SIZE=SIZE, num_warps=1, generate_native_code=True, grf_mode='default')
    zebin = tmp_path / "kernel.zebin"
    zebin.write_bytes(k.kernel)
    spill_size = extract_spill_size_from_zebin(str(zebin))
    spill_slots = spill_slots_per_lane(spill_size, k.metadata.threads_per_warp)
    # The gate differs by driver line (issue #8106), so ask the predicate the
    # backend itself uses rather than re-deriving the rolling rule here.
    is_lts = XPUBackend.is_lts(k.metadata.target.arch.get("driver_version"))
    if accepts_default_grf(spill_size, k.metadata.threads_per_warp, is_lts):
        threshold = "any spill" if is_lts else f"{MAX_REG_SPILL_SLOTS_PER_LANE} dword-equivalents/lane"
        pytest.skip(f"Kernel did not spill above the threshold ({spill_slots} dword-equivalents/lane "
                    f"from {spill_size} B/thread, limit is {threshold}); auto-large-GRF path was not "
                    "exercised. Consider increasing SIZE.")
    assert "-cl-intel-256-GRF-per-thread" in k.metadata.build_flags
