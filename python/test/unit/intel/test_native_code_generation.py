import pytest
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri
from triton.backends.intel.compiler import (MAX_REG_SPILL_SLOTS_PER_LANE, extract_spill_size_from_zebin,
                                            spill_slots_per_lane)


def test_empty_kernel(device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    x = to_triton(numpy_random(SIZE, dtype_str="bfloat16"), device=device, dst_type="bfloat16")
    kernel[(1, )](x, SIZE=SIZE, num_warps=4, generate_native_code=True)


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
    if spill_slots <= MAX_REG_SPILL_SLOTS_PER_LANE:
        pytest.skip(f"Kernel did not spill above the threshold ({spill_slots} <= "
                    f"{MAX_REG_SPILL_SLOTS_PER_LANE} dword-equivalents/lane, from {spill_size} B/thread); "
                    "auto-large-GRF path was not exercised. Consider increasing SIZE.")
    assert "-cl-intel-256-GRF-per-thread" in k.metadata.build_flags
