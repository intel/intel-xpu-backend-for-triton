import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri
from triton.backends.intel.compiler import (MAX_REG_SPILL_SLOTS_PER_LANE, XPUBackend, extract_spill_size_from_zebin,
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


def test_igc_disable_loop_sink(device):
    # The LoopSink workaround for the software fp8e4m3 -> fp16 sequence (issue 8046)
    # must stay scoped: it applies only on LTS drivers, and only to kernels that
    # actually contain that conversion.
    from triton.runtime import driver

    arch = driver.active.get_current_target().arch
    expected = XPUBackend.is_lts(arch.get("driver_version")) and not arch.get("has_f8_conversions", False)

    SIZE = 128

    @triton.jit
    def upcast_kernel(X, Y, SIZE: tl.constexpr):
        offs = tl.arange(0, SIZE)
        tl.store(Y + offs, tl.load(X + offs).to(tl.float16))

    @triton.jit
    def scale_kernel(X, Y, SIZE: tl.constexpr):
        offs = tl.arange(0, SIZE)
        tl.store(Y + offs, tl.load(X + offs) * 2.0)

    y = torch.empty(SIZE, dtype=torch.float16, device=device)
    x8 = torch.randint(0, 256, (SIZE, ), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    k = upcast_kernel[(1, )](x8, y, SIZE=SIZE, generate_native_code=True)
    assert k.metadata.igc_disable_loop_sink == expected

    k = scale_kernel[(1, )](y, y, SIZE=SIZE, generate_native_code=True)
    assert not k.metadata.igc_disable_loop_sink
