import pytest
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri
from triton.backends.intel.compiler import (REBUILD_SPILL_BYTES_PER_THREAD, XPUBackend, accepts_default_grf,
                                            extract_spill_size_from_zebin)


def test_empty_kernel(device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    x = to_triton(numpy_random(SIZE, dtype_str="bfloat16"), device=device, dst_type="bfloat16")
    kernel[(1, )](x, SIZE=SIZE, num_warps=4, generate_native_code=True)


# Both operands are bytes per hardware thread, the unit both spill probes report, so
# there is no sub-group width to parametrize over -- the gate cannot depend on one.
@pytest.mark.parametrize(
    "spill_size, is_lts, accepted",
    [
        # No spill: nothing to rebuild for, on either driver line.
        (0, False, True),
        (0, True, True),
        # 64 B truncates to 0 dword-equivalents/lane at SIMD32, so a slot-wise rule
        # accepts it on both lines. LTS must not (issue #8106): a 64 B config is one
        # of the 12 that rule covers.
        (64, False, True),
        (64, True, False),
        # The rolling boundary.
        (960, False, True),
        (1024, False, False),  # first rebuild
        # Inside the band #7959 silenced: at SIMD32 these are 11 and 16
        # dword-equivalents/lane, both at or under its 16-slot budget, and 1408 B is a
        # real mixnet_l size from the #8077 census.
        (1408, False, False),
        (2048, False, False),
        # LTS rebuilds for any spill regardless of magnitude.
        (960, True, False),
        (2048, True, False),
    ],
)
def test_accepts_default_grf(spill_size, is_lts, accepted):
    assert accepts_default_grf(spill_size, is_lts) is accepted


def test_auto_large_grf(device, tmp_path):
    # CRI's larger register file only spills past the large-GRF upgrade at a
    # bigger tile; other targets already spill at 2048.
    SIZE = 4096 if is_xpu_cri() else 2048

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        x = tl.arange(0, SIZE)
        y = tl.sort(x, descending=True)
        tl.store(X + x, y)

    x = to_triton(numpy_random(SIZE, dtype_str="float32"), device=device, dst_type="float32")
    # Triton XPU chooses large GRF mode once the spill frame reaches
    # `REBUILD_SPILL_BYTES_PER_THREAD` bytes per hardware thread.
    k = kernel[(1, )](x, SIZE=SIZE, num_warps=1, generate_native_code=True, grf_mode='default')
    # CRI's auto-GRF upgrade targets 512-GRF; other targets use 256 (see the
    # retry_grf_mode_list branch in the Intel backend compiler).
    large_grf_flag = "-cl-intel-512-GRF-per-thread" if is_xpu_cri() else "-cl-intel-256-GRF-per-thread"
    if large_grf_flag in k.metadata.build_flags:
        return  # the rebuild fired, which is the whole assertion

    # Read the outcome before the predicate: `make_zebin` keeps the *adopted*
    # binary, so a successful rebuild lowers `k.kernel`'s spill below the gate and
    # asking the predicate first would skip exactly when it should assert. Reaching
    # here means no rebuild happened, so this spill is the one the gate acted on.
    zebin = tmp_path / "kernel.zebin"
    zebin.write_bytes(k.kernel)
    spill_size = extract_spill_size_from_zebin(str(zebin))
    # Only for the diagnostic below: the gate compares bytes. Mirrors
    # `Spills::slotsPerLane` in driver.c, whose truncation is what makes the per-lane
    # count a poor thing to gate on.
    spill_slots = spill_size // (4 * k.metadata.threads_per_warp)
    # The gate differs by driver line (issue #8106), so ask the predicate the
    # backend itself uses rather than re-deriving the rolling rule here.
    is_lts = XPUBackend.is_lts(k.metadata.target.arch.get("driver_version"))
    threshold = "any spill" if is_lts else f"{REBUILD_SPILL_BYTES_PER_THREAD} B/thread"
    observed = (f"{spill_size} B/thread, i.e. {spill_slots} dword-equivalents/lane at "
                f"SIMD{k.metadata.threads_per_warp}; rebuild at {threshold}")
    if not accepts_default_grf(spill_size, is_lts):
        pytest.fail(f"Gate should have rebuilt at large GRF but did not ({observed}).")
    pytest.skip(f"Kernel did not spill up to the threshold ({observed}); auto-large-GRF path was not "
                "exercised. Consider increasing SIZE.")


@pytest.mark.parametrize("generate_native_code", [False, True], ids=["load_binary", "make_zebin"])
@pytest.mark.parametrize("grf_mode", ["128", "256", "512", "auto"])
def test_explicit_grf_mode_uses_maximum_registers(device, grf_mode, generate_native_code):
    if grf_mode == "512" and not is_xpu_cri():
        pytest.xfail("512-GRF mode is only available on CRI")

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        x = tl.arange(0, SIZE)
        tl.store(X + x, x)

    SIZE = 128
    x = to_triton(numpy_random(SIZE, dtype_str="int32"), device=device, dst_type="int32")
    k = kernel[(1, )](x, SIZE=SIZE, num_warps=4, grf_mode=grf_mode, generate_native_code=generate_native_code)

    if XPUBackend.is_lts(k.metadata.target.arch.get("driver_version")):
        # The LTS translator does not enable SPV_INTEL_maximum_registers.
        assert "!MaximumRegisters" not in k.asm["llir"]
        assert "GRF" in k.metadata.build_flags
        return

    # Off LTS the GRF mode is requested by the kernel itself, not by an IGC build flag.
    assert "!MaximumRegisters" in k.asm["llir"]
    assert "GRF" not in k.metadata.build_flags
    if grf_mode != "auto":
        assert k.n_regs == int(grf_mode)
