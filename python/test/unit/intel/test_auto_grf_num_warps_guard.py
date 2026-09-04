"""Regression test for the auto-GRF-256 upgrade num_warps guard.

https://github.com/intel/intel-xpu-backend-for-triton/issues/7450

256-GRF mode doubles the registers per hardware thread, which halves the
maximum work-group size the device can launch. The explicit `grf_mode='256'`
path already refuses `num_warps > 32` (a 64-warp / 1024-work-item group does
not fit at 256-GRF on BMG), but the *automatic* large-GRF upgrade did not.
A `num_warps > 32` kernel that spilled enough to trigger the silent upgrade
therefore produced exactly the config the explicit path forbids and failed to
launch with the raw `ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION`.

The fix drops the large-GRF entries from `make_zebin`'s retry list when
`num_warps > 32`, so both automatic upgrade triggers are covered:

  * the spill-based upgrade (`spill_size > MAX_REG_SPILL_PER_LANE * threads_per_warp`), and
  * the build-failure retry (e.g. the LTS2 degenerate-zebin case),

keeping the working — if slower, spilling — default-GRF binary in both cases.
`num_warps <= 32` kernels are unaffected.

The AOT sites are made deterministically reachable by stubbing the spill-size
probe. A JIT (`driver.c` / `load_binary`) regression test is included too: there
the large-GRF recompile fails to build and the runtime must keep the original
working binary rather than swapping in the null handles, which used to abort the
process with `UR_RESULT_ERROR_INVALID_NULL_HANDLE`.
"""
import shutil
import subprocess

import pytest
import torch
import triton
import triton.language as tl

import triton.backends.intel.compiler as intel_compiler
from triton._internal_testing import is_xpu_cri
from triton.runtime.errors import IntelGPUError

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)


def _has_ocloc() -> bool:
    """The AOT auto-upgrade path shells out to `ocloc compile`."""
    return shutil.which("ocloc") is not None


# The guard's threshold is a work-group-size limit, so the launch geometry has to
# be exact: these tests pin `warp_size` and vary `num_warps` around 32.
WARP_SIZE = 16


def _unsupported_reason(num_warps: int) -> str | None:
    """Why this device cannot launch `num_warps` sub-groups of `WARP_SIZE`.

    `num_warps = 64` at `warp_size = 16` is a 1024-work-item group. A device that
    lacks sub-group size 16, or caps work-groups below the requested size, would
    fail these tests for reasons unrelated to the GRF guard, so skip instead.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return "Intel XPU device not available"
    driver = triton.runtime.driver.active
    props = driver.utils.get_device_properties(driver.get_current_device())
    if WARP_SIZE not in props["sub_group_sizes"]:
        return f"device does not support sub-group size {WARP_SIZE} (has {props['sub_group_sizes']})"
    if num_warps * WARP_SIZE > props["max_work_group_size"]:
        return (f"device max_work_group_size {props['max_work_group_size']} < "
                f"{num_warps} warps x {WARP_SIZE}")
    return None


def _requires_warps(num_warps: int):
    """Skip marker for a test that launches `num_warps` x `WARP_SIZE` work-items."""
    reason = _unsupported_reason(num_warps)
    return pytest.mark.skipif(reason is not None, reason=reason or "")


def _make_add1():
    """Build a *fresh* @triton.jit kernel per test.

    `JITFunction` memoizes compiled kernels in-process (`device_caches`) keyed on
    signature + options, and `fresh_triton_cache` only isolates the on-disk cache.
    Sharing one module-level kernel would let an earlier test's compilation
    satisfy a later one, so `make_zebin` would never re-run and the stubs/spies
    below would silently observe nothing.
    """

    @triton.jit
    def _add1(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)

    return _add1


def _launch(num_warps, n=4096, kernel=None):
    kernel = kernel if kernel is not None else _make_add1()
    x = torch.arange(n, device='xpu', dtype=torch.float32)
    y = torch.empty_like(x)
    kernel[(1, )](x, y, n, BLOCK=n, num_warps=num_warps, warp_size=WARP_SIZE)
    torch.xpu.synchronize()
    return x, y


@triton.jit
def _heavy_spill(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """A kernel with enough live values to spill at a large BLOCK, so the
    runtime spill-based recompile (driver.c, any nonzero spill bytes) fires."""
    offs = tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    b = tl.load(b_ptr + offs, mask=m, other=0.0)
    c = tl.load(c_ptr + offs, mask=m, other=0.0)
    d = tl.load(d_ptr + offs, mask=m, other=0.0)
    acc = a
    # Keep a wide set of intermediates live to force register spilling.
    for i in tl.static_range(64):
        a = a * 1.001 + b
        b = b * 1.002 + c
        c = c * 1.003 + d
        d = d * 1.004 + a
        acc = acc + a + b + c + d
    tl.store(out_ptr + offs, acc, mask=m)


@_requires_warps(64)
@pytest.mark.skipif(not _has_ocloc(), reason="`ocloc` not on PATH — AOT path can't be exercised")
def test_auto_grf_upgrade_skipped_for_num_warps_64(monkeypatch, fresh_triton_cache):
    """A num_warps > 32 kernel hitting the spill-based auto-256-GRF upgrade must
    skip the upgrade and keep the working 128-GRF binary, launching correctly
    instead of failing with the raw INVALID_GROUP_SIZE_DIMENSION."""
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "1")
    # Force the spill-based auto-upgrade to fire regardless of the real kernel.
    monkeypatch.setattr(intel_compiler, "extract_spill_size_from_zebin", lambda _f: 1 << 20)

    x, y = _launch(num_warps=64)
    assert torch.allclose(y, x + 1.0)


@_requires_warps(32)
@pytest.mark.skipif(not _has_ocloc(), reason="`ocloc` not on PATH — AOT path can't be exercised")
def test_auto_grf_upgrade_still_applies_for_num_warps_32(monkeypatch, fresh_triton_cache):
    """num_warps <= 32 is unaffected: a spilling kernel still auto-upgrades to
    256-GRF and launches correctly (behavior unchanged)."""
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "1")
    monkeypatch.setattr(intel_compiler, "extract_spill_size_from_zebin", lambda _f: 1 << 20)

    x, y = _launch(num_warps=32)
    assert torch.allclose(y, x + 1.0)


def _spy_on_ocloc(monkeypatch):
    """Record every `ocloc` command line `make_zebin` issues.

    Returns the list that accumulates them. Used to tell whether a large-GRF
    retry was attempted: a retry shows up as an extra invocation whose
    `-options` argument carries the large-GRF flag.
    """
    calls = []
    real_check_output = subprocess.check_output

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and 'ocloc' in str(cmd[0]):
            calls.append(list(cmd))
        return real_check_output(cmd, *args, **kwargs)

    monkeypatch.setattr(intel_compiler.subprocess, "check_output", spy)
    return calls


# `make_zebin` picks the large-GRF retry flag per target: CRI upgrades straight to
# 512-GRF, everything else to 256-GRF. Match whichever this device would use, so
# the assertions below track the guard and not the flag choice.
LARGE_GRF_FLAG = "-cl-intel-512-GRF-per-thread" if is_xpu_cri() else "-cl-intel-256-GRF-per-thread"


def _large_grf_retries(calls) -> int:
    return sum(1 for c in calls if any(LARGE_GRF_FLAG in str(a) for a in c))


def _stub_degenerate_first_build(monkeypatch):
    """Make the spill-size probe raise for the *default*-GRF build only.

    Simulates the documented LTS2 degenerate-zebin case (no `.text`/`.symtab`),
    which is what drives `make_zebin` into its `except` branch and on to the
    large-GRF retry. `make_zebin` probes spills on every retry iteration, so the
    stub must let the large-GRF build through — otherwise it would also sabotage
    the retry it is meant to observe.
    """
    calls = {"n": 0}

    def raise_degenerate_first(fbin):
        calls["n"] += 1
        if calls["n"] == 1:
            raise IntelGPUError("simulated degenerate zebin (no .text/.symtab)")
        return 0

    monkeypatch.setattr(intel_compiler, "extract_spill_size_from_zebin", raise_degenerate_first)


@_requires_warps(64)
@pytest.mark.skipif(not _has_ocloc(), reason="`ocloc` not on PATH — AOT path can't be exercised")
def test_exception_retry_not_attempted_for_num_warps_64(monkeypatch, fresh_triton_cache):
    """The AOT build-exception retry must not upgrade a num_warps > 32 kernel.

    Simulates the documented LTS2 degenerate-zebin case (the spill probe raising
    `IntelGPUError`) to reach `make_zebin`'s `except` block. Without the guard,
    the retry list still holds the large-GRF flag, so `ocloc` is re-run with it
    and produces exactly the unlaunchable config the explicit
    `grf_mode='256'`/`'512'` paths forbid. With the guard, no large-GRF retry is
    issued and the original error surfaces instead.
    """
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "1")
    _stub_degenerate_first_build(monkeypatch)
    calls = _spy_on_ocloc(monkeypatch)

    # The degenerate-zebin error is not recoverable for num_warps > 32, so
    # compilation fails — but it must fail *without* a large-GRF retry.
    with pytest.raises(IntelGPUError):
        _launch(num_warps=64)

    assert _large_grf_retries(calls) == 0, (f"no {LARGE_GRF_FLAG} retry should be attempted for num_warps > 32; "
                                            f"got ocloc calls: {calls}")


@_requires_warps(32)
@pytest.mark.skipif(not _has_ocloc(), reason="`ocloc` not on PATH — AOT path can't be exercised")
def test_exception_retry_still_attempted_for_num_warps_32(monkeypatch, fresh_triton_cache):
    """Counterpart to the above: num_warps <= 32 still gets the large-GRF retry,
    so the guard didn't disable the recovery path it is meant to preserve."""
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "1")
    _stub_degenerate_first_build(monkeypatch)
    calls = _spy_on_ocloc(monkeypatch)

    # The retry re-compiles successfully here (the zebin itself is fine; only the
    # first spill probe was stubbed to raise), so the launch succeeds.
    x, y = _launch(num_warps=32)
    assert torch.allclose(y, x + 1.0)
    assert _large_grf_retries(calls) == 1, (f"expected exactly one {LARGE_GRF_FLAG} retry for num_warps <= 32; "
                                            f"got: {calls}")


@_requires_warps(64)
def test_jit_spill_recompile_num_warps_64_does_not_crash(monkeypatch, fresh_triton_cache):
    """JIT path (driver.c load_binary): a genuinely spilling num_warps > 32
    kernel triggers the runtime large-GRF recompile, which fails to build. The
    null handles must not be swapped in and reach make_kernel_bundle (which
    aborted the process with UR_RESULT_ERROR_INVALID_NULL_HANDLE); the working
    default-GRF binary is kept and the kernel launches correctly."""
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "0")  # force JIT / load_binary path

    n = 65536
    a, b, c, d = (torch.randn(n, device='xpu', dtype=torch.float32) for _ in range(4))
    out = torch.empty(n, device='xpu', dtype=torch.float32)

    # num_warps=64 (1024 work-items) spills hard at 128-GRF, so the runtime
    # recompiles at 256-GRF — which fails because the group no longer fits.
    _heavy_spill[(1, )](a, b, c, d, out, n, BLOCK=n, num_warps=64, warp_size=WARP_SIZE)
    torch.xpu.synchronize()
    assert torch.isfinite(out).all()
