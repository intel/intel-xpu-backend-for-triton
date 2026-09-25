"""EU thread-pause primitive (`tl.extra.intel.thread_pause`, issue #8092)."""

import pytest
import torch

import triton
import triton.language as tl

from triton.backends.intel.compiler import XPUBackend, supports_thread_pause
from triton.language.extra.intel.utils import _pause_duration_ns

MAX_UNITS = 31
UNIT_CYCLES = 32
# vISA destination of the pause: element 4 of the timestamp register.
PAUSE_DST = "%tsc(0,4)"


def pause_asm(units):
    return f"mov (M1_NM, 1) {PAUSE_DST}<1> {units * UNIT_CYCLES:#x}:ud"


@pytest.fixture
def properties():
    return XPUBackend(triton.runtime.driver.active.get_current_target()).properties


@triton.jit
def _pause_kernel(out_ptr, n_iter, UNITS: tl.constexpr):
    for _ in range(n_iter):
        tl.extra.intel.thread_pause(UNITS)
    tl.store(out_ptr, n_iter)


@triton.jit
def _timed_pause_kernel(out_ptr, n_iter, UNITS: tl.constexpr):
    start = tl.extra.intel.globaltimer()
    for _ in range(n_iter):
        tl.extra.intel.thread_pause(UNITS)
    tl.store(out_ptr, tl.extra.intel.globaltimer() - start)


@triton.jit
def _pause_duration_kernel(out_ptr, UNITS: tl.constexpr):
    tl.store(out_ptr, tl.extra.intel.thread_pause_duration_ns(UNITS))


# SYCL `architecture` values, from sycl/ext/oneapi/experimental/device_architecture.def.
PVC = 0x000000030F000700
PVC_VG = 0x000000030F400700
BMG_G21 = 0x0000000500400400
ACM_G11 = 0x000000030E000500
UNKNOWN = 0x9900000000000000


@pytest.mark.parametrize(
    "device_arch, architecture, supported",
    [
        ("bmg", BMG_G21, True),
        # An architecture SYCL knows but the parser does not name.
        ("", ACM_G11, True),
        ("pvc", PVC, False),
        # The Xe-HPC variant the parser does not name.
        ("", PVC_VG, False),
        # SYCL cannot identify the device, so Xe-HPC is not ruled out.
        ("unknown", UNKNOWN, False),
        # Neither can a target that does not report its architecture.
        ("", 0, False),
        # `TRITON_INTEL_DEVICE_ARCH=pvc` set on another device.
        ("pvc", BMG_G21, False),
    ],
)
def test_supports_thread_pause(device_arch, architecture, supported):
    assert supports_thread_pause(device_arch, architecture) is supported


@pytest.mark.parametrize(
    "units, has_thread_pause, clock_rate_khz, expected",
    [
        # Arc B580 reports 2.85 GHz: 31 units * 32 cycles = 992 cycles = 348 ns.
        (MAX_UNITS, True, 2850000, 348),
        (4, True, 2850000, 44),
        # No pause counter, and an unknown clock rate, both mean "cannot delay".
        (MAX_UNITS, False, 2850000, 0),
        (MAX_UNITS, True, 0, 0),
    ],
)
def test_pause_duration_ns(units, has_thread_pause, clock_rate_khz, expected):
    assert _pause_duration_ns(units, has_thread_pause, clock_rate_khz) == expected


@pytest.mark.parametrize("units", [4, MAX_UNITS])
def test_thread_pause_codegen(device, properties, units):
    out = torch.zeros(1, dtype=torch.int32, device=device)
    compiled = _pause_kernel[(1, )](out, 8, UNITS=units, num_warps=1)
    assert out.item() == 8

    llir = compiled.asm["llir"]
    if _pause_duration_ns(units, properties["has_thread_pause"], properties["core_clock_rate"]) > 0:
        # Only that the pause is emitted with the requested delay; that it runs
        # on every iteration is what `test_thread_pause_delays` measures.
        assert pause_asm(units) in llir, llir
    else:
        assert PAUSE_DST not in llir, llir


def test_thread_pause_duration_ns(device, properties):
    out = torch.zeros(1, dtype=torch.int32, device=device)
    _pause_duration_kernel[(1, )](out, UNITS=MAX_UNITS, num_warps=1)
    assert out.item() == _pause_duration_ns(MAX_UNITS, properties["has_thread_pause"], properties["core_clock_rate"])


def test_thread_pause_delays(device, properties):
    per_pause_ns = _pause_duration_ns(MAX_UNITS, properties["has_thread_pause"], properties["core_clock_rate"])
    if per_pause_ns == 0:
        pytest.skip("target cannot delay: no EU pause counter or unknown core clock rate")

    n_iter = 4096
    elapsed = torch.zeros(1, dtype=torch.int64, device=device)
    _timed_pause_kernel[(1, )](elapsed, n_iter, UNITS=MAX_UNITS, num_warps=1)

    # The duration assumes the peak core clock, so the real delay moves with
    # clock scaling; a loose lower bound keeps this robust while still failing if
    # the pause does not delay at all.
    assert elapsed.item() >= n_iter * per_pause_ns // 2


@pytest.mark.parametrize("units", [0, 3, MAX_UNITS + 1, 2.0])
def test_thread_pause_rejects_bad_units(device, units):
    out = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(triton.CompilationError, match="thread pause units"):
        _pause_kernel[(1, )](out, 1, UNITS=units, num_warps=1)
