"""Unit test for `get_max_grf_mode()`'s "cri" vs. everything-else GRF policy.

https://github.com/intel/intel-xpu-backend-for-triton/issues/8074

`get_max_grf_mode()` is the single source of truth `parse_target` calls to
populate `dev_prop['max_grf_mode']`, from which every consumer (the ocloc
auto-large-GRF retry, `driver.c`'s JIT retry, and the `ttig.max_grf_mode`
module attribute read by `RegisterPressureAnalysis`) reaches it as
`opt.max_grf_mode` rather than re-deriving it. A driver- or out-of-tree
arch-module-supplied override participates via `parse_target`'s own
`tgt_prop.get('max_grf_mode', get_max_grf_mode(tgt_prop))` call -- the same
`tgt_prop.get(key, default)` idiom every other per-target capability in that
function uses, so it is not this function's own concern and has no dedicated
test here, matching its siblings. Before this test existed, nothing pinned
the actual policy ("512" on "cri", "256" everywhere else): the one hardcoded
assertion of "256" elsewhere in the suite (`test_native_code_generation.py`)
is `@pytest.mark.xfail(is_xpu_cri())`, i.e. inert on exactly the target this
policy treats differently.

Pure `dict -> str`, no device access, so this needs no XPU and no `device`
fixture.
"""
from types import SimpleNamespace

import pytest

import triton
import triton.language as tl
import triton.backends.intel.compiler as intel_compiler
from triton.backends.intel.compiler import get_grf_modes, get_max_grf_mode


def test_get_max_grf_mode_cri():
    assert get_max_grf_mode({"arch": "cri"}) == "512"


def test_get_max_grf_mode_non_cri():
    assert get_max_grf_mode({"arch": "bmg"}) == "256"
    assert get_max_grf_mode({"arch": "pvc"}) == "256"


def test_get_grf_modes_cri():
    assert get_grf_modes({"arch": "cri"}) == ("128", "160", "192", "256", "512")


def test_get_grf_modes_non_cri():
    assert get_grf_modes({"arch": "bmg"}) == ("128", "256")
    assert get_grf_modes({"arch": "pvc"}) == ("128", "256")


def test_get_grf_modes_includes_max_grf_mode():
    # The automatic escalation target must itself be a mode the target accepts.
    for arch in ("cri", "bmg", "pvc"):
        assert get_max_grf_mode({"arch": arch}) in get_grf_modes({"arch": arch})


@pytest.fixture
def option_backend(monkeypatch):
    monkeypatch.setattr(intel_compiler, "accepts_long_vector", lambda *args: True)
    monkeypatch.setattr(intel_compiler, "accepts_maximum_registers", lambda *args: True)
    return SimpleNamespace(
        properties={"core_clock_rate": 2000000, "is_lts": False, "max_grf_mode": "256"},
        target=SimpleNamespace(arch={"arch": "bmg", "device_id": 0}),
        device_arch="bmg",
    )


@pytest.mark.parametrize("requested", [None, "default", "auto", "128", "256"])
@pytest.mark.parametrize("forced", ["", "256"])
def test_force_grf_mode_overrides_kernel_options(monkeypatch, option_backend, requested, forced):
    monkeypatch.setenv("TRITON_XPU_FORCE_GRF_MODE", forced)
    supplied = {} if requested is None else {"grf_mode": requested}
    options = intel_compiler.XPUBackend.parse_options(option_backend, supplied)
    assert options.grf_mode == (forced or requested or "default")
    assert supplied == ({} if requested is None else {"grf_mode": requested})


def test_force_grf_mode_changes_options_hash(monkeypatch, option_backend):
    monkeypatch.delenv("TRITON_XPU_FORCE_GRF_MODE", raising=False)
    baseline = intel_compiler.XPUBackend.parse_options(option_backend, {"grf_mode": "128"})
    monkeypatch.setenv("TRITON_XPU_FORCE_GRF_MODE", "256")
    forced = intel_compiler.XPUBackend.parse_options(option_backend, {"grf_mode": "128"})
    assert forced.hash() != baseline.hash()


def test_force_grf_mode_rejects_invalid_value(monkeypatch, option_backend):
    monkeypatch.setenv("TRITON_XPU_FORCE_GRF_MODE", "128")
    with pytest.raises(ValueError, match="must be 256 or empty"):
        intel_compiler.XPUBackend.parse_options(option_backend, {})


@pytest.mark.parametrize("forced", ["", "256"])
def test_force_grf_mode_native_kernel(monkeypatch, forced):
    import torch

    if not torch.xpu.is_available():
        pytest.skip("Intel XPU device not available")
    monkeypatch.setenv("TRITON_XPU_FORCE_GRF_MODE", forced)

    @triton.jit
    def copy_kernel(x, y, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.store(y + offsets, tl.load(x + offsets))

    x = torch.arange(256, device="xpu", dtype=torch.float32)
    y = torch.empty_like(x)
    kernel = copy_kernel[(1, )](x, y, BLOCK=256, grf_mode="128")
    torch.xpu.synchronize()
    assert torch.equal(x, y)
    assert kernel.n_regs == int(forced or "128")
    print(f"GRF override smoke: forced={forced or 'off'}, actual={kernel.n_regs}, spills={kernel.n_spills}")
