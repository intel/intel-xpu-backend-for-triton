"""The per-chip module hook: `XPUBackend` uses `arch/<name>.py` for the chip it compiles for. No device access: the
chip is named through `TRITON_INTEL_DEVICE_ARCH` and the targets are synthetic."""

import importlib
import sys
import textwrap

import pytest

import triton.backends.intel.arch as arch_pkg
from triton.backends.compiler import GPUTarget
from triton.backends.intel.compiler import XPUBackend

CHIP = "test_arch_module_chip"
BROKEN_CHIP = "test_arch_module_broken_chip"
TARGET = GPUTarget("xpu", {"driver_version": "1.6.35096+9", "__intel_already_queried_extensions__": True}, warp_size=32)


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(XPUBackend, "arch_to_impl", {})


@pytest.fixture
def chip_modules(monkeypatch, tmp_path):
    """`arch/<CHIP>.py` and a `arch/<BROKEN_CHIP>.py` that fails to import, added to the `arch` package's path."""
    (tmp_path / f"{CHIP}.py").write_text(
        textwrap.dedent("""
        from ..compiler import XPUBackend


        class XPUBackendImpl(XPUBackend):

            def parse_target(self, tgt_prop):
                dev_prop = super().parse_target(tgt_prop)
                dev_prop['set_by_chip_module'] = True
                return dev_prop
    """))
    (tmp_path / f"{BROKEN_CHIP}.py").write_text("import test_arch_module_missing_dependency\n")
    importlib.invalidate_caches()
    monkeypatch.setattr(arch_pkg, "__path__", [*arch_pkg.__path__, str(tmp_path)])
    yield
    sys.modules.pop(f"{arch_pkg.__name__}.{CHIP}", None)
    if hasattr(arch_pkg, CHIP):
        delattr(arch_pkg, CHIP)


def test_chip_without_module_keeps_base_properties(monkeypatch):
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", "no_such_chip")
    backend = XPUBackend(TARGET)
    assert type(backend).__bases__ == (XPUBackend, )
    assert backend.properties == XPUBackend.parse_target(backend, TARGET.arch)


def test_backend_follows_a_device_arch_change(monkeypatch, chip_modules):
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", CHIP)
    first = XPUBackend(TARGET)
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", "no_such_chip")
    second = XPUBackend(TARGET)
    assert type(first).__module__ == f"{arch_pkg.__name__}.{CHIP}"
    assert (first.device_arch, first.properties.get("set_by_chip_module")) == (CHIP, True)
    assert (second.device_arch, "set_by_chip_module" in second.properties) == ("no_such_chip", False)


def test_chip_module_that_fails_to_import_raises(monkeypatch, chip_modules):
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", BROKEN_CHIP)
    with pytest.raises(ModuleNotFoundError, match="test_arch_module_missing_dependency"):
        XPUBackend(TARGET)
    assert BROKEN_CHIP not in XPUBackend.arch_to_impl
