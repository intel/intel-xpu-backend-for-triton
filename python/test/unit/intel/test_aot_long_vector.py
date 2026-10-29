"""AOT (`generate_native_code`) must not emit `SPV_EXT_long_vector` for an `ocloc` that predates it.

https://github.com/intel/intel-xpu-backend-for-triton/issues/8376

The AOT path hands the SPIR-V to `ocloc`, whose IGC is independent of the installed
driver. An `ocloc` older than `SPV_EXT_long_vector` (e.g. the one in an older oneAPI)
rejects the module with "uses unknown extension 'SPV_EXT_long_vector'", while the
driver-based JIT path accepts it. The old `ocloc` is emulated with a wrapper that
rejects any SPIR-V mentioning the extension and otherwise runs the real `ocloc`.
"""
import os
import shutil
import stat
import sys

import pytest
import torch
import triton
import triton.language as tl
import triton.backends.intel.ocloc_utils as ocloc_utils
from triton.backends.intel.compiler import XPUBackend
from triton.backends.intel.ocloc_utils import ocloc_accepts_long_vector
from triton.compiler.compiler import make_backend

pytestmark = [
    pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                       reason="Intel XPU device not available"),
    pytest.mark.skipif(shutil.which("ocloc") is None, reason="`ocloc` not found"),
    pytest.mark.skipif(sys.platform == "win32", reason="the `ocloc` wrapper is a shell script"),
]

REJECT = """#!/bin/sh
for f in "$@"; do
  if [ -f "$f" ] && grep -q SPV_EXT_long_vector "$f"; then
    echo "InvalidModule: Invalid SPIR-V module: input SPIR-V module uses unknown extension 'SPV_EXT_long_vector'"
    exit 10
  fi
done
exec "{real}" "$@"
"""


@pytest.fixture
def old_ocloc(tmp_path, monkeypatch):
    wrapper = tmp_path / "ocloc"
    wrapper.write_text(REJECT.format(real=shutil.which("ocloc")))
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    ocloc_accepts_long_vector.cache_clear()
    yield
    ocloc_accepts_long_vector.cache_clear()


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.delenv("TRITON_XPU_GEN_NATIVE_CODE", raising=False)
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert isinstance(backend, XPUBackend)
    if backend.properties["is_lts"]:
        pytest.skip("LTS driver never uses `SPV_EXT_long_vector`")
    return backend


@triton.jit
def _dot_kernel(A, B, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = tl.load(A + rm[:, None] * BK + rk[None, :])
    b = tl.load(B + rk[:, None] * BN + rn[None, :])
    tl.store(C + rm[:, None] * BN + rn[None, :], tl.dot(a, b))


def _run(device, **options):
    a = torch.randn(64, 32, dtype=torch.float16, device=device)
    b = torch.randn(32, 64, dtype=torch.float16, device=device)
    c = torch.empty(64, 64, dtype=torch.float32, device=device)
    kernel = _dot_kernel[(1, )](a, b, c, BM=64, BN=64, BK=32, **options)
    torch.testing.assert_close(c, torch.matmul(a.float(), b.float()), rtol=1e-2, atol=1e-2)
    return kernel.asm["spv"]


@pytest.mark.parametrize("native", [False, True])
def test_long_vector_follows_consumer(native, backend, old_ocloc, device):
    spv = _run(device, generate_native_code=native)
    # JIT: the driver consumes the module. AOT: the (old) `ocloc` does.
    assert (b"SPV_EXT_long_vector" in spv) == (not native)
    assert (b"SPV_INTEL_vector_compute" in spv) == native


def test_probe_result_is_part_of_the_cache_key(backend, monkeypatch):
    hashes = {}
    for accepted in (True, False):
        monkeypatch.setattr(ocloc_utils, "ocloc_accepts_long_vector", lambda arch, accepted=accepted: accepted)
        options = backend.parse_options({"generate_native_code": True})
        assert options.allow_long_vector == accepted
        hashes[accepted] = options.hash()
    assert hashes[True] != hashes[False]
