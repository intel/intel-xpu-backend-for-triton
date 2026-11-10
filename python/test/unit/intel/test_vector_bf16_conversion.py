"""f32 -> bf16 RTNE conversions use the 8-wide `__spirv_ConvertFToBF16INTEL` when the consumer accepts it.

https://github.com/intel/intel-xpu-backend-for-triton/issues/8594

IGC keeps scalar conversion calls scalar, which raises register pressure in bf16 epilogues. The vector form is gated by
`allow_vector_bf16_conversion`, a hashed option resolved from a fail-closed consumer probe or the
`vector_bf16_conversion` override.
"""
import os
import shutil
import stat

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
import triton.backends.intel.spirv_probes as spirv_probes
from triton.backends.intel.compiler import XPUBackend
from triton.compiler.compiler import make_backend

pytestmark = pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                                reason="Intel XPU device not available")

VECTOR_CALL = "_Z27__spirv_ConvertFToBF16INTELDv8_f("


@pytest.fixture
def backend():
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert isinstance(backend, XPUBackend)
    return backend


def test_decision_is_part_of_the_cache_key(backend, monkeypatch):
    hashes = {}
    for accepted in (True, False):
        monkeypatch.setattr(spirv_probes, "ocloc_accepts_vector_bf16_conversion", lambda arch, a=accepted: a)
        options = backend.parse_options({"generate_native_code": True})
        assert options.allow_vector_bf16_conversion == accepted
        hashes[accepted] = options.hash()
    assert hashes[True] != hashes[False]
    assert backend.parse_options({"vector_bf16_conversion": "on"}).allow_vector_bf16_conversion
    assert not backend.parse_options({"vector_bf16_conversion": "off"}).allow_vector_bf16_conversion
    with pytest.raises(ValueError):
        backend.parse_options({"vector_bf16_conversion": "yes"})


# "failing" writes its output but exits nonzero; "silent" exits zero without output.
WRAPPERS = {"failing": 'while [ $# -gt 0 ]; do [ "$1" = -o ] && : > "$2"; shift; done; exit 1', "silent": "exit 0"}


@pytest.mark.parametrize("ocloc", ["missing", "failing", "silent"])
def test_native_probe_fails_closed(ocloc, tmp_path, monkeypatch):
    if ocloc in WRAPPERS:
        wrapper = tmp_path / "ocloc"
        wrapper.write_text("#!/bin/sh\n" + WRAPPERS[ocloc] + "\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not spirv_probes._ocloc_compiles("bmg", spirv_probes._VECTOR_BF16_PROBE_SPV)
    if shutil.which("ocloc", path=os.defpath + os.pathsep + "/usr/bin"):
        monkeypatch.setenv("PATH", os.defpath)
        assert spirv_probes._ocloc_compiles("bmg", spirv_probes._VECTOR_BF16_PROBE_SPV)


@triton.jit
def _to_bf16(X, Y, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offs, tl.load(X + offs).to(tl.bfloat16))


def _rtne_bf16_bits(x):
    """Independent RTNE f32 -> bf16 on the integer bits (non-NaN inputs)."""
    u = x.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def test_edge_values_and_override(device):
    bits = np.array([
        0x00000000, 0x80000000, 0x3F808000, 0x3F818000, 0xBF808000, 0x3F808001, 0x3F807FFF, 0x00000001, 0x007FFFFF,
        0x00800000, 0x7F7FFFFF, 0x7F7F7FFF, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFA00000
    ], dtype=np.uint32)
    x = torch.from_numpy(np.resize(bits, 4096).view(np.float32)).to(device)
    outs = {}
    for mode in ("auto", "off"):  # the same JIT function: the override must reach the warmed kernel's cache key
        y = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
        # 2048 elements over 4 warps: at least 16 per thread, so full groups of 8 exist.
        k = _to_bf16[(x.numel() // 2048, )](x, y, BLOCK=2048, num_warps=4, vector_bf16_conversion=mode)
        outs[mode] = (y.cpu().view(torch.int16).numpy().view(np.uint16), k)
    if not outs["auto"][1].metadata.allow_vector_bf16_conversion:
        pytest.skip("the consumer does not accept the vector conversion")
    assert VECTOR_CALL in outs["auto"][1].asm["llir"]
    assert VECTOR_CALL not in outs["off"][1].asm["llir"]
    xs = x.cpu().numpy()
    for got, _ in outs.values():
        nan = np.isnan(xs)
        np.testing.assert_array_equal(got[~nan], _rtne_bf16_bits(xs[~nan]))
        g = got[nan]
        assert ((g & 0x7F80) == 0x7F80).all() and ((g & 0x007F) != 0).all()  # still NaN
        assert ((g >> 15) == (xs[nan].view(np.uint32) >> 31)).all()  # sign kept


@pytest.mark.parametrize("mode", ["on", "off"])
def test_saved_ttgir_attribute_follows_the_option(mode, tmp_path, device):
    x = torch.zeros(2048, device=device)
    y = torch.empty(2048, dtype=torch.bfloat16, device=device)
    ttgir = _to_bf16.warmup(x, y, BLOCK=2048, num_warps=4, grid=(1, )).asm["ttgir"]
    attr = '"ttig.support_vector_bf16_conversion"'
    # A stale attribute in either direction: present for "off", absent for "on".
    ttgir = ttgir.replace("module attributes {", "module attributes {" + attr + ", ", 1) if mode == "off" else \
        ttgir.replace(attr + ", ", "").replace(", " + attr, "")
    path = tmp_path / "k.ttgir"
    path.write_text(ttgir)
    k = triton.compile(str(path), target=triton.runtime.driver.active.get_current_target(),
                       options={"vector_bf16_conversion": mode})
    assert (VECTOR_CALL in k.asm["llir"]) == (mode == "on")
