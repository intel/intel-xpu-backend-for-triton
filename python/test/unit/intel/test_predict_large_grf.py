"""`TRITON_INTEL_PREDICT_LARGE_GRF`: build predicted rebuilds at large GRF first (native path only)."""
import dataclasses
import subprocess

import pytest
import torch
import triton
import triton.language as tl

import triton.backends.intel.compiler as intel_compiler
from triton._C.libtriton import intel, ir
from triton.backends.intel.driver import is_lts
from triton.runtime.errors import IntelGPUError

_HEAD = """
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
#bl3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#bdot0 = #ttg.dot_op<{opIdx = 0, parent = #bl3}>
#bdot1 = #ttg.dot_op<{opIdx = 1, parent = #bl3}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @k() {
"""
_TAIL = """    tt.return
  }
}
"""


def _dot(m, n, k, batch=""):
    a, b, c = (f"tensor<{batch}{m}x{k}xf16, #{'b' if batch else ''}dot0>",
               f"tensor<{batch}{k}x{n}xf16, #{'b' if batch else ''}dot1>",
               f"tensor<{batch}{m}x{n}xf32, #{'bl3' if batch else 'dpas'}>")
    tag = f"{batch.replace('x', '_')}{m}_{n}_{k}"
    return (f"    %a{tag} = arith.constant dense<1.0> : {a}\n    %b{tag} = arith.constant dense<1.0> : {b}\n"
            f"    %c{tag} = arith.constant dense<0.0> : {c}\n"
            f"    %d{tag} = tt.dot %a{tag}, %b{tag}, %c{tag} : {a} * {b} -> {c}\n")


# Bytes of A + B + result: (8x64, 64x16) is 1024 + 2048 + 512; batched (2x8x16, 2x16x16) is 512 + 1024 + 1024.
LARGEST_IN_MIDDLE = _HEAD + _dot(8, 16, 16) + _dot(8, 16, 64) + _dot(8, 16, 32) + _TAIL
# The batched dot wins only through its batch, and the later dot of equal work does not displace it.
BATCH_THEN_TIE = _HEAD + _dot(8, 16, 16) + _dot(8, 16, 16, batch="2x") + _dot(8, 32, 16) + _TAIL


def _module(tmp_path, src):
    path = tmp_path / "module.ttgir"
    path.write_text(src)
    ctx = ir.context()
    ir.load_dialects(ctx)
    intel.load_dialects(ctx)
    return ir.parse_mlir_module(str(path), ctx), ctx


@pytest.mark.parametrize("src, expected", [(LARGEST_IN_MIDDLE, 3584), (BATCH_THEN_TIE, 2560)],
                         ids=["largest_in_middle", "batch_then_tie"])
def test_largest_dot_tile_bytes(tmp_path, src, expected):
    mod, _ctx = _module(tmp_path, src)
    assert intel.largest_dot_tile_bytes(mod) == expected


# BATCH_THEN_TIE's 2560 B over 4 warps is 640 B per hardware thread.
@pytest.mark.parametrize("threshold, change, expected", [
    (640, {}, True),
    (641, {}, False),
    (640, {"predict_large_grf": False}, False),
    (640, {"generate_native_code": False}, False),
    (640, {"is_lts": True}, False),
    (640, {"arch": "ptl_h"}, False),
    (640, {"disable_igc_opt": True}, False),
])
def test_predicts_large_grf(tmp_path, monkeypatch, threshold, change, expected):
    mod, _ctx = _module(tmp_path, BATCH_THEN_TIE)
    change = dict(change)
    monkeypatch.setattr(intel_compiler, "LARGE_GRF_TILE_BYTES", threshold)
    monkeypatch.setattr(intel_compiler.XPUBackend, "device_arch", change.pop("arch", "bmg"), raising=False)
    monkeypatch.setenv("TRITON_INTEL_DISABLE_IGC_OPT", "1" if change.pop("disable_igc_opt", False) else "0")
    fields = {"num_warps": 4, "predict_large_grf": True, "generate_native_code": True, "is_lts": False, **change}
    opt = dataclasses.replace(intel_compiler.XPUOptions(), **fields)
    assert intel_compiler.XPUBackend.predicts_large_grf(mod, opt) == expected


def _make_dot_kernel():

    @triton.jit
    def _dot_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        a = tl.load(a_ptr + offs[:, None] * N + offs[None, :])
        b = tl.load(b_ptr + offs[:, None] * N + offs[None, :])
        tl.store(c_ptr + offs[:, None] * N + offs[None, :], tl.dot(a, b))

    return _dot_kernel


@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="Intel XPU device not available")
@pytest.mark.parametrize("failure, expected_builds", [(None, 1), ("ocloc", 2), ("degenerate", 2)])
def test_large_grf_first_build(monkeypatch, fresh_triton_cache, failure, expected_builds):
    """The large build comes first and is kept; if it fails, today's default-first path builds the kernel."""
    target = triton.runtime.driver.active.get_current_target()
    if is_lts(target.arch.get("driver_version")) or intel_compiler.XPUBackend(target).device_arch not in ("bmg", "lnl"):
        pytest.skip("validated on Xe2 rolling drivers only")
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "1")
    monkeypatch.setenv("TRITON_INTEL_PREDICT_LARGE_GRF", "1")
    monkeypatch.setattr(intel_compiler, "LARGE_GRF_TILE_BYTES", 0)
    options, real = [], subprocess.check_output

    def spy(cmd, *args, **kwargs):
        if not (isinstance(cmd, list) and "ocloc" in str(cmd[0])):
            return real(cmd, *args, **kwargs)
        options.append(cmd[-1])
        if failure == "ocloc" and len(options) == 1:
            raise subprocess.CalledProcessError(1, cmd, output="simulated large-GRF build failure")
        return real(cmd, *args, **kwargs)

    def spill_size(fbin):  # a zero-exit build with no .text/.symtab raises here
        if failure == "degenerate" and len(options) == 1:
            raise IntelGPUError("simulated degenerate zebin")
        return 0

    monkeypatch.setattr(intel_compiler.subprocess, "check_output", spy)
    monkeypatch.setattr(intel_compiler, "extract_spill_size_from_zebin", spill_size)
    a = torch.randn(32, 32, device="xpu", dtype=torch.float16)
    b = torch.randn(32, 32, device="xpu", dtype=torch.float16)
    c = torch.empty(32, 32, device="xpu", dtype=torch.float32)
    kernel = _make_dot_kernel()[(1, )](a, b, c, N=32)
    torch.testing.assert_close(c, a.float() @ b.float(), atol=1e-2, rtol=1e-2)
    large = f"-cl-intel-{kernel.metadata.max_grf_mode}-GRF-per-thread"
    assert len(options) == expected_builds and large in options[0], options
    assert (large in kernel.metadata.build_flags) == (failure is None)
