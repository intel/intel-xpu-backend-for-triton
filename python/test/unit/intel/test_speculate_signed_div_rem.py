"""Tests for the `triton-intel-speculate-signed-div-rem` pass.

The AxisInfo constancy/contiguity/divisibility deductions for `//` and `%` are
only sound for a non-negative dividend. The pass converts the signed operations
those deductions would have fired on to their unsigned form, guarded by a
`tt.assert` on the dividend. See `TRITON_SPECULATE_SIGNED_DIV_REM`.
"""
import os
import subprocess
import sys
import tempfile

import pytest
import torch

import triton
import triton.language as tl


def is_xpu():
    return triton.runtime.driver.active.get_current_target().backend == "xpu"


pytestmark = pytest.mark.skipif(not is_xpu(), reason="XPU-specific pass")

N, D = 128, 16


@triton.jit
def div_and_rem_kernel(in_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, N)
    # Offsetting by a loaded multiple of 2*N keeps the index contiguous and
    # divisible by N, so the deduction fires, while making its sign unprovable.
    # This is the NATTEN shape in miniature.
    idx = offs + tl.load(in_ptr) * (N * 2)
    tl.store(out_ptr + offs, idx // D + idx % D)


@triton.jit
def loaded_dividend_kernel(in_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    # A loaded dividend has contiguity and divisibility 1, so the deduction
    # could not have concluded anything and no assertion is warranted - even
    # though the values may well be negative.
    tl.store(out_ptr + offs, x // D + x % D)


@triton.jit
def negative_dividend_kernel(in_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, N)
    # `in_ptr` is unused: this kernel only exists to give the pass a dividend
    # whose every element is provably negative.
    idx = offs - N
    tl.store(out_ptr + offs, idx // D + idx % D)


def compile_ttir(kernel):
    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*i32",
            "out_ptr": "*i32",
            "N": "constexpr",
            "D": "constexpr",
        },
        constexprs={"N": N, "D": D},
    )
    target = triton.runtime.driver.active.get_current_target()
    return triton.compile(src, target=target).asm["ttir"]


def test_speculates_and_shares_one_assert():
    ttir = compile_ttir(div_and_rem_kernel)
    assert "arith.divsi" not in ttir and "arith.remsi" not in ttir, ttir
    assert "arith.divui" in ttir and "arith.remui" in ttir, ttir
    # The division and the remainder share a dividend, so one assertion covers
    # both.
    assert ttir.count("tt.assert") == 1, ttir


def test_no_assert_for_loaded_dividend():
    # Regression test against frontend-check-style false positives: this is the
    # `test_core.py::test_bin_op` shape, where an unconditional check would
    # abort for no benefit.
    ttir = compile_ttir(loaded_dividend_kernel)
    assert "tt.assert" not in ttir, ttir
    assert "arith.divsi" in ttir and "arith.remsi" in ttir, ttir


def test_no_assert_for_provably_negative_dividend():
    # Speculating here would change the result and emit an assertion that fails
    # on every launch, so the kernel must be left signed.
    ttir = compile_ttir(negative_dividend_kernel)
    assert "tt.assert" not in ttir, ttir
    assert "arith.divsi" in ttir and "arith.remsi" in ttir, ttir


def test_knob_disables_the_pass(monkeypatch):
    monkeypatch.setenv("TRITON_SPECULATE_SIGNED_DIV_REM", "0")
    triton.knobs.refresh_knobs()
    try:
        ttir = compile_ttir(div_and_rem_kernel)
    finally:
        monkeypatch.delenv("TRITON_SPECULATE_SIGNED_DIV_REM")
        triton.knobs.refresh_knobs()
    assert "tt.assert" not in ttir, ttir
    assert "arith.divsi" in ttir and "arith.remsi" in ttir, ttir


def test_speculated_result_is_correct():
    idx = torch.zeros(1, dtype=torch.int32, device="xpu")
    out = torch.empty(N, dtype=torch.int32, device="xpu")
    div_and_rem_kernel[(1, )](idx, out, N=N, D=D)
    offs = torch.arange(N, dtype=torch.int32, device="xpu")
    torch.testing.assert_close(out, offs // D + offs % D)


# The assertion aborts the process, so it has to be checked out of band.
_NEGATIVE_DIVIDEND_PROGRAM = """
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(in_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, N)
    # The sign depends on the loaded value, so no prover can rule out a negative
    # dividend and the pass speculates. The load is negative at runtime, so the
    # assertion must fire.
    idx = offs + tl.load(in_ptr) * (N * 2)
    tl.store(out_ptr + offs, idx // D + idx % D)


idx = torch.full((1, ), -1, dtype=torch.int32, device="xpu")
out = torch.empty(128, dtype=torch.int32, device="xpu")
kernel[(1, )](idx, out, N=128, D=16)
torch.xpu.synchronize()
"""


def test_negative_dividend_aborts():
    with tempfile.TemporaryDirectory() as tmpdir:
        script = os.path.join(tmpdir, "negative_dividend.py")
        with open(script, "w") as f:
            f.write(_NEGATIVE_DIVIDEND_PROGRAM)
        env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(tmpdir, "cache"))
        result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    # The message must name the escape hatch that is actually correct.
    assert "TRITON_SPECULATE_SIGNED_DIV_REM=0" in output, output
