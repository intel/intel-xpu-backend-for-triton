"""Tests for the `triton-intel-speculate-signed-div-rem` pass.

The AxisInfo constancy/contiguity/divisibility deductions for `//` and `%` are
only sound for a non-negative dividend. The pass converts the signed operations
those deductions would have fired on to their unsigned form, guarded by a
`tt.assert` on the dividend. See `TRITON_INTEL_SPECULATE_SIGNED_DIV_REM`.
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

N, D, STEPS = 128, 16, 4


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


@triton.jit
def loop_div_and_rem_kernel(in_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr, STEPS: tl.constexpr):
    offs = tl.arange(0, N)
    acc = tl.zeros((N, ), dtype=tl.int32)
    for i in range(STEPS):
        # The dividend is loaded afresh on every iteration, so its check varies
        # per iteration and cannot simply be hoisted out: it has to be folded
        # into a loop-carried flag.
        idx = offs + tl.load(in_ptr + i) * (N * 2)
        acc += idx // D + idx % D
    tl.store(out_ptr + offs, acc)


def compile_ttir(kernel, **constexprs):
    constexprs = {"N": N, "D": D, **constexprs}
    src = triton.compiler.ASTSource(
        fn=kernel,
        signature={
            "in_ptr": "*i32",
            "out_ptr": "*i32",
            **{name: "constexpr"
               for name in constexprs},
        },
        constexprs=constexprs,
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
    monkeypatch.setenv("TRITON_INTEL_SPECULATE_SIGNED_DIV_REM", "0")
    triton.knobs.refresh_knobs()
    try:
        ttir = compile_ttir(div_and_rem_kernel)
    finally:
        monkeypatch.delenv("TRITON_INTEL_SPECULATE_SIGNED_DIV_REM")
        triton.knobs.refresh_knobs()
    assert "tt.assert" not in ttir, ttir
    assert "arith.divsi" in ttir and "arith.remsi" in ttir, ttir


def test_speculated_result_is_correct():
    idx = torch.zeros(1, dtype=torch.int32, device="xpu")
    out = torch.empty(N, dtype=torch.int32, device="xpu")
    div_and_rem_kernel[(1, )](idx, out, N=N, D=D)
    offs = torch.arange(N, dtype=torch.int32, device="xpu")
    torch.testing.assert_close(out, offs // D + offs % D)


def test_loop_check_is_carried_and_asserted_once():
    ttir = compile_ttir(loop_div_and_rem_kernel, STEPS=STEPS)
    assert "arith.divsi" not in ttir and "arith.remsi" not in ttir, ttir
    # An assertion inside the loop body would cost several times the kernel
    # runtime, so the check is accumulated into a loop-carried flag and asserted
    # once outside.
    assert "arith.andi" in ttir, ttir
    assert ttir.count("tt.assert") == 1, ttir
    body = ttir.split("scf.for")[1].split("scf.yield")[0]
    assert "tt.assert" not in body, ttir


def test_loop_speculated_result_is_correct():
    idx = torch.zeros(STEPS, dtype=torch.int32, device="xpu")
    out = torch.empty(N, dtype=torch.int32, device="xpu")
    loop_div_and_rem_kernel[(1, )](idx, out, N=N, D=D, STEPS=STEPS)
    offs = torch.arange(N, dtype=torch.int32, device="xpu")
    torch.testing.assert_close(out, STEPS * (offs // D + offs % D))


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

# A dividend that only turns negative partway through the loop: the conjunction
# has to survive the iterations that pass and still flag the one that does not.
_NEGATIVE_IN_LATER_ITERATION_PROGRAM = """
import torch
import triton
import triton.language as tl


@triton.jit
def kernel(in_ptr, out_ptr, N: tl.constexpr, D: tl.constexpr, STEPS: tl.constexpr):
    offs = tl.arange(0, N)
    acc = tl.zeros((N, ), dtype=tl.int32)
    for i in range(STEPS):
        idx = offs + tl.load(in_ptr + i) * (N * 2)
        acc += idx // D + idx % D
    tl.store(out_ptr + offs, acc)


idx = torch.zeros(4, dtype=torch.int32, device="xpu")
idx[2] = -1
out = torch.empty(128, dtype=torch.int32, device="xpu")
kernel[(1, )](idx, out, N=128, D=16, STEPS=4)
torch.xpu.synchronize()
"""


def run_out_of_band(program):
    with tempfile.TemporaryDirectory() as tmpdir:
        script = os.path.join(tmpdir, "program.py")
        with open(script, "w") as f:
            f.write(program)
        # Force the knob on: an exported `=0` would otherwise turn the abort these
        # tests are looking for into a correctly computed result.
        env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(tmpdir, "cache"),
                   TRITON_INTEL_SPECULATE_SIGNED_DIV_REM="1")
        # A device-side hang after the assertion must fail the test rather than
        # wedge the job, so TimeoutExpired is deliberately left to propagate.
        result = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    # The message must name the escape hatch that is actually correct.
    assert "TRITON_INTEL_SPECULATE_SIGNED_DIV_REM=0" in output, output


def test_negative_dividend_aborts():
    run_out_of_band(_NEGATIVE_DIVIDEND_PROGRAM)


def test_negative_dividend_in_later_iteration_aborts():
    run_out_of_band(_NEGATIVE_IN_LATER_ITERATION_PROGRAM)
