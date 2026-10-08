"""Runtime coverage for the Intel sub-group scan lowering.

`TargetInfo::warpScan` replaces upstream's log2(scanDim)-deep `shuffle_up` +
select chain with a single `spirv.GroupNonUniform*` `InclusiveScan` op when the
scan axis fills the sub-group. `test/TritonIntelGPU/tritongpu_scan_op_lowering.mlir`
pins the emitted IR; these tests pin the semantics on hardware, for each seam the
builtin has to compose with.
"""
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)

# Element count and `num_warps` together fix the blocked layout: `sizePerThread`
# is `M / (num_warps * threads_per_warp)`. The names describe the default 32-lane
# sub-group.
SEAMS = {
    (32, 1): "one-element-per-lane",
    (128, 1): "multiple-elements-per-thread",  # the intra-thread carry
    (128, 4): "multiple-axis-warps",  # the cross-warp SLM + barrier combine
    (1024, 4): "both-seams",
}
SEAM_PARAMS = [pytest.param(m, w, id=name) for (m, w), name in SEAMS.items()]

_GROUP_OP = re.compile(r"__spirv_GroupNonUniform(\w+)")


def _assert_builtin(kernel, M, num_warps, family):
    """Fail loudly if a case stopped reaching the builtin instead of silently passing."""
    warp_size = kernel.metadata.threads_per_warp
    assert M >= num_warps * warp_size, (f"M={M} < {num_warps} warps x {warp_size} lanes: the layout "
                                        "broadcasts, so this case no longer covers the seam its id names")
    ops = sorted(set(_GROUP_OP.findall(kernel.asm["llir"])))
    assert any(op.startswith(family) for op in ops), f"expected a GroupNonUniform{family} scan, got {ops}"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("M, num_warps", SEAM_PARAMS)
def test_cumsum_i32(M, num_warps, reverse, device):
    """Integer add is associative, so the hardware scan must match exactly."""

    @triton.jit
    def kernel(out_ptr, in_ptr, M: tl.constexpr, REVERSE: tl.constexpr):
        offs = tl.arange(0, M)
        tl.store(out_ptr + offs, tl.cumsum(tl.load(in_ptr + offs), 0, reverse=REVERSE))

    torch.manual_seed(17)
    x = torch.randint(-100, 100, (M, ), dtype=torch.int32, device=device)
    out = torch.empty_like(x)
    _assert_builtin(kernel[(1, )](out, x, M, reverse, num_warps=num_warps), M, num_warps, "IAdd")

    ref = torch.cumsum(x.flip(0) if reverse else x, 0)
    torch.testing.assert_close((ref.flip(0) if reverse else ref).to(torch.int32), out, atol=0, rtol=0)


@triton.jit
def _and(a, b):
    return a & b


@triton.jit
def _or(a, b):
    return a | b


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _mul(a, b):
    return a * b


# The `i1` combines the scan gate allows, with the builtin each must lower to.
# `addi`/`maxsi`/`minsi` on `i1` are rejected, so they keep the shuffle chain and
# are covered by the lit negatives instead.
I1_OPS = [
    pytest.param(_and, "LogicalAnd", lambda a, b: a and b, id="andi"),
    pytest.param(_or, "LogicalOr", lambda a, b: a or b, id="ori"),
    pytest.param(_xor, "LogicalXor", lambda a, b: a != b, id="xori"),
    pytest.param(_mul, "LogicalAnd", lambda a, b: a and b, id="muli"),
]


@pytest.mark.parametrize("combine_fn, family, ref_fn", I1_OPS)
def test_i1_truth_table(combine_fn, family, ref_fn, device):
    """Needs at least two `true` inputs: a wrong logical mapping is invisible otherwise."""

    # `combine_fn` is captured from the enclosing scope, not passed as an argument:
    # a `JITFunction` argument arrives wrapped in `constexpr`, which
    # `associative_scan` cannot trace.
    @triton.jit
    def kernel(out_ptr, in_ptr, M: tl.constexpr):
        offs = tl.arange(0, M)
        x = tl.load(in_ptr + offs) != 0
        tl.store(out_ptr + offs, tl.associative_scan(x, 0, combine_fn).to(tl.int32))

    M = 32
    # Contains true-after-true and true-after-false transitions, so and/or/xor
    # each produce a distinct result vector.
    bits = [1, 1, 0, 1, 0, 0, 1, 0] * (M // 8)
    x = torch.tensor(bits, dtype=torch.int32, device=device)
    out = torch.empty_like(x)
    _assert_builtin(kernel[(1, )](out, x, M, num_warps=1), M, 1, family)

    acc, ref = None, []
    for bit in bits:
        acc = bool(bit) if acc is None else ref_fn(acc, bool(bit))
        ref.append(int(acc))
    assert out.tolist() == ref


_ARM = textwrap.dedent("""
    import json, re, sys
    import torch, triton, triton.language as tl

    @triton.jit
    def kernel(out_ptr, in_ptr, M: tl.constexpr, REVERSE: tl.constexpr):
        offs = tl.arange(0, M)
        tl.store(out_ptr + offs, tl.cumsum(tl.load(in_ptr + offs), 0, reverse=REVERSE))

    builtins, results = 0, {}
    for M, num_warps in json.loads(sys.argv[1]):
        for reverse in (False, True):
            torch.manual_seed(17)
            x = torch.randint(-100, 100, (M, ), dtype=torch.int32, device="xpu")
            out = torch.empty_like(x)
            kernel_handle = kernel[(1, )](out, x, M, reverse, num_warps=num_warps)
            builtins += len(re.findall("__spirv_GroupNonUniformIAdd", kernel_handle.asm["llir"]))
            results["%d-%d-%s" % (M, num_warps, reverse)] = out.tolist()
    print(json.dumps({"builtins": builtins, "results": results}))
    """)


def _run_arm(tmp_path, enabled):
    """Run every seam in a fresh process with the knob pinned.

    One process per arm because the knob is read at compile time and an in-process
    re-launch never recompiles: `JITFunction`'s kernel cache key is built from the
    specialization and options only, so the env var -- which participates in the
    *on-disk* cache key -- is never consulted on a hit.
    """
    script = tmp_path / f"arm{enabled:d}.py"
    script.write_text(_ARM)
    # The child does not inherit `sys.path`, so point it at the same `triton` this
    # process imported. Matters when running out of a worktree.
    pythonpath = [os.path.dirname(os.path.dirname(triton.__file__))]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    result = subprocess.run([sys.executable, str(script), json.dumps(list(SEAMS))], capture_output=True, text=True,
                            env={
                                **os.environ, "TRITON_INTEL_SUBGROUP_SCAN": "1" if enabled else "0", "PYTHONPATH":
                                os.pathsep.join(pythonpath)
                            })
    assert result.returncode == 0, (f"arm enabled={enabled} exited with {result.returncode}\n"
                                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return json.loads(result.stdout)


def test_knob_arms_agree(tmp_path):
    """`TRITON_INTEL_SUBGROUP_SCAN=0` must fall back to the shuffle chain, bit-identically."""
    on = _run_arm(tmp_path, True)
    off = _run_arm(tmp_path, False)
    assert on["builtins"] > 0, "knob on emitted no builtin, so the A/B compares nothing"
    assert off["builtins"] == 0, "knob off still emitted the builtin"
    assert on["results"] == off["results"]
