"""Runtime out-of-bounds test for symbolic mask removal.

With `TRITON_INTEL_SYMBOLIC_MASKS=1`, RemoveMasks may drop a mask it cannot
match structurally, under a runtime guard derived from the symbolic bounds
prover. If such a guard is too weak, the unmasked loop copy runs for a shape
where the mask was load-bearing and reads past the logical end of the tensor.
A plain numerical comparison can miss that, because the bytes past the end are
often another row's data, so every operand below is allocated with a NaN tail
*inside the same allocation*: an out-of-bounds read that reaches the result
surfaces as a NaN.

Both modes are run for every shape and both must match an eager reference
exactly, so a case fails whether a dropped mask reads out of bounds or the
guard changes the computed value. The shapes whose K is not a multiple of
BLOCK_K (770, 1000) fail the guard at runtime and must execute the masked copy;
they are what pins "the mask is kept where it is not redundant".

Running this from a git worktree: `python/conftest.py` drops every
`<tree>/python` entry from `sys.path` and re-appends it last, so an editable
install's `.pth` wins over `PYTHONPATH` and `import triton` resolves to the
*other* checkout -- whose libtriton.so has no symbolic-mask support, making the
`symbolic="1"` cases fail for a reason that has nothing to do with the code
under test. Import triton before pytest starts, which pins `sys.modules`:

    python3 -c "import sys, triton, pytest; sys.exit(pytest.main(sys.argv[1:]))" \\
      -q python/test/unit/intel/test_remove_masks_symbolic.py --device xpu
"""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def masked_k_sum_kernel(x_ptr, out_ptr, K, BLOCK_K: tl.constexpr):
    # An inductor reduction shape: for k in range(0, K, BLOCK_K),
    # load x[k + lane] under `lane + k < K`. No legacy validator matches this
    # shape, so only the symbolic path versions the loop.
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    lane = tl.arange(0, BLOCK_K)
    for k in range(0, K, BLOCK_K):
        idx = k + lane
        acc += tl.load(x_ptr + idx, mask=idx < K, other=0.0)
    tl.store(out_ptr + lane, acc)


@triton.jit
def masked_k_matmul_kernel(a_ptr, b_ptr, c_ptr, K, BLOCK_K: tl.constexpr):
    # The tutorial-03 shape: one 16x16 tile, K loop with a cdiv bound.
    # The legacy canonical validator recognizes this one, so both modes version.
    offs = tl.arange(0, 16)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros([16, 16], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # `k_off` and `rem` are written once for readability; CSE runs before
        # RemoveMasks (intel/backend/compiler.py:477) and unifies them anyway.
        k_off = k * BLOCK_K
        rem = K - k_off
        a = tl.load(a_ptr + offs[:, None] * K + offs_k[None, :] + k_off, mask=offs_k[None, :] < rem, other=0.0)
        b = tl.load(b_ptr + (offs_k[:, None] + k_off) * 16 + offs[None, :], mask=offs_k[:, None] < rem, other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * 16 + offs[None, :], acc)


def _fresh(kernel):
    """Return a fresh `JITFunction` wrapping `kernel`'s python function.

    `JITFunction` memoizes compiled kernels in-process (`device_caches`) keyed
    on specialization and options only, not on the environment
    (runtime/jit.py:757-758), so the second mode of a parametrized pair would
    otherwise be served the first mode's kernel. Same reason as the per-test
    kernel factory in test_auto_grf_num_warps_guard.py:81.
    """
    return triton.jit(kernel.fn)


def _versioned(ttir: str) -> bool:
    """True if RemoveMasks versioned a loop in `ttir`.

    Versioning clones the loop into the two arms of an `scf.if`, so a versioned
    loop is an `scf.if` plus a second `scf.for`. `asm["ttir"]` is the end of
    `make_ttir`, where RemoveMasks runs (intel/backend/compiler.py:480); nothing
    later in that pipeline clones a pointer-based loop, as stride and descriptor
    versioning only select `tt.descriptor_load`.
    """
    return "scf.if" in ttir and ttir.count("scf.for") >= 2


# The divisibility half of the guard, `K % BLOCK_K == 0`. `SimplifySignedArithmetic`
# runs after RemoveMasks and rewrites `remsi` to `remui` when both operands are
# provably non-negative (SimplifySignedArithmetic.cpp:76-81), so accept either.
_REM_GUARD = re.compile(r"arith\.rem[su]i")


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize("K", [0, 64, 768, 770, 1000])
@pytest.mark.parametrize("symbolic", ["0", "1"])
def test_masked_k_sum_no_oob(K, symbolic, monkeypatch, device):
    # The switch is cache-invalidating (Tools/Sys/GetEnv.h), so it is part of the
    # on-disk cache key too.
    monkeypatch.setenv("TRITON_INTEL_SYMBOLIC_MASKS", symbolic)
    BLOCK_K = 64

    # NaN sentinels: the logical tensor is followed by BLOCK_K NaNs inside the
    # same allocation, so a read past K under a dropped mask pulls a NaN into
    # `acc` instead of touching unmapped memory.
    buf = torch.full((K + BLOCK_K, ), float("nan"), device=device, dtype=torch.float32)
    x = torch.arange(1, K + 1, device=device, dtype=torch.float32)
    buf[:K] = x
    out = torch.empty((BLOCK_K, ), device=device, dtype=torch.float32)

    compiled = _fresh(masked_k_sum_kernel)[(1, )](buf, out, K, BLOCK_K=BLOCK_K)

    # The numerical checks below also pass on the legacy path, which leaves this
    # shape alone, so assert that the symbolic path really versioned the loop --
    # and that the legacy one still does not.
    ttir = compiled.asm["ttir"]
    # The triton path is in the message because the usual cause of this
    # assertion firing is importing a different checkout's build (see the
    # module docstring), not a prover regression.
    assert _versioned(ttir) == (symbolic == "1"), f"triton from {triton.__file__}\n{ttir}"
    if symbolic == "1":
        assert _REM_GUARD.search(ttir), ttir

    # Eager reference: the same iteration space, masked, with the gather clamped
    # so the reference itself never indexes past K.
    ref = torch.zeros((BLOCK_K, ), device=device, dtype=torch.float32)
    lane = torch.arange(0, BLOCK_K, device=device)
    for k in range(0, K, BLOCK_K):
        idx = k + lane
        ref += torch.where(idx < K, x[idx.clamp(max=max(K - 1, 0))], torch.zeros_like(ref))

    assert not torch.isnan(out).any(), f"K={K}: a read past the logical end reached the NaN sentinels"
    # Integer-valued fp32 lane sums below 2**24, so equality is exact.
    assert torch.equal(out, ref), f"K={K}: {out} != {ref}"


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize("K", [64, 768, 770])
@pytest.mark.parametrize("symbolic", ["0", "1"])
def test_masked_k_matmul_no_oob(K, symbolic, monkeypatch, device):
    monkeypatch.setenv("TRITON_INTEL_SYMBOLIC_MASKS", symbolic)
    BLOCK_K = 64
    torch.manual_seed(17)

    # NaN tails after each logical operand, as above.
    a_buf = torch.full((16 * K + BLOCK_K, ), float("nan"), device=device, dtype=torch.float16)
    b_buf = torch.full(((K + BLOCK_K) * 16, ), float("nan"), device=device, dtype=torch.float16)
    a = torch.randn((16, K), device=device, dtype=torch.float16)
    b = torch.randn((K, 16), device=device, dtype=torch.float16)
    a_buf[:16 * K] = a.flatten()
    b_buf[:K * 16] = b.flatten()
    c = torch.empty((16, 16), device=device, dtype=torch.float32)

    compiled = _fresh(masked_k_matmul_kernel)[(1, )](a_buf, b_buf, c, K, BLOCK_K=BLOCK_K)

    # The legacy canonical validator versions this shape too, so both modes must.
    ttir = compiled.asm["ttir"]
    assert _versioned(ttir), ttir
    assert _REM_GUARD.search(ttir), ttir

    assert not torch.isnan(c).any(), f"K={K}: a read past the logical end reached the NaN sentinels"
    torch.testing.assert_close(c, a.float() @ b.float(), atol=1e-2, rtol=1e-2)
