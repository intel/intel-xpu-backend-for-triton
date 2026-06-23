"""End-to-end test for PTSS overflow handling on non-DPAS Intel GPUs.

Compiles a real Triton matmul kernel sized to overflow per-thread scratch
space (256 KB on ARL-S) and verifies that the failure surfaces as
`triton.runtime.errors.OutOfResources` rather than an opaque
`IntelGPUError` / `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`.

Both compilation paths are exercised:
  * `generate_native_code = True`  -> AOT ocloc compile (compiler.py path)
  * `generate_native_code = False` -> JIT online compile via Level Zero
                                      (driver.py load_binary path)

Skipped on hardware that supports DPAS (PVC/BMG/Xe2) since those paths
don't trigger the FMA-induced spill that motivates this fix.

Related: https://github.com/intel/intel-xpu-backend-for-triton/issues/7273
"""
import os
import shutil

import pytest
import torch
import triton
import triton.language as tl

from triton.runtime.errors import OutOfResources

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)


def _has_ocloc() -> bool:
    """The AOT path shells out to `ocloc compile`; skip [aot] without it."""
    return shutil.which("ocloc") is not None


def _has_dpas() -> bool:
    """Return True on hardware where FMA fallback isn't exercised."""
    try:
        target = triton.runtime.driver.active.get_current_target()
        return bool(target.arch.get("has_subgroup_matrix_multiply_accumulate", False))
    except Exception:
        return False


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# A tile config known to spill past the 256 KB ARL-S PTSS limit when
# `tt.dot` lowers via FMA (i.e. on hardware without DPAS).
_OVERFLOW_M, _OVERFLOW_N, _OVERFLOW_K = 64, 128, 128


def _launch_overflowing_kernel():
    a = torch.randn((_OVERFLOW_M, _OVERFLOW_K), device='xpu', dtype=torch.float16)
    b = torch.randn((_OVERFLOW_K, _OVERFLOW_N), device='xpu', dtype=torch.float16)
    c = torch.zeros((_OVERFLOW_M, _OVERFLOW_N), device='xpu', dtype=torch.float16)
    grid = (1, 1)
    _matmul_kernel[grid](
        a,
        b,
        c,
        _OVERFLOW_M,
        _OVERFLOW_N,
        _OVERFLOW_K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_OVERFLOW_M,
        BLOCK_N=_OVERFLOW_N,
        BLOCK_K=_OVERFLOW_K,
        num_warps=4,
    )
    torch.xpu.synchronize()


@pytest.mark.skipif(_has_dpas(), reason="DPAS hardware doesn't trip FMA PTSS overflow")
@pytest.mark.parametrize(
    "generate_native_code",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not _has_ocloc(),
                reason="`ocloc` not on PATH — AOT path can't be exercised",
            ),
        ),
    ],
    ids=["jit", "aot"],
)
def test_ptss_overflow_raises_out_of_resources(generate_native_code, monkeypatch):
    """Both AOT and JIT compilation paths should map PTSS overflow to OutOfResources.

    The autotuner only catches `OutOfResources`. If either compilation path
    surfaces a different exception, autotune sweeps that include this tile
    config will fail hard instead of skipping it.
    """
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "1" if generate_native_code else "0")
    # Force fresh compilation so the path under test actually runs.
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    with pytest.raises(OutOfResources) as excinfo:
        _launch_overflowing_kernel()

    msg = str(excinfo.value)
    assert "scratch" in msg.lower() or "ptss" in msg.lower(), (
        f"OutOfResources should mention scratch/PTSS; got: {msg!r}")
