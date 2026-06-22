"""Regression tests for LSC d16 block prefetch crash on BMG/CRI (gen12.9).

The Triton compiler's prefetch emit path previously split d16 (16-bit) tile_width=32
into {tile_width=16, vBlocks=2}, producing block_width=32 bytes with array_length=2.
This violated the TBX emulator constraint (block_width × array_length == 128 when
array_length > 1), causing a CB_FATAL.

The fix removes the d16 split entirely, emitting single-block prefetches
(tile_width=32, vBlocks=1) which are always valid and more efficient.

Two layers of verification:
1. **IR-level (runs on any platform)**: assert the lowered IR never contains the
   buggy {tile_width=16, v_blocks=2} pattern for d16. This catches a regression
   regardless of whether the test machine is CRI/BMG/PVC.
2. **End-to-end (XPU only)**: run the matmul and check correctness against
   torch.matmul. On CRI this is what previously crashed.

References:
    PYTORCHDGQ-9207
    https://gfxspecs.intel.com/Predator/Home/Index/57329
"""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _matmul_kernel_tdesc(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_a0,
    stride_b0,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_a0, 1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_b0, 1],
                                       block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K, num_stages=NUM_STAGES):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, k])
        b = b_desc.load([k, pid_n * BLOCK_SIZE_N])
        accumulator += tl.dot(a, b)

    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[stride_cm, 1],
                                       block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator.to(c_ptr.dtype.element_ty))


@triton.jit
def _matmul_kernel_ptrs(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # No explicit tl.prefetch here: with num_stages > 1, the loop pipeliner
    # invokes rewriteRegularPointerPrefetch, which emits 2D block prefetches
    # for the loop-carried pointer loads — the path under test for this bug.
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty))


# The lowered LLVM IR contains the SPIRV prefetch call with positional integer args:
#   __spirv_Subgroup2DBlockPrefetchINTEL...(i32 elem_size_bytes, i32 tile_width,
#                                           i32 tile_height, i32 v_blocks, ...)
# The original bug emitted (i32 2, i32 16, i32 X, i32 2, ...) — d16 (2 bytes),
# tile_width=16, any tile_height, v_blocks=2 → block_width × array_length = 32 × 2
# = 64 ≠ 128 → CB_FATAL on CRI.
_LLIR_PREFETCH_CALL = re.compile(
    r"@_Z\d+__spirv_Subgroup2DBlockPrefetchINTEL\w*\("
    r"\s*i32\s+(\d+)\s*,"  # elem_size_bytes
    r"\s*i32\s+(\d+)\s*,"  # tile_width
    r"\s*i32\s+(\d+)\s*,"  # tile_height
    r"\s*i32\s+(\d+)\s*,",  # v_blocks
)


def _find_buggy_d16_prefetch_calls(llir):
    """Return a list of (tile_width, tile_height, v_blocks) for every d16 prefetch
    call in `llir` that matches the buggy crash shape (tile_width=16, v_blocks=2).

    On a fixed compiler this list will be empty.
    """
    buggy = []
    for m in _LLIR_PREFETCH_CALL.finditer(llir):
        elem_bytes, tile_w, tile_h, v_blocks = (int(g) for g in m.groups())
        if elem_bytes == 2 and tile_w == 16 and v_blocks == 2:
            buggy.append((tile_w, tile_h, v_blocks))
    return buggy


# ---------------------------------------------------------------------------
# IR-level regression check (runs on any platform — does not require CRI/BMG).
# Catches the regression at compile time by inspecting the lowered IR.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K", [[32, 32, 32], [64, 64, 32], [32, 32, 64]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.skipif(not is_xpu(), reason="XPU compiler target required (compile-only, no kernel launch)")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io'],
                   reason="d16 prefetch path is not exercised on targets without 2D block I/O", run=False)
def test_d16_prefetch_ir_no_buggy_split_tdesc(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, num_stages, device):
    """Compile-only check: lowered LLIR must not contain a (2, 16, *, 2, ...) prefetch call.

    The bug emitted `__spirv_Subgroup2DBlockPrefetchINTEL(2, 16, X, 2, ...)` for
    d16 inputs — block_width(32 bytes) × array_length(2) = 64 ≠ 128, which crashed
    on CRI. The fix collapses the split, so the corresponding call should be
    `(2, 32, X, 1, ...)` (single-block).

    This compile-only check runs on any XPU build host with 2D block I/O support
    (no CRI/BMG required), so a regression of PYTORCHDGQ-9207 is caught in
    standard CI rather than only on the daily CRI job. xfail-with-run=False on
    targets that lack `has_subgroup_2d_block_io` (matches the project pattern in
    test_block_load.py:18). Asserts that at least one d16 prefetch was emitted —
    zero d16 prefetches on a capable target would mean the compiler silently
    stopped prefetching, which itself is a regression.
    """
    M, N, K = 128, 128, 128
    a = torch.empty((M, K), device=device, dtype=dtype)
    b = torch.empty((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=dtype)

    compiled = _matmul_kernel_tdesc.warmup(
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_STAGES=num_stages,
        grid=(1, ),
    )

    llir = compiled.asm.get("llir", "")
    n_d16 = sum(1 for m in _LLIR_PREFETCH_CALL.finditer(llir) if int(m.group(1)) == 2)
    assert n_d16 > 0, ("Compiler emitted zero d16 2D block prefetch calls on a 2D-block-I/O-capable target — "
                       "this is itself a regression (the optimization silently stopped firing).")
    buggy = _find_buggy_d16_prefetch_calls(llir)
    assert not buggy, (f"Found {len(buggy)} buggy d16 prefetch call(s) (tile_width=16, v_blocks=2) in LLIR. "
                       f"This is the PYTORCHDGQ-9207 crash shape — block_width(32B) × array_length(2) = 64 != 128. "
                       f"Sample: {buggy[0]}")


# ---------------------------------------------------------------------------
# IR-level regression check for the tensor-of-pointers path. Mirrors the
# tensor-descriptor IR test above but exercises rewriteRegularPointerPrefetch.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K", [[32, 32, 32], [64, 64, 32], [32, 32, 64]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.skipif(not is_xpu(), reason="XPU compiler target required (compile-only, no kernel launch)")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io'],
                   reason="d16 prefetch path is not exercised on targets without 2D block I/O", run=False)
def test_d16_prefetch_ir_no_buggy_split_ptrs(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, num_stages, device):
    """Compile-only check for the regular-pointer prefetch path.

    Same crash shape as test_d16_prefetch_ir_no_buggy_split_tdesc, but produced
    by rewriteRegularPointerPrefetch on a tensor-of-pointers kernel rather than
    the tensor-descriptor cooperative path.
    """
    M, N, K = 128, 128, 128
    a = torch.empty((M, K), device=device, dtype=dtype)
    b = torch.empty((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=dtype)

    compiled = _matmul_kernel_ptrs.warmup(
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_STAGES=num_stages,
        grid=(1, ),
    )

    llir = compiled.asm.get("llir", "")
    n_d16 = sum(1 for m in _LLIR_PREFETCH_CALL.finditer(llir) if int(m.group(1)) == 2)
    assert n_d16 > 0, ("Compiler emitted zero d16 2D block prefetch calls on a 2D-block-I/O-capable target — "
                       "this is itself a regression (the optimization silently stopped firing).")
    buggy = _find_buggy_d16_prefetch_calls(llir)
    assert not buggy, (f"Found {len(buggy)} buggy d16 prefetch call(s) (tile_width=16, v_blocks=2) in LLIR. "
                       f"This is the PYTORCHDGQ-9207 crash shape — block_width(32B) × array_length(2) = 64 != 128. "
                       f"Sample: {buggy[0]}")


# ---------------------------------------------------------------------------
# End-to-end matmul via tensor descriptor (cooperative prefetch path).
# Runs on actual XPU hardware; previously crashed on CRI.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K", [[32, 32, 32], [64, 64, 32], [32, 32, 64]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific 2D block prefetch test")
@pytest.mark.xfail(
    not (triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io']
         and triton.runtime.driver.active.get_current_target().arch['has_subgroup_matrix_multiply_accumulate']),
    reason="Block loads and/or DPAS not supported on this architecture", run=False)
def test_d16_prefetch_tensor_desc(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, num_stages, device):
    """End-to-end correctness check via tensor descriptor path.

    Before the fix, this kernel would crash on CRI with CB_FATAL because the
    compiler emitted an LSC 2D block prefetch with `block_width=32 bytes,
    array_length=2`, violating the gen12.9 hardware/emulator constraint.
    """
    M, N, K = 128, 128, 128
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=dtype, requires_grad=False)
    b = torch.randn((K, N), device=device, dtype=dtype, requires_grad=False)
    c = torch.empty((M, N), device=device, dtype=dtype)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    compiled = _matmul_kernel_tdesc[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_STAGES=num_stages,
    )

    # Verify a d16 prefetch was actually emitted — guards against the compiler
    # silently dropping the optimization (which would let correctness checks
    # pass while losing the perf path under test).
    llir = compiled.asm.get("llir", "")
    n_d16 = sum(1 for m in _LLIR_PREFETCH_CALL.finditer(llir) if int(m.group(1)) == 2)
    assert n_d16 > 0, "Expected at least one d16 2D block prefetch in LLIR but found none."

    ref = torch.matmul(a.float(), b.float()).to(dtype)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# End-to-end matmul via tensor-of-pointers (regular pointer prefetch path).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K", [[32, 32, 32], [64, 64, 32], [32, 32, 64]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific 2D block prefetch test")
@pytest.mark.xfail(
    not (triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io']
         and triton.runtime.driver.active.get_current_target().arch['has_subgroup_matrix_multiply_accumulate']),
    reason="Block loads and/or DPAS not supported on this architecture", run=False)
def test_d16_prefetch_tensor_of_ptr(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, num_stages, device):
    """End-to-end correctness check via tensor-of-pointers path.

    Exercises rewriteRegularPointerPrefetch via a matmul with explicit
    pointer arithmetic and num_stages > 1.
    """
    M, N, K = 128, 128, 128
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=dtype, requires_grad=False)
    b = torch.randn((K, N), device=device, dtype=dtype, requires_grad=False)
    c = torch.empty((M, N), device=device, dtype=dtype)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    compiled = _matmul_kernel_ptrs[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_STAGES=num_stages,
    )

    # Verify a d16 prefetch was actually emitted — guards against the compiler
    # silently dropping the optimization (which would let correctness checks
    # pass while losing the perf path under test).
    llir = compiled.asm.get("llir", "")
    n_d16 = sum(1 for m in _LLIR_PREFETCH_CALL.finditer(llir) if int(m.group(1)) == 2)
    assert n_d16 > 0, "Expected at least one d16 2D block prefetch in LLIR but found none."

    ref = torch.matmul(a.float(), b.float()).to(dtype)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
