"""BDPAS simulation throughput reproducer.

Measures TFLOPS for MXFP4×FP8 scaled matmul on tile shapes known to cause
register spill on BMG (Xe2). Run with pytest -s to see the printed numbers.

Baseline spill sizes on main (before Opportunity 1 / exponent-add):
    BDPAS_SIM_OPTIMIZATION_NOTES.md §6, §9 — all measured on BMG Arc B570:

    256×128×128:  0 bytes  (after Fix 1 / broadcastScale sync — ALREADY MERGED)
    128×256×128:  6720 bytes
    128×128×256:  20480 bytes
    128×256×256:  unknown (in CI parametrization, expected larger)

With register spill the kernel runs 10–50× slower due to scratch-memory spill
traffic serializing execution. The throughput difference is directly observable.
"""
import time
import pytest
import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)

_WARMUP = 3
_RUNS = 10

# ---------------------------------------------------------------------------
# Kernel — copied from python/test/unit/language/test_matmul.py
# (mxfp8_mxfp4_matmul, lines 1184-1251)
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp8_mxfp4_matmul(a_ptr, b_ptr, output_ptr, a_scale, b_scale, M, N, K, stride_scale, stride_am, stride_ak,
                        stride_bk, stride_bn, stride_cm, stride_cn, tensor_scale: tl.constexpr, DTYPE_A: tl.constexpr,
                        DTYPE_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        NUM_STAGES: tl.constexpr, PACK_B_ALONG_K: tl.constexpr = True):
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    DIV_FACTOR_B_K: tl.constexpr = DIV_FACTOR_B if PACK_B_ALONG_K else 1
    DIV_FACTOR_B_N: tl.constexpr = 1 if PACK_B_ALONG_K else DIV_FACTOR_B
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N // DIV_FACTOR_B_N + tl.arange(0, BLOCK_N // DIV_FACTOR_B_N)
    offs_bn_scale = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_ak = tl.arange(0, BLOCK_K // DIV_FACTOR_A)
    offs_bk = tl.arange(0, BLOCK_K // DIV_FACTOR_B_K)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)

    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn_scale[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            if tensor_scale:
                scale_a = tl.load(a_scale_ptr)
            else:
                scale_a = tl.full(a_scale_ptr.shape, a_scale.to(tl.int8), dtype=tl.int8)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator, rhs_k_pack=PACK_B_ALONG_K)
        a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR_B_K) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // 32
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // 32

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Shapes from BDPAS_SIM_OPTIMIZATION_NOTES.md §6 and §9.
# (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, description, spill_bytes_on_main)
# spill_bytes_on_main: measured on main before Opportunity 1, BMG Arc B570.
# ---------------------------------------------------------------------------

_SHAPES = [
    (1024, 1024, 1024, 256, 128, 128, "BLK=256x128x128 | spill=0 bytes after Fix1", 0),
    (1024, 1024, 1024, 128, 256, 128, "BLK=128x256x128 | spill=6720 bytes on main", 6720),
    (1024, 1024, 1024, 128, 128, 256, "BLK=128x128x256 | spill=20480 bytes on main", 20480),
    (1024, 1024, 1024, 128, 256, 256, "BLK=128x256x256 | spill=unknown (CI parametrization)", -1),
]

_IDS = [s[6].split("|")[0].strip() for s in _SHAPES]

# Performance gate: after Opportunity 1 (exponent-add with saturation) all
# spilling shapes should come within this factor of the no-spill baseline.
# Currently the spilling shapes fall well below this — confirming the problem.
_MIN_TFLOPS_AFTER_FIX = 7.0  # adjust once measured post-fix on the same hardware


def _build_inputs(M, N, K, device):
    """Build FP8e5m2 A and FP4e2m1 B (K-packed) with E8M0 scales."""
    torch.manual_seed(42)
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8, device=device).view(torch.float8_e5m2)
    b_fp4 = MXFP4Tensor(size=(K, N), device=device).random()
    b = b_fp4.to_packed_tensor(dim=0)  # K-pack → shape [K//2, N], strides [N, 1]
    a_scale = MXScaleTensor(size=(M, (K + 31) // 32), device=device).random(high=32.0).data
    b_scale = MXScaleTensor(size=(N, (K + 31) // 32), device=device).random(high=32.0).data
    return a, b, a_scale, b_scale


def _check_smem(bm, bn, bk):
    """Skip if this tile needs more shared memory than the device provides."""
    driver = triton.runtime.driver.active
    props = driver.utils.get_device_properties(driver.get_current_device())
    smem = props.get("max_shared_mem", props.get("max_shared_memory", 0))
    # 128×256×256 requires ~196 KB; most B-series cards cap at 128 KB.
    needed = {(128, 256, 256): 196608}.get((bm, bn, bk), 0)
    if needed and smem < needed:
        pytest.skip(f"Not enough shared memory ({smem} < {needed}) for BLK={bm}×{bn}×{bk}")


@pytest.mark.parametrize("M,N,K,BM,BN,BK,desc,spill", _SHAPES, ids=_IDS)
def test_mxfp4_fp8_throughput(M, N, K, BM, BN, BK, desc, spill, device):
    """Throughput reproducer for BDPAS simulation register spill.

    PURPOSE
    -------
    This test *reproduces the performance problem* described in
    BDPAS_SIM_OPTIMIZATION_NOTES.md §6/§9: register spill on large tiles
    causes 10–50× throughput degradation on BMG (Xe2).

    BEFORE Opportunity 1 (exponent-add): spilling shapes score well below
    _MIN_TFLOPS_AFTER_FIX → the test FAILS with a clear message.

    AFTER Opportunity 1: f32 intermediates eliminated, spill drops to ~0,
    all shapes should reach _MIN_TFLOPS_AFTER_FIX → test PASSES.

    Run with pytest -s to see the printed TFLOPS numbers.
    """
    _check_smem(BM, BN, BK)
    a, b, a_scale, b_scale = _build_inputs(M, N, K, device)
    output = torch.empty((M, N), dtype=torch.float32, device=device)
    stride_scale = a_scale.stride(0)  # = K//32 (same for a_scale and b_scale)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), 1)

    def _launch():
        _mxfp8_mxfp4_matmul[grid](
            a,
            b,
            output,
            a_scale,
            b_scale,
            M,
            N,
            K,
            stride_scale,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
            True,
            "e5m2",
            "e2m1",
            BM,
            BN,
            BK,
            NUM_STAGES=3,
            PACK_B_ALONG_K=True,
            num_warps=32,
            grf_mode="256",
        )

    for _ in range(_WARMUP):
        _launch()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(_RUNS):
        _launch()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = 2.0 * M * N * K * _RUNS / elapsed / 1e12
    print(f"\n[bdpas_sim_throughput] {desc} | {tflops:.2f} TFLOPS")

    # Shapes with known spill (spill > 0) must reach the post-fix performance
    # floor. They currently DON'T — that's the problem this test reproduces.
    if spill > 0:
        assert tflops >= _MIN_TFLOPS_AFTER_FIX, (
            f"PERFORMANCE REGRESSION / SPILL NOT FIXED: {BM}x{BN}x{BK} achieved "
            f"{tflops:.2f} TFLOPS but needs >= {_MIN_TFLOPS_AFTER_FIX:.1f} TFLOPS.\n"
            f"Known spill on main: {spill} bytes. "
            f"Root cause: bf16 mulf widened to f32 by arith_emulate_unsupported_floats. "
            f"Fix: Opportunity 1 (E8M0 exponent-add with saturation) in "
            f"DecomposeScaledBlocked.cpp.")
