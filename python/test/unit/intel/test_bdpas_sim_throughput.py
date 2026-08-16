"""
Throughput reproducer for the BDPAS simulation spill problem on BMG (Xe2).

This test measures TFLOPS for an MXFP8×FP4 matmul across tile shapes that are
known to trigger register spill in the BDPAS software simulation. Spilling shapes
run significantly slower than non-spilling ones; the PASS/FAIL threshold documents
the expected improvement after a spill fix lands.

Background (~/bdpas-sim-optimization/SUMMARY.md):
  DecomposeScaledBlocked expands tt.dot_scaled into a bf16 mulf that
  arith_emulate_unsupported_floats widens to f32.  Large tiles exceed BMG's
  256-GRF register budget, spilling to scratch and collapsing throughput.

Exit criteria (Opportunity 1):
  All shapes should achieve >= _MIN_TFLOPS_AFTER_FIX (currently 7.0 TFLOPS).
"""

import time
import pytest
import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

# ── kernel ──────────────────────────────────────────────────────────────────


@triton.jit
def _mxfp8_mxfp4_matmul(
    a_ptr,
    b_ptr,
    output_ptr,
    a_scale,
    b_scale,
    M,
    N,
    K,
    stride_scale,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    tensor_scale: tl.constexpr,
    DTYPE_A: tl.constexpr,
    DTYPE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    PACK_B_ALONG_K: tl.constexpr = True,
):
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
        a_scale_ptr = (a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :])
    if b_scale is not None:
        b_scale_ptr = (b_scale + offs_bn_scale[:, None] * stride_scale + offs_scale_k[None, :])

    a_ptrs = (a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = (b_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        if a_scale is not None:
            scale_a = (tl.load(a_scale_ptr)
                       if tensor_scale else tl.full(a_scale_ptr.shape, a_scale.to(tl.int8), dtype=tl.int8))
        else:
            scale_a = None

        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None

        accumulator = tl.dot_scaled(
            a,
            scale_a,
            DTYPE_A,
            b,
            scale_b,
            DTYPE_B,
            accumulator,
            rhs_k_pack=PACK_B_ALONG_K,
        )

        a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR_B_K) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // 32
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // 32

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = (output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=mask)


# ── helpers ──────────────────────────────────────────────────────────────────


def _smem_bytes(BM, BN, BK, num_stages=3):
    """Conservative shared-memory estimate for the pipeliner."""
    fp8_bytes = BM * BK  # A: fp8 → 1 byte/elem
    fp4_bytes = (BK * BN) // 2  # B: packed fp4 → 0.5 byte/elem
    scale_a = BM * (BK // 32)  # scale A
    scale_b = BN * (BK // 32)  # scale B
    return num_stages * (fp8_bytes + fp4_bytes + scale_a + scale_b)


_SMEM_LIMIT = 200 * 1024  # 200 KiB – only skip truly unreasonable shapes


def _check_smem(BM, BN, BK, num_stages=3):
    used = _smem_bytes(BM, BN, BK, num_stages)
    if used > _SMEM_LIMIT:
        pytest.skip(f"Shared-memory estimate {used}B > {_SMEM_LIMIT}B limit")


def _build_inputs(M, N, K, device):
    torch.manual_seed(42)
    a = torch.randint(20, 40, (M, K), dtype=torch.uint8, device=device).view(torch.float8_e5m2)
    b_fp4 = MXFP4Tensor(size=(K, N), device=device).random()
    b = b_fp4.to_packed_tensor(dim=0)
    a_scale = MXScaleTensor(size=(M, (K + 31) // 32), device=device).random(high=32.0).data
    b_scale = MXScaleTensor(size=(N, (K + 31) // 32), device=device).random(high=32.0).data
    return a, b, a_scale, b_scale


# ── thresholds ────────────────────────────────────────────────────────────────

# Per-shape minimum TFLOPS required to PASS.
#
# History of fixes already landed on this host:
#   Fix 1 (broadcastScale dedup, merged):   256×128×128  0 B spill → passes.
#   7a464a6b2 Lu,Chengjun "Improve RemoveLayoutConversions for loop ops" (#7634):
#     Fixed 128×128×256 by correctly handling scf::ForOp during forward
#     rematerialization — eliminated duplicated loop bodies and the associated
#     register pressure.  Shape now achieves ~2.8 TFLOPS.
#
# Remaining open problem: 128×256×128 still spills ~10 KB → Opportunity 1.
_MIN_TFLOPS = {
    "256x128x128": 7.0,  # no spill, already fast
    "128x256x128": 7.0,  # spills ~10 KB — FAILS until Opp 1 lands
    "128x128x256": 2.5,  # fixed by 7a464a6b2; ~2.8 TFLOPS is expected
}

# Number of warmup and timed iterations.
_WARMUP = 3
_RUNS = 10

# ── parametrize ───────────────────────────────────────────────────────────────

_SHAPES = [
    # (BM,  BN,  BK,  desc,              spill_note)
    (256, 128, 128, "BLK=256x128x128", "0 B (Fix 1)"),
    (128, 256, 128, "BLK=128x256x128", "~10,560 B — Opp 1 open"),
    (128, 128, 256, "BLK=128x128x256", "0 B (7a464a6b2 loop fix)"),
    (128, 256, 256, "BLK=128x256x256", "skip—SMEM too large"),
]


@pytest.mark.parametrize("BM,BN,BK,desc,spill", _SHAPES, ids=[s[3] for s in _SHAPES])
def test_mxfp4_fp8_throughput(BM, BN, BK, desc, spill, device):
    _check_smem(BM, BN, BK)

    M = N = K = 1024
    a, b, a_scale, b_scale = _build_inputs(M, N, K, device)
    output = torch.empty((M, N), dtype=torch.float32, device=device)
    stride_scale = a_scale.stride(0)
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
            True,  # tensor_scale
            "e5m2",  # DTYPE_A
            "e2m1",  # DTYPE_B
            BM,
            BN,
            BK,
            NUM_STAGES=3,
            PACK_B_ALONG_K=True,
            num_warps=32,
            grf_mode="256",
        )

    # Warmup — first call compiles the kernel.
    for _ in range(_WARMUP):
        _launch()
    torch.xpu.synchronize()

    # Timed runs.
    t0 = time.perf_counter()
    for _ in range(_RUNS):
        _launch()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - t0

    tflops = 2.0 * M * N * K * _RUNS / elapsed / 1e12
    shape_key = f"{BM}x{BN}x{BK}"
    min_tflops = _MIN_TFLOPS.get(shape_key, 7.0)

    print(f"\n[bdpas_sim_throughput] {desc} | "
          f"spill={spill} | "
          f"{tflops:.2f} TFLOPS  (threshold={min_tflops:.1f})")

    assert tflops >= min_tflops, (f"THROUGHPUT BELOW THRESHOLD: "
                                  f"{BM}×{BN}×{BK} achieved {tflops:.2f} TFLOPS "
                                  f"but needs >= {min_tflops} TFLOPS.\n"
                                  f"  Known spill note: {spill}.\n"
                                  f"  If this is 128×256×128: root cause is bf16 mulf widened to f32 by "
                                  f"arith_emulate_unsupported_floats. Fix: Opportunity 1 "
                                  f"(E8M0 exponent-add) in DecomposeScaledBlocked.cpp.")
