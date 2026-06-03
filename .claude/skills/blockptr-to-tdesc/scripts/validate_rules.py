#!/usr/bin/env python
"""Phase-0 validation harness for the blockptr-to-tdesc skill.

For each transformation rule, this defines a minimal kernel pair:
  - a BEFORE kernel using the deprecated block-pointer API
    (tl.make_block_ptr / tl.advance / tl.load(boundary_check=...))
  - an AFTER kernel using the device-side tensor-descriptor API
    (tl.make_tensor_descriptor / desc.load / desc.store)

It then checks, on real XPU hardware, that:
  1. CORRECTNESS: the BEFORE kernel matches a torch reference, AND the AFTER
     kernel produces a numerically equal result (torch.testing.assert_close).
     A rule is only "validated" when before == after == reference.
  2. EFFICIENCY (optional, --check-ir): the AFTER kernel's TTGIR is dumped via
     MLIR_ENABLE_DUMP and inspected for a 2D-block-load (the fast path) vs a
     scalarized pointer/gather. This proves the descriptor form the skill emits
     is the one the XPU backend lowers efficiently.

The validated pairs are the ground truth for references/examples.md — never put
a transformation in the skill that has not passed this harness.

ENVIRONMENT:
    Requires an Intel XPU host with torch (XPU build) + the Intel XPU Triton
    backend installed, and the oneAPI runtime initialized in the shell
    (typically: `source <oneapi-install>/setvars.sh`, default /opt/intel/oneapi).
    Then, from the repo root or the skill's scripts/ directory:
        python validate_rules.py
        python validate_rules.py --only gemm
        python validate_rules.py --check-ir

    If a worktree build has rebound the global Triton install (a known issue in
    the Intel XPU backend repo when compile-triton.sh runs in a worktree), pin
    to the main build explicitly:
        PYTHONPATH=<repo>/python python validate_rules.py --check-ir

Exit code 0 = all selected rules validated; non-zero = at least one failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Rule 1/2/3/4/6/7/10: simple 2D GEMM — the canonical before/after pair.
#   BEFORE: benchmarks/.../gemm_block_ptr_benchmark.py:47-66
#   AFTER : benchmarks/.../gemm_benchmark.py:82-99
# Exercises: make_block_ptr->make_tensor_descriptor (R1), advance->off_k (R2),
#            load boundary_check drop (R3), store boundary_check drop (R4),
#            static offsets passed directly (R7).
# --------------------------------------------------------------------------- #
@triton.jit
def gemm_before(a_ptr, b_ptr, c_ptr, M, N, K,  #
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c_bp = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                             offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_bp, acc, boundary_check=(0, 1))


@triton.jit
def gemm_after(a_ptr, b_ptr, c_ptr, M, N, K,  #
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_K, BLOCK_N))
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_M, BLOCK_N))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_N])
        acc += tl.dot(a, b)
        off_k += BLOCK_K
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


def run_gemm(kernel, dtype=torch.float16):
    torch.manual_seed(0)  # identical inputs for BEFORE and AFTER kernels
    M, N, K = 256, 256, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    a = torch.randn((M, K), device="xpu", dtype=dtype)
    b = torch.randn((K, N), device="xpu", dtype=dtype)
    c = torch.empty((M, N), device="xpu", dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M, BLOCK_N, BLOCK_K)
    return c, (a.float() @ b.float())


# --------------------------------------------------------------------------- #
# Rule 6: transposed B operand. BEFORE uses order=(0,1); AFTER swaps
# shape/strides to keep last stride == 1 and applies .T after load.
# B is stored transposed as (N, K) row-major; logically we want (K, N).
# --------------------------------------------------------------------------- #
@triton.jit
def gemm_transb_before(a_ptr, b_ptr, c_ptr, M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0),
                             block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    # B is (N, K) row-major. A K x N block read in (K, N) logical order is a
    # transposed read: order=(0, 1) means dim 0 (K) is the contiguous-in-block dim.
    b_bp = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N),
                             block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c_bp = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                             offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_bp, acc, boundary_check=(0, 1))


@triton.jit
def gemm_transb_after(a_ptr, b_ptr, c_ptr, M, N, K,  #
                      stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,  #
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_M, BLOCK_K))
    # Keep last stride == 1: describe B in its native (N, K) row-major layout,
    # read an (N, K) block, and transpose in registers with .T to get (K, N).
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(N, K), strides=(stride_bn, stride_bk),
                                       block_shape=(BLOCK_N, BLOCK_K))
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_M, BLOCK_N))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, off_k])
        b = b_desc.load([pid_n * BLOCK_N, off_k]).T
        acc += tl.dot(a, b)
        off_k += BLOCK_K
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


def run_gemm_transb(kernel, dtype=torch.float16):
    torch.manual_seed(0)  # identical inputs for BEFORE and AFTER kernels
    M, N, K = 256, 256, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    a = torch.randn((M, K), device="xpu", dtype=dtype)
    b_t = torch.randn((N, K), device="xpu", dtype=dtype)  # B stored transposed
    c = torch.empty((M, N), device="xpu", dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b_t, c, M, N, K, a.stride(0), a.stride(1), b_t.stride(0), b_t.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M, BLOCK_N, BLOCK_K)
    return c, (a.float() @ b_t.float().t())


# --------------------------------------------------------------------------- #
# Rule 8 / Rule 11 (rank-3 fold-to-2D): batched GEMM. The batch offset is
# pre-added to base; the descriptor stays 2D over the M x K / K x N slice.
# --------------------------------------------------------------------------- #
@triton.jit
def bgemm_before(a_ptr, b_ptr, c_ptr, M, N, K,  #
                 stride_az, stride_am, stride_ak, stride_bz, stride_bk, stride_bn,  #
                 stride_cz, stride_cm, stride_cn,  #
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bid = tl.program_id(2)
    off_a = bid.to(tl.int64) * stride_az
    off_b = bid.to(tl.int64) * stride_bz
    a_bp = tl.make_block_ptr(base=a_ptr + off_a, shape=(M, K), strides=(stride_am, stride_ak),
                             offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(base=b_ptr + off_b, shape=(K, N), strides=(stride_bk, stride_bn),
                             offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    off_c = bid.to(tl.int64) * stride_cz
    c_bp = tl.make_block_ptr(base=c_ptr + off_c, shape=(M, N), strides=(stride_cm, stride_cn),
                             offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_bp, acc, boundary_check=(0, 1))


@triton.jit
def bgemm_after(a_ptr, b_ptr, c_ptr, M, N, K,  #
                stride_az, stride_am, stride_ak, stride_bz, stride_bk, stride_bn,  #
                stride_cz, stride_cm, stride_cn,  #
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bid = tl.program_id(2)
    off_a = bid.to(tl.int64) * stride_az
    off_b = bid.to(tl.int64) * stride_bz
    a_desc = tl.make_tensor_descriptor(base=a_ptr + off_a, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr + off_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_K, BLOCK_N))
    off_c = bid.to(tl.int64) * stride_cz
    c_desc = tl.make_tensor_descriptor(base=c_ptr + off_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_M, BLOCK_N))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_N])
        acc += tl.dot(a, b)
        off_k += BLOCK_K
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


def run_bgemm(kernel, dtype=torch.float16):
    torch.manual_seed(0)  # identical inputs for BEFORE and AFTER kernels
    B, M, N, K = 4, 128, 128, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    a = torch.randn((B, M, K), device="xpu", dtype=dtype)
    b = torch.randn((B, K, N), device="xpu", dtype=dtype)
    c = torch.empty((B, M, N), device="xpu", dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B)
    kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2),
                 c.stride(0), c.stride(1), c.stride(2), BLOCK_M, BLOCK_N, BLOCK_K)
    return c, torch.bmm(a.float(), b.float())


# --------------------------------------------------------------------------- #
# Rule 9: masked tensor-of-pointer load with a BOUNDARY mask. The mask is a
# range check (offs < limit) that folds into the descriptor `shape`. Uses a
# GEMM with K not a multiple of BLOCK_K so the K-edge tile is masked.
# --------------------------------------------------------------------------- #
@triton.jit
def gemm_kmask_before(a_ptr, b_ptr, c_ptr, M, N, K,  #
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@triton.jit
def gemm_kmask_after(a_ptr, b_ptr, c_ptr, M, N, K,  #
                     stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  #
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # The boundary mask `offs_k < K - k*BLOCK_K` folds into shape=(M, K) / (K, N):
    # the descriptor zero-pads the K-edge tile automatically.
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_K, BLOCK_N))
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_M, BLOCK_N))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([pid_m * BLOCK_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_N])
        acc += tl.dot(a, b)
        off_k += BLOCK_K
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


def run_gemm_kmask(kernel, dtype=torch.float16):
    torch.manual_seed(0)  # identical inputs for BEFORE and AFTER kernels
    M, N, K = 128, 128, 96  # K = 96 not a multiple of BLOCK_K=64 -> edge tile masked
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    a = torch.randn((M, K), device="xpu", dtype=dtype)
    b = torch.randn((K, N), device="xpu", dtype=dtype)
    c = torch.empty((M, N), device="xpu", dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M, BLOCK_N, BLOCK_K)
    return c, (a.float() @ b.float())


# --------------------------------------------------------------------------- #
# Rule 11 (rank-1): 1D copy with mask. BEFORE uses a rank-1 block pointer;
# AFTER uses a 1D tensor descriptor. (Will NOT get 2D block I/O today — this
# pair validates correctness + the "lowers via pointer path" expectation.)
# --------------------------------------------------------------------------- #
@triton.jit
def copy1d_before(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_bp = tl.make_block_ptr(base=x_ptr, shape=(N, ), strides=(1, ), offsets=(pid * BLOCK, ), block_shape=(BLOCK, ),
                             order=(0, ))
    y_bp = tl.make_block_ptr(base=y_ptr, shape=(N, ), strides=(1, ), offsets=(pid * BLOCK, ), block_shape=(BLOCK, ),
                             order=(0, ))
    v = tl.load(x_bp, boundary_check=(0, ))
    tl.store(y_bp, v + 1.0, boundary_check=(0, ))


@triton.jit
def copy1d_after(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(base=x_ptr, shape=(N, ), strides=(1, ), block_shape=(BLOCK, ))
    y_desc = tl.make_tensor_descriptor(base=y_ptr, shape=(N, ), strides=(1, ), block_shape=(BLOCK, ))
    v = x_desc.load([pid * BLOCK])
    y_desc.store([pid * BLOCK], v + 1.0)


def run_copy1d(kernel, dtype=torch.float32):
    torch.manual_seed(0)  # identical inputs for BEFORE and AFTER kernels
    N = 8192 + 17  # non-multiple of BLOCK so the edge is masked
    BLOCK = 1024
    x = torch.randn((N, ), device="xpu", dtype=dtype)
    y = torch.empty((N, ), device="xpu", dtype=dtype)
    grid = (triton.cdiv(N, BLOCK), )
    kernel[grid](x, y, N, BLOCK)
    return y, x + 1.0


# --------------------------------------------------------------------------- #
# Rule 11 (rank-1, NON-unit stride): a strided 1D view, e.g. column h of an
# (N, H) tensor read with stride H — exactly SGLang FLA's p_beta/p_g pattern
# tl.make_block_ptr(beta + ... + i_h, (T,), (H,), (i_t*BT,), (BT,), (0,)).
# A descriptor's last (only) stride must be 1, so strides=(H,) is ILLEGAL
# ("Tensor descriptor last dim must be 1 but got H"). The AFTER kernel therefore
# does NOT build a descriptor: it reproduces the boundary_check as a masked
# tensor-of-pointer load (no 2D block I/O — pointer path).
# --------------------------------------------------------------------------- #
@triton.jit
def copy1d_strided_before(x_ptr, y_ptr, col, N, H, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_bp = tl.make_block_ptr(base=x_ptr + col, shape=(N, ), strides=(H, ), offsets=(pid * BLOCK, ),
                             block_shape=(BLOCK, ), order=(0, ))
    y_bp = tl.make_block_ptr(base=y_ptr + col, shape=(N, ), strides=(H, ), offsets=(pid * BLOCK, ),
                             block_shape=(BLOCK, ), order=(0, ))
    v = tl.load(x_bp, boundary_check=(0, ))
    tl.store(y_bp, v + 1.0, boundary_check=(0, ))


@triton.jit
def copy1d_strided_after(x_ptr, y_ptr, col, N, H, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Non-unit stride (H) => no legal 1D descriptor. Reproduce boundary_check as a
    # mask on an explicit tensor-of-pointer load/store (Rule 11, strided rank-1).
    o = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + col + o * H, mask=o < N, other=0.0)
    tl.store(y_ptr + col + o * H, v + 1.0, mask=o < N)


def run_copy1d_strided(kernel, dtype=torch.float32):
    torch.manual_seed(0)  # identical inputs for BEFORE and AFTER kernels
    N = 8192 + 17  # non-multiple of BLOCK so the edge is masked
    H = 8  # inner stride: x is logically (N, H), we touch column `col`
    col = 3
    BLOCK = 1024
    x = torch.randn((N, H), device="xpu", dtype=dtype)
    y = torch.zeros((N, H), device="xpu", dtype=dtype)
    grid = (triton.cdiv(N, BLOCK), )
    kernel[grid](x, y, col, N, H, BLOCK)
    ref = y.clone()
    ref[:, col] = x[:, col] + 1.0
    return y, ref


# --------------------------------------------------------------------------- #
# Registry of validated pairs.
# --------------------------------------------------------------------------- #
@dataclass
class RulePair:
    name: str
    rules: str  # which transformation rules this exercises
    before: Callable
    after: Callable
    runner: Callable  # run(kernel) -> (triton_out, reference_out)
    rtol: float = 1e-2
    atol: float = 1e-2
    expect_block_io: bool = True  # whether AFTER should reach 2D block I/O
    notes: str = ""


PAIRS: list[RulePair] = [
    RulePair("gemm", "R1,R2,R3,R4,R7", gemm_before, gemm_after, run_gemm, notes="canonical simple 2D GEMM"),
    RulePair("gemm_transb", "R6", gemm_transb_before, gemm_transb_after, run_gemm_transb,
             notes="transposed B: keep last stride==1, .T after load"),
    RulePair("bgemm", "R8,R11(rank3)", bgemm_before, bgemm_after, run_bgemm,
             notes="batched: fold batch into base, 2D descriptor"),
    RulePair("gemm_kmask", "R9(boundary)", gemm_kmask_before, gemm_kmask_after, run_gemm_kmask,
             notes="boundary mask folds into shape"),
    RulePair("copy1d", "R11(rank1)", copy1d_before, copy1d_after, run_copy1d, expect_block_io=False,
             notes="rank-1 unit-stride: 1D descriptor, lowers via pointer path (no block I/O today)"),
    RulePair("copy1d_strided", "R11(rank1-strided)", copy1d_strided_before, copy1d_strided_after, run_copy1d_strided,
             expect_block_io=False,
             notes="rank-1 NON-unit stride: no legal descriptor -> masked tensor-of-pointer load (pointer path)"),
]
# NOTE: R5 (no-op), R10 (annotation hygiene, source-level only), R12
# (block-ptr-as-helper-argument, an interface change) are validated separately
# in the eval phase, not via numeric equality here.


def check_correctness(pair: RulePair) -> tuple[bool, str]:
    torch.manual_seed(0)
    try:
        before_out, ref = pair.runner(pair.before)
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001
        return False, f"BEFORE kernel raised: {e}"
    try:
        after_out, ref2 = pair.runner(pair.after)
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001
        return False, f"AFTER kernel raised: {e}"
    try:
        torch.testing.assert_close(before_out, ref.to(before_out.dtype), rtol=pair.rtol, atol=pair.atol)
    except AssertionError as e:
        return False, f"BEFORE != reference: {e}"
    try:
        torch.testing.assert_close(after_out, before_out, rtol=pair.rtol, atol=pair.atol)
    except AssertionError as e:
        return False, f"AFTER != BEFORE: {e}"
    return True, "before == after == reference"


def check_ir(pair: RulePair) -> tuple[bool, str]:
    """Re-run the AFTER kernel in a subprocess with MLIR_ENABLE_DUMP and grep the
    TTGIR for a 2D block load. Returns (matches_expectation, detail)."""
    env = dict(os.environ)
    env["MLIR_ENABLE_DUMP"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    # Register the dynamically-imported module in sys.modules BEFORE exec_module,
    # otherwise the @dataclass decorator on RulePair fails (dataclasses looks up
    # cls.__module__ in sys.modules, which would be None for an unregistered
    # module). This is why an earlier version captured an empty dump.
    snippet = ("import torch, triton, triton.language as tl, importlib.util, sys\n"
               f"spec = importlib.util.spec_from_file_location('vr', {__file__!r})\n"
               "m = importlib.util.module_from_spec(spec); sys.modules['vr'] = m\n"
               "spec.loader.exec_module(m)\n"
               f"pair = [p for p in m.PAIRS if p.name == {pair.name!r}][0]\n"
               "pair.runner(pair.after); torch.xpu.synchronize()\n")
    try:
        proc = subprocess.run([sys.executable, "-c", snippet], env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "IR dump timed out"
    dump = proc.stderr + proc.stdout
    # A surviving descriptor lowered to 2D block I/O shows up as a 2D block load
    # in TTGIR/LLIR. The exact op spelling can drift; match the common forms.
    block_io_markers = ["2Dblockload", "2d_block", "Subgroup2DBlock", "block_io", "make_tensor_descriptor"]
    found = [m for m in block_io_markers if m in dump]
    reached = any(m in dump for m in ["2Dblockload", "2d_block", "Subgroup2DBlock", "block_io"])
    if pair.expect_block_io:
        ok = reached
        detail = f"block-IO markers found: {found}" if ok else f"NO block-IO markers (found only: {found})"
    else:
        ok = True  # we don't require block I/O; just report
        detail = f"(block I/O not expected) markers seen: {found}"
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="run only the pair with this name")
    ap.add_argument("--check-ir", action="store_true",
                    help="also dump TTGIR and check whether AFTER reaches 2D block I/O")
    ap.add_argument("--list", action="store_true", help="list pairs and exit")
    args = ap.parse_args()

    if args.list:
        for p in PAIRS:
            print(f"{p.name:14s} rules={p.rules:16s} block_io={p.expect_block_io}  {p.notes}")
        return 0

    if not torch.xpu.is_available():
        print("ERROR: XPU not available. Initialize the oneAPI runtime "
              "(source <oneapi-install>/setvars.sh) and run on an XPU host.")
        return 2

    pairs = [p for p in PAIRS if (args.only is None or p.name == args.only)]
    if not pairs:
        print(f"No pair named {args.only!r}. Use --list.")
        return 2

    all_ok = True
    for p in pairs:
        ok, detail = check_correctness(p)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {p.name:14s} ({p.rules}) — {detail}")
        all_ok &= ok
        if ok and args.check_ir:
            ir_ok, ir_detail = check_ir(p)
            ir_status = "IR-OK" if ir_ok else "IR-MISS"
            print(f"        [{ir_status}] {ir_detail}")
            if p.expect_block_io:
                all_ok &= ir_ok
    print()
    print("ALL VALIDATED" if all_ok else "SOME FAILED — do not encode failing rules in the skill")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
