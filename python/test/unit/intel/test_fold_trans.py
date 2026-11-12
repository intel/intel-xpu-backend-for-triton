"""Tests for the `fold_trans` XPUOptions field, which makes AccelerateMatmul fold a tt.trans
feeding a dot's A operand into the ops that produce it, so the transpose needs no SLM."""

import pytest
import torch

import triton
import triton.language as tl
from triton.backends.intel.compiler import XPUBackend
from triton.compiler.compiler import make_backend


@triton.jit
def _dkdv_kernel(Q, K, V, DO, M, D, DK, DV, stride_tok, N_CTX, num_steps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 HEAD_DIM: tl.constexpr, MASK: tl.constexpr):
    # The dK/dV loop of the attention backward pass with dot(trans(p), do) and dot(trans(ds), q).
    # The fold only reaches descriptor loads, so Q, K, V and DO are loaded through descriptors.
    # MASK applies a causal mask to p, which makes the mask a select condition leaf of the fold.
    start_n = tl.program_id(0) * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    q_desc = tl.make_tensor_descriptor(Q, shape=[N_CTX, HEAD_DIM], strides=[stride_tok, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    do_desc = tl.make_tensor_descriptor(DO, shape=[N_CTX, HEAD_DIM], strides=[stride_tok, 1],
                                        block_shape=[BLOCK_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(K, shape=[N_CTX, HEAD_DIM], strides=[stride_tok, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(V, shape=[N_CTX, HEAD_DIM], strides=[stride_tok, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    curr_m = 0
    for blk_idx in range(num_steps):
        q = q_desc.load([blk_idx * BLOCK_M, 0])
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        m = tl.load(M + offs_m)
        qk = tl.dot(q, tl.trans(k))
        p = tl.math.exp2(qk - m[:, None])
        if MASK:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        do = do_desc.load([blk_idx * BLOCK_M, 0])
        dv = tl.dot(tl.trans(p).to(tl.float16), do, dv)
        Di = tl.load(D + offs_m)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - Di[:, None])
        dk = tl.dot(tl.trans(ds).to(tl.float16), q, dk)
        curr_m += BLOCK_M
    offs_d = tl.arange(0, HEAD_DIM)
    kv_off = offs_n[:, None] * stride_tok + offs_d[None, :]
    tl.store(DV + kv_off, dv.to(tl.float16))
    tl.store(DK + kv_off, dk.to(tl.float16))


def _has_dpas_and_2d_block_io():
    arch = triton.runtime.driver.active.get_current_target().arch
    return arch.get('has_2d_block_io', False) and arch.get('has_subgroup_matrix_multiply_accumulate', False)


@pytest.mark.skipif(not _has_dpas_and_2d_block_io(), reason="the fold targets DPAS dots on 2D block loads")
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("fold_trans", [True, False])
def test_fold_trans_reaches_accelerate_matmul(fold_trans, mask, device, with_allocator):
    N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N = 128, 64, 64, 64
    torch.manual_seed(0)
    q, k, v, do = (torch.randn((N_CTX, HEAD_DIM), dtype=torch.float16, device=device) * 0.5 for _ in range(4))
    # M is the row max of q @ k^T, so exp2(qk - m) stays in (0, 1] as in attention.
    qk = q.float() @ k.float().T
    m = qk.max(dim=1).values
    d = torch.randn((N_CTX, ), dtype=torch.float32, device=device)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    kernel = _dkdv_kernel[(N_CTX // BLOCK_N, )](q, k, v, do, m, d, dk, dv, HEAD_DIM, N_CTX, N_CTX // BLOCK_M,
                                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, MASK=mask,
                                                num_warps=8, num_stages=2, fold_trans=fold_trans)

    if not mask:
        ttgir = kernel.asm["ttgir"]
        # Without the fold, each of the two transposes feeding a dot's A stays a tt.trans and is
        # staged through SLM by a ttg.local_alloc; with it, both are gone.
        assert ("tt.trans " in ttgir) != fold_trans
        assert ("ttg.local_alloc" in ttgir) != fold_trans

    p = torch.exp2(qk - m[:, None])
    if mask:
        offs = torch.arange(N_CTX, device=device)
        p = torch.where(offs[:, None] >= offs[None, :], p, 0.0)
    dv_ref = p.half().float().T @ do.float()
    ds = p * (do.float() @ v.float().T - d[:, None])
    dk_ref = ds.half().float().T @ q.float()
    torch.testing.assert_close(dv.float(), dv_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dk.float(), dk_ref, rtol=1e-2, atol=1e-2)


def test_fold_trans_default_on():
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert isinstance(backend, XPUBackend)
    assert backend.parse_options({}).fold_trans is True
    assert backend.parse_options({"fold_trans": False}).fold_trans is False
