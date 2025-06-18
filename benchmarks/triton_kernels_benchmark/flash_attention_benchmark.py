import os
import contextlib
from typing import Callable, Optional

import torch
from torch.profiler import record_function
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark import xetla_kernel
from triton_kernels_benchmark import cutlass_kernel
import numpy as np


# pylint: disable=unused-argument
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_with_block_pointers(Q, K, V, sm_scale, M, Out,  #
                                  stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr,
                                  stride_qk: tl.constexpr,  #
                                  stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
                                  stride_kk: tl.constexpr,  #
                                  stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr,
                                  stride_vn: tl.constexpr,  #
                                  stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr,
                                  stride_on: tl.constexpr,  #
                                  Z: tl.constexpr, H: tl.constexpr,  #
                                  N_CTX: tl.constexpr,  #
                                  BLOCK_M: tl.constexpr,  #
                                  BLOCK_DMODEL: tl.constexpr,  #
                                  BLOCK_N: tl.constexpr,  #
                                  STAGE: tl.constexpr  #
                                  ):  # pylint: disable=unused-argument

    start_m = tl.program_id(2)
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    if N_CTX <= 512:
        start_m = tl.program_id(0)
        off_z = tl.program_id(2)
        qvk_offset = off_z.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 the kernel gets 1 as its STAGE
    # For causal = False, STAGE = 1, the kernel gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'grf_mode': 'large', 'one_matrix_per_load_for_bt': True}, num_stages=s, num_warps=w) \
    for BM in [128, 256] \
    for BN in [32, 64] \
    for s in [2, 3, 4] \
    for w in [8, 16, 32] \
    ]

tuner = triton.autotune(configs, key=['N_CTX', 'BLOCK_DMODEL', 'STAGE'])


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
# pylint: disable=unused-variable
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs).to(tl.float16)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
# pylint: disable=unused-variable
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do.to(tl.float16), vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):
    tune_attn_fwd: Callable = None
    attn_fwd: Callable = None

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dq, dk, dv, delta):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=torch.float32)
        BLOCK_M = 128
        BLOCK_N = 64
        num_stages = 3
        num_warps = 8 if Lq == 64 else 16
        stage = 3 if causal else 1
        grid = lambda args: (q.shape[0], q.shape[1], triton.cdiv(q.shape[2], args['BLOCK_M']))
        n_ctx = q.shape[2]
        if n_ctx <= 512:
            grid = lambda args: (triton.cdiv(q.shape[2], args['BLOCK_M']), 1, q.shape[0] * q.shape[1])
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        if os.getenv('TRITON_INTEL_ADVANCED_PATH', '0') == '0':
            # default pipeline
            _attention.tune_attn_fwd[grid](  # pylint: disable=unsubscriptable-object
                q, k, v, sm_scale, M, o,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                BLOCK_DMODEL=Lk,  #
                STAGE=stage,  #
                split_barriers_scope='None',  # possible scope value: 'Subgroup','Workgroup'
            )
        else:
            _attention.attn_fwd[grid](  # pylint: disable=unsubscriptable-object
                q, k, v, sm_scale, M, o,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1],  #
                N_CTX=q.shape[2],  #
                BLOCK_M=BLOCK_M,  #
                BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=Lk,  #
                STAGE=stage,  #
                num_warps=num_warps,  #
                num_stages=num_stages,  #
                grf_mode='large',  #
                advanced_path=True,  #
            )

        ctx.save_for_backward(q, k, v, o, M, dq, dk, dv, delta)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        # FIXME: There is no certainty as to how much such behavior is expected.
        # Consider removing `record_function` call from here once
        # https://github.com/pytorch/pytorch/issues/144778 has more details.
        with record_function(
                '__profile_kernel_of_func_bwd_fa'
        ) if benchmark_suite.BENCHMARKING_METHOD == 'UPSTREAM_PYTORCH_PROFILER' else contextlib.nullcontext():
            q, k, v, o, M, dq, dk, dv, delta = ctx.saved_tensors
            assert do.is_contiguous()
            assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
            BATCH, N_HEAD, N_CTX = q.shape[:3]
            PRE_BLOCK = 128
            NUM_WARPS, NUM_STAGES = 4, 5
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
            BLK_SLICE_FACTOR = 2
            RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
            arg_k = k
            arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
            PRE_BLOCK = 128
            assert N_CTX % PRE_BLOCK == 0
            pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
            _attn_bwd_preprocess[pre_grid](
                o, do,  #
                delta,  #
                BATCH, N_HEAD, N_CTX,  #
                BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
            )
            grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
            _attn_bwd[grid](
                q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
                M, delta,  #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                N_HEAD, N_CTX,  #
                BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
                BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
                BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
                HEAD_DIM=ctx.HEAD_DIM,  #
                num_warps=NUM_WARPS,  #
                num_stages=NUM_STAGES  #
            )

        return dq, dk, dv, None, None, None, None, None, None


attention = _attention.apply


def check_close(f_val, f_ref, atol, rtol):
    x = f_val()
    y = f_ref()
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    close = np.isclose(x, y, atol=atol, rtol=rtol)
    num_close = np.count_nonzero(close)
    num_not_close = close.size - num_close
    num_perc = num_not_close / close.size * 100
    if num_not_close != 0:
        print(f'Warning: {num_not_close}, out of {close.size} elements do not match ({num_perc:.2f}%) in XeTLA impl')


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    fa_kernel_mode='fwd',
    attn_fwd=_attn_fwd_with_block_pointers,
):
    """
    Returns a Mark object containing a Benchmark object constructed at runtime and parameterized by the provided option values.
    The benchmark can then be executed by calling the :code:`.run` method on the return value.
    """

    supported_providers = {
        'triton': 'Triton',
        'xetla': 'XeTLA',
        'cutlass': 'CUTLASS',
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    # Initialize _attention class forward kernel (untuned for the advanced path and tuned for the default path).
    _attention.attn_fwd = attn_fwd
    _attention.tune_attn_fwd = tuner(attn_fwd)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['Z', 'H', 'N_CTX', 'D_HEAD', 'CAUSAL', 'MODE'],
            x_vals=[[z, h, 16384 // z, dhead, causal, mode]
                    for z in [1, 2, 4, 8, 16, 32]
                    for (h, dhead) in [(16, 128), (32, 64)]
                    for causal in [False, True]
                    for mode in [fa_kernel_mode]]  #
            + [[4, 48, 1024, 64, causal, mode] for causal in [False, True] for mode in [fa_kernel_mode]],
            line_arg='provider',
            # argument name whose value corresponds to a different line in the plot
            # possible values for `line_arg``
            line_vals=list(providers.keys()),
            # label name for the lines
            line_names=list(providers.values()),
            # line styles
            styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
            ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
            plot_name='attn-performance',
            # name for the plot. Used also as a file name for saving the plot.
            args={},
        ))
    # pylint: disable=too-many-branches
    def benchmark(Z, H, N_CTX, D_HEAD, CAUSAL, MODE, provider):
        modes = ['fwd', 'bwd']
        if MODE not in modes:
            raise AssertionError(f'Unknown {MODE}, supported modes are {modes}')
        dtype = torch.float16
        q = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
        k = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
        v = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
        sm_scale = 0.125
        dq, dk, dv, delta = None, None, None, None
        if MODE == 'bwd':
            sm_scale = 1.3
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            delta = torch.empty_like(q)
        quantiles = [0.5, 0.0, 1.0]
        atol = 1e-1 if N_CTX == 16384 else 1e-2
        # FIXME: use torch sdpa for result check after https://github.com/intel/intel-xpu-backend-for-triton/issues/2042 fixed
        torch_fn = lambda: torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(
        ), attn_mask=None, dropout_p=0.0, is_causal=CAUSAL, scale=sm_scale).to(torch.float32)
        if MODE == 'bwd':
            torch_o = torch_fn()
            torch_do = torch.randn_like(torch_o)
            torch_fn = lambda: torch_o.backward(torch_do, retain_graph=True)

        if provider == 'onednn':
            _, min_ms, max_ms, mean, cv = benchmark_suite.do_bench(
                torch_fn,
                n_warmup=10,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'triton':
            triton_fn = lambda: attention(q, k, v, CAUSAL, sm_scale, dq, dk, dv, delta)
            if MODE == 'bwd':
                triton_o = triton_fn()
                triton_do = torch.randn_like(triton_o)
                triton_fn = lambda: triton_o.backward(triton_do, retain_graph=True)
            if MODE == 'fwd':
                benchmark_suite.assert_close(triton_fn, torch_fn, atol=atol, rtol=1e-3, err_msg='triton to torch')
            else:
                benchmark_suite.assert_close(
                    lambda: triton_o,
                    lambda: torch_o,
                    atol=1e-2,
                    rtol=0,
                    err_msg='triton to torch',
                )
            _, min_ms, max_ms, mean, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=10,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'xetla':
            if MODE == 'bwd':
                module_name = f'flash_attn_bwd_causal_{CAUSAL}'.lower()
                func = getattr(xetla_kernel, module_name)
                grad_out = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                bias = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                dropout = torch.empty_like(q, device='xpu', dtype=torch.uint8)
                out = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                log_sumexp = torch.zeros(q.size(), device='xpu', dtype=dtype, requires_grad=True)
                workspace = torch.zeros(q.size(), device='xpu', dtype=dtype, requires_grad=True)
                grad_q_tmp = torch.zeros(q.size(), device='xpu', dtype=dtype, requires_grad=True)
                alpha = sm_scale
                dropout_prob = 0
                grad_query = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                grad_key = torch.empty_like(k, device='xpu', dtype=dtype, requires_grad=True)
                grad_value = torch.empty_like(v, device='xpu', dtype=dtype, requires_grad=True)
                grad_bias = torch.empty_like(bias, device='xpu', dtype=dtype, requires_grad=True)
                bias_strideB = -1
                bias_strideN = -1
                bias_strideF = -1
                attn_mask_padding = 0

                def xetla_bwd_fn():
                    func(grad_out, q, k, v, bias, dropout, out, log_sumexp, workspace, grad_q_tmp, alpha, dropout_prob,
                         grad_query, grad_key, grad_value, grad_bias, Z, H, D_HEAD, N_CTX, N_CTX, bias_strideB,
                         bias_strideN, bias_strideF, attn_mask_padding)
                    return out

                _, min_ms, max_ms, mean, cv = benchmark_suite.do_bench(
                    xetla_bwd_fn,
                    n_warmup=10,
                    n_repeat=10,
                    quantiles=quantiles,
                )

            else:
                min_ms = float('nan')
                max_ms = float('nan')
                mean = float('nan')
                cv = float('nan')

        elif provider == 'cutlass':
            if MODE == 'fwd':
                name = 'attention'
                func = getattr(cutlass_kernel, name)
                out = torch.zeros((Z, H, N_CTX, D_HEAD), device='xpu', dtype=torch.float32, requires_grad=True)

                def cutlass_fwd_fn():
                    func(q, k, v, out, Z, H, H, N_CTX, N_CTX, D_HEAD, D_HEAD, CAUSAL, sm_scale)
                    return out

                benchmark_suite.assert_close(cutlass_fwd_fn, torch_fn, atol=atol, rtol=1e-3, err_msg='cutlass to torch')

                _, min_ms, max_ms, mean, cv = benchmark_suite.do_bench(
                    cutlass_fwd_fn,
                    n_warmup=10,
                    n_repeat=10,
                    quantiles=quantiles,
                )

            else:
                min_ms = float('nan')
                max_ms = float('nan')
                mean = float('nan')
                cv = float('nan')

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        tflops = lambda mean: 2 * 2 * Z * H * N_CTX * N_CTX * D_HEAD * (1e-12) / (mean * 1e-3)
        gbps = lambda mean: Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

        if MODE == 'bwd':
            tflops = lambda mean: 2.5 * 2 * 2 * Z * H * N_CTX * N_CTX * D_HEAD * (1e-12) / (mean * 1e-3)
            gbps = lambda mean: 2.5 * Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

        return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    _benchmark = get_benchmark(fa_kernel_mode=os.getenv('FA_KERNEL_MODE', 'fwd'), )
    _benchmark.run(show_plots=False, print_data=True)
