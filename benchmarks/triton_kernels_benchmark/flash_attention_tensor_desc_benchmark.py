import os
from typing import Optional

import triton
import triton.language as tl

from triton_kernels_benchmark import flash_attention_benchmark


# pylint: disable=unused-argument
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, start_m, qk_scale,  #
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
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([0, offsetk_y])
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
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([offsetv_y, 0])
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_with_tensor_desc(Q, K, V, sm_scale, M, Out,  #
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

    # tensor descriptors
    y_dim = Z * H * N_CTX
    desc_q = tl.make_tensor_descriptor(base=Q, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
                                       block_shape=(BLOCK_M, BLOCK_DMODEL))
    desc_v = tl.make_tensor_descriptor(base=V, shape=(y_dim, BLOCK_DMODEL), strides=(BLOCK_DMODEL, 1),
                                       block_shape=(BLOCK_N, BLOCK_DMODEL))
    desc_k = tl.make_tensor_descriptor(base=K, shape=(BLOCK_DMODEL, y_dim), strides=(1, BLOCK_DMODEL),
                                       block_shape=(BLOCK_DMODEL, BLOCK_N))
    desc_o = tl.make_tensor_descriptor(base=Out + qvk_offset, shape=(N_CTX, BLOCK_DMODEL),
                                       strides=(stride_om, stride_on), block_shape=(BLOCK_M, BLOCK_DMODEL))

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
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
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3, the kernel gets 1 as its STAGE
    # For causal = False, STAGE = 1, the kernel gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, desc_k, desc_v,  #
                                        offset_y, start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, desc_k, desc_v,  #
                                        offset_y, start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    desc_o.store([start_m * BLOCK_M, 0], acc.to(Out.type.element_ty))


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    fa_kernel_mode='fwd',
    xetla_assert_result=False,
    xetla_warn_mismatch=True,
):
    return flash_attention_benchmark.get_benchmark(
        providers_filter=providers_filter,
        fa_kernel_mode=fa_kernel_mode,
        attn_fwd=_attn_fwd_with_tensor_desc,
        xetla_assert_result=xetla_assert_result,
        xetla_warn_mismatch=xetla_warn_mismatch,
    )


if __name__ == '__main__':
    _benchmark = get_benchmark(
        fa_kernel_mode=os.getenv('FA_KERNEL_MODE', 'fwd'),
        xetla_assert_result=(os.getenv('XETLA_ASSERT_RESULT', '0') == '1'),
        xetla_warn_mismatch=(os.getenv('XETLA_WARN_MISMATCH', '1') == '1'),
    )
    _benchmark.run(show_plots=False, print_data=True)
