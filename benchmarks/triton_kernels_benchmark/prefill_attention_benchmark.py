# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suit


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :])
    off_k = offs_n[None, :] * stride_kbs + \
        cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + \
        cur_kv_head * stride_vh + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    end_n = (cur_batch_seq_len if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len))
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if IS_CAUSAL:
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_seq_len)
                & (offs_m[:, None] >= (start_n + offs_n[None, :])),
                0,
                float('-inf'),
            )
        else:
            qk += tl.where((start_n + offs_n[None, :]) < cur_batch_seq_len, 0, float('-inf'))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :])
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]))


def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True):
    BLOCK_M = 128
    BLOCK_N = 64

    # Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    Lq, Lk = q.shape[-1], k.shape[-1]

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))
    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )

    return o


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['BATCH', 'SEQ_LENS', 'Q_HEAD_NUM', 'KV_HEAD_NUM', 'HEAD_DIM', 'CAUSAL', 'MODE', 'VALIDATE'],
        x_vals=[  #
            [bs, [1024], 32, 8, 128, causal, 'fwd', False] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  # noqa
            [bs, [1024], 32, 32, 96, causal, 'fwd', False] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  # noqa
            [bs, [1024], 28, 4, 128, causal, 'fwd', False]
            for causal in [True, False]
            for bs in [1, 16, 32, 64, 128]  # noqa
        ] + [
            # [4, [1024], 48, 48, 64, causal, 'fwd', True] for causal in [True, False]
            # ] + [
            # [bs, [16384 // bs], h, h, dhead, causal, 'fwd', True] for bs in [1, 2, 4, 8, 16, 32] for (h, dhead) in [(16, 128), (32, 64)] for causal in [False, True]
        ],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=[
            'triton',
        ],
        # label name for the lines
        line_names=[
            'Triton',
        ],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='prefill-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(BATCH, SEQ_LENS, Q_HEAD_NUM, KV_HEAD_NUM, HEAD_DIM, CAUSAL, MODE, VALIDATE, provider):
    dtype = torch.bfloat16
    device = 'xpu'
    N_CTX = sum(SEQ_LENS)
    max_seq_len = max(SEQ_LENS)

    # Create random input tensors
    q = torch.randn((BATCH * N_CTX, Q_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((BATCH * N_CTX, KV_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((BATCH * N_CTX, KV_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    o = torch.zeros((BATCH * N_CTX, Q_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)

    # Create b_start_loc and b_seq_len tensors
    b_start_loc = torch.tensor([0, SEQ_LENS[0]], device=device)
    b_seq_len = torch.tensor(SEQ_LENS, device=device)

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':
        triton_fn = lambda: context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=CAUSAL)

        if VALIDATE:
            # FIXME: use torch sdpa for result check after https://github.com/intel/intel-xpu-backend-for-triton/issues/2042 fixed
            atol = 1e-1 if N_CTX == 16384 else 1e-2
            torch_fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q.cpu().permute(1, 0, 2),
                k.cpu().permute(1, 0, 2),
                v.cpu().permute(1, 0, 2), is_causal=CAUSAL).permute(1, 0, 2).to(torch.float32)
            benchmark_suit.assert_close(triton_fn, torch_fn, atol=atol, rtol=1e-3, err_msg='triton to torch')

        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: 2 * BATCH * (Q_HEAD_NUM + KV_HEAD_NUM) * N_CTX * N_CTX * HEAD_DIM * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: 2 * BATCH * (Q_HEAD_NUM + KV_HEAD_NUM) * N_CTX * HEAD_DIM * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
