# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unified Attention benchmark
==========================

This benchmark is based on the test_triton_unified_attention.py tests
"""
import os
from itertools import product
from typing import Optional

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite

# Import vLLM attention functions
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.platforms import current_platform

# from vllm.platforms import current_platform
# from vllm.triton_utils import tl, triton

float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr, use_q_block_mode: tl.constexpr):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d_td(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # index of the q_block in the entire batch, have some redundancy
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    # Find the q_sequence index corresponding to the selected q_block
    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    # Index of a q_block that started the current q_sequence. If q_block_global_idx == q_block_start_idx,
    # then we are working on a first block of this q_sequence
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    # Local index of the q_block inside the current q_sequence
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    # Start index of the current q_sequence in the query_pointer
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    # Length of a current q_sequence
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    # We had some redundant q_blocks that are outside of the current q_sequence
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    q_block_local_len = min(BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q)

    # BLOCK_Q will describe how many queries we want to cover in this tile, it can be just one
    # BLOCK_Q = BLOCK_M // num_queries_per_kv
    # BLOCK_M actually shows how many q_heads will be processed, BLOCK_Q is just a derivative
    # queries_per_kv_head padded we cover all of them in one block
    offs_m = tl.arange(0, BLOCK_M)
    # head_size padded, head size is the same for qkv
    # offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    # tensor with local query positions with repeats for each kv_head
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    # query_end = q_block_local_idx * BLOCK_Q + min(BLOCK_Q, cur_batch_query_len - query_pos) // num_queries_per_kv

    #
    # query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    # query_offset = (query_offset_0[:, None] * query_stride_0 +
    #                 query_offset_1[:, None] * query_stride_1 +
    #                 offs_d[None, :])

    # So we end up with shape: (num_queries_per_kv * several q tokens, HEAD_SIZE)
    # Queries will lie like: (q_heads for token 1, q_heads for token 2, ..., HEAD_SIZE)

    # dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q shape is (q_tokens, q_heads, head_size)
    # We want to load all q_heads that correspond to the current kv_head and
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    # Block pointer
    # Shape (query_tokens, HEAD_SIZE * num_queries_per_kv)
    # stride (query_stride_0, 1)
    # Then we need to reshape it to (query_tokens * num_queries_per_kv, HEAD_SIZE)
    # Q = tl.load(
    #     query_ptr + query_offset,
    #     mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    #     other=0.0,
    # )
    q_base = (query_ptr + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0 +
              (kv_head_idx * num_queries_per_kv) * query_stride_1)
    # query_to
    q_desc = tl.make_tensor_descriptor(base=q_base, shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
                                       strides=(query_stride_0, query_stride_1, 1),
                                       block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    Q_td = q_desc.load([0, 0, 0])
    Q = Q_td.reshape(BLOCK_M, HEAD_SIZE_PADDED)
    # q_base = (query_ptr + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0 +
    #           (kv_head_idx * num_queries_per_kv) * query_stride_1)
    # q_desc = tl.make_tensor_descriptor(
    #     base=q_base,
    #     shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
    #     strides=(query_stride_0, 1),
    #     block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE)
    # )
    # Q_raw = q_desc.load([0, 0])  # Shape: (BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED)

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0)  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles (blocks) that need to be processed to
    # cover the longest sequence prefix (due to causal masking, blocks beyond
    # this prefix can be skipped)
    num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        # v_offset = (physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 +
        #             offs_d[None, :] * stride_v_cache_3 + offs_n[:, None] * stride_v_cache_1)

        # K shape (N_BLOCKS, BLOCK_SIZE, kv_heads, HEAD_SIZE)
        # k_offset = (physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 +
        #             offs_d[:, None] * stride_k_cache_3 + offs_n[None, :] * stride_k_cache_1)

        v_base = value_cache_ptr + physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
        v_desc = tl.make_tensor_descriptor(base=v_base, shape=(BLOCK_SIZE, HEAD_SIZE),
                                           strides=(stride_v_cache_1, stride_v_cache_3),
                                           block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED))

        k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
        # k_desc = tl.make_tensor_descriptor(base=k_base, shape=(HEAD_SIZE, BLOCK_SIZE),
        #                                    strides=(stride_k_cache_3, stride_k_cache_1),
        #                                    block_shape=(HEAD_SIZE_PADDED, BLOCK_SIZE))
        k_desc = tl.make_tensor_descriptor(base=k_base, shape=(BLOCK_SIZE, HEAD_SIZE),
                                           strides=(stride_k_cache_1, stride_k_cache_3),
                                           block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED))
        # K : (HEAD_SIZE, BLOCK_SIZE)
        # K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)
        K_load = k_desc.load([0, 0]).T

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        # V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)
        V_load = v_desc.load([0, 0])

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + offs_n

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, BLOCK_SIZE)
        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, BLOCK_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # output_offset = (query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 +
    #                  offs_d[None, :])

    output_offset = (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * output_stride_0 + (
        kv_head_idx * num_queries_per_kv) * output_stride_1
    output_base = output_ptr + output_offset
    output_desc = tl.make_tensor_descriptor(base=output_base, shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
                                            strides=(output_stride_0, output_stride_1, 1),
                                            block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    # output_desc.store([0, 0, 0], acc.view((BLOCK_M // num_queries_per_kv, num_queries_per_kv, HEAD_SIZE_PADDED)))
    output_desc.store([0, 0, 0], acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    # tl.store(
    #     output_ptr + output_offset,
    #     acc,
    #     mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    # )


@triton.jit
def kernel_unified_attention_3d_td(segm_output_ptr,
                                   # [num_tokens, num_query_heads, num_segments, head_size]
                                   segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
                                   segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
                                   query_ptr,  # [num_tokens, num_query_heads, head_size]
                                   key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
                                   value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
                                   sink_ptr,  # [num_query_heads]
                                   block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
                                   seq_lens_ptr,  # [num_seqs]
                                   alibi_slopes_ptr,  # [num_query_heads]
                                   qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
                                   scale,  # float32
                                   k_scale,  # float32
                                   v_scale,  # float32
                                   softcap,  # float32
                                   num_query_heads: tl.constexpr,  # int
                                   num_queries_per_kv: tl.constexpr,  # int
                                   block_table_stride: tl.int64,  # int
                                   query_stride_0: tl.int64,  # int
                                   query_stride_1: tl.int64,  # int, should be equal to head_size
                                   qq_bias_stride_0: tl.int64,  # int
                                   BLOCK_SIZE: tl.constexpr,  # int
                                   HEAD_SIZE: tl.constexpr,  # int
                                   HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
                                   USE_ALIBI_SLOPES: tl.constexpr,  # bool
                                   USE_QQ_BIAS: tl.constexpr,  # bool
                                   USE_SOFTCAP: tl.constexpr,  # bool
                                   USE_SINKS: tl.constexpr,  # bool
                                   SLIDING_WINDOW: tl.constexpr,  # int
                                   stride_k_cache_0: tl.int64,  # int
                                   stride_k_cache_1: tl.int64,  # int
                                   stride_k_cache_2: tl.int64,  # int
                                   stride_k_cache_3: tl.constexpr,  # int
                                   stride_v_cache_0: tl.int64,  # int
                                   stride_v_cache_1: tl.int64,  # int
                                   stride_v_cache_2: tl.int64,  # int
                                   stride_v_cache_3: tl.constexpr,  # int
                                   query_start_len_ptr,  # [num_seqs+1]
                                   BLOCK_Q: tl.constexpr,  # int
                                   num_seqs: tl.int32, BLOCK_M: tl.constexpr,  # int
                                   NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
                                   ):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    q_block_local_len = min(BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    if segm_idx * blocks_per_segment * BLOCK_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Load Q using tensor descriptor (same as 2D case)
    q_base = (query_ptr + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0 +
              (kv_head_idx * num_queries_per_kv) * query_stride_1)
    q_desc = tl.make_tensor_descriptor(base=q_base, shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
                                       strides=(query_stride_0, query_stride_1, 1),
                                       block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    Q_td = q_desc.load([0, 0, 0])
    Q = Q_td.reshape(BLOCK_M, HEAD_SIZE_PADDED)

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0)  # shape: [BLOCK_M]

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles within current segment
    for j in range(
            segm_idx * blocks_per_segment,
            min((segm_idx + 1) * blocks_per_segment, num_blocks),
    ):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        # Use tensor descriptors for V and K (same as 2D case)
        v_base = value_cache_ptr + physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
        v_desc = tl.make_tensor_descriptor(base=v_base, shape=(BLOCK_SIZE, HEAD_SIZE),
                                           strides=(stride_v_cache_1, stride_v_cache_3),
                                           block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED))

        k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
        k_desc = tl.make_tensor_descriptor(base=k_base, shape=(BLOCK_SIZE, HEAD_SIZE),
                                           strides=(stride_k_cache_1, stride_k_cache_3),
                                           block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED))

        K_load = k_desc.load([0, 0]).T

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = v_desc.load([0, 0])

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + offs_n

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, BLOCK_SIZE)
        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # Store output using tensor descriptor (similar to 2D case but for segments)
    segm_output_offset = (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * (
        num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + (kv_head_idx * num_queries_per_kv) * (
            NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + segm_idx * HEAD_SIZE_PADDED

    segm_output_base = segm_output_ptr + segm_output_offset
    segm_output_desc = tl.make_tensor_descriptor(
        base=segm_output_base, shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED, NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    segm_output_desc.store([0, 0, 0], acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))

    segm_offset = (cur_batch_in_all_start_index + query_pos).to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ) + \
                   query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments_td(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    #[num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full([NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32)
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ) +
                   query_head_idx * NUM_SEGMENTS_PER_SEQ + tl.arange(0, NUM_SEGMENTS_PER_SEQ))
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
                          query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) +
                          tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED +
                          tl.arange(0, HEAD_SIZE_PADDED)[None, :])
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (query_token_idx * output_stride_0 + query_head_idx * output_stride_1 +
                     tl.arange(0, HEAD_SIZE_PADDED))
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention_td(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
):
    """
    Tensors:
    q: Query tensor [num_tokens, num_query_heads, head_size] - The query embeddings
    k: Key cache [num_blocks, block_size, num_kv_heads, head_size] - Paged key cache
    v: Value cache [num_blocks, block_size, num_kv_heads, head_size] - Paged value cache
    out: Output tensor [num_tokens, num_query_heads, head_size] - Where results are written
    cu_seqlens_q: Cumulative query sequence lengths [num_seqs+1] - Used to identify which sequence each token belongs to
    seqused_k: Actual KV sequence lengths [num_seqs] - Length of KV cache for each sequence
    block_table: Block table [num_seqs, max_num_blocks_per_seq] - Maps logical blocks to physical blocks in the paged KV cache

    Scalar Arguments:
    max_seqlen_q: Maximum query sequence length in the batch
    max_seqlen_k: Maximum KV sequence length in the batch
    softmax_scale: Scaling factor for attention scores (typically 1/sqrt(head_size))
    causal: Boolean - must be True (only causal masking supported)
    window_size: Tuple for sliding window attention (left, right)
    softcap: Soft capping value for attention scores (0 = disabled)

    Optional Scaling Arguments (for FP8):
    q_descale: Query descaling factor (not supported, must be None)
    k_descale: Key descaling factor for FP8
    v_descale: Value descaling factor for FP8
    output_scale: Output scaling factor for FP8

    Optional Feature Arguments:
    alibi_slopes: ALiBi attention bias slopes [num_query_heads]
    qq_bias: Query-query attention bias [num_query_tokens, num_query_tokens]
    sinks: Sink token values [num_query_heads] - special tokens that attend to everything
    """

    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, \
        "Block size must be at least 32 for fp8"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], \
        "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    assert num_query_heads % num_kv_heads == 0, \
        "num_query_heads must be divisible by num_kv_heads"
    head_size = q.shape[2]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    # if batch contains a prefill
    if max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128:
        print("Calling 2d")
        kernel_unified_attention_2d_td[(
            total_num_q_blocks,
            num_kv_heads,
        )](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )
    else:
        # for initial version, NUM_SEGMENTS = 16 is chosen as a default
        # value that showed good performance in tests
        # print("Calling 3d")
        NUM_SEGMENTS = 16

        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        kernel_unified_attention_3d_td[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        )

        reduce_segments_td[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            USE_FP8=output_scale is not None,
        )


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].reshape(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].reshape(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# Benchmark configurations for unified attention
# (seq_lens, num_heads, head_size, block_size, dtype, sliding_window, soft_cap)
# NUM_HEADS = [(4, 4), (8, 2)]
# HEAD_SIZES = [128, 256]
PAGED_ATTENTION_MMAP_SIZE = 16
# Models
MODEL_CONFIGS = [
    # q_heads, k_heads, head_size, dtype, qdtype
    # llama3-8B
    (32, 8, 128, torch.bfloat16, None),
    # llama3-70B
    # (64, 8, 128, torch.bfloat16, None)
]

# QDTYPES = [None, torch.float8_e4m3fn] if not current_platform.is_rocm() else [
#     None, torch.float8_e4m3fnuz
# ]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
# NUM_BLOCKS = [32768]  #, 2048]
SEQ_LENS = [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
SEQ_LENS = [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)],
            [(1, k) for k in [1513, 245, 102, 123, 3454, 434, 345, 34]]]
# SEQ_LENS = [[(1, 1328), (5, 18), (129, 463)]]  #, [(1, 523), (1, 37), (1, 2011)]]
# SOFT_CAPS = [None, 50.0]
SOFT_CAPS = [None]
# SLIDING_WINDOWS = [None, 256]
SLIDING_WINDOWS = [None]
ATTENTION_CONFIGS_BF16 = []
for model_config, seq_len, sliding_window, soft_cap, num_blocks in product(MODEL_CONFIGS, SEQ_LENS, SLIDING_WINDOWS,
                                                                           SOFT_CAPS, NUM_BLOCKS):
    if model_config[-1] is not None and model_config[-1].itemsize < 2 and PAGED_ATTENTION_MMAP_SIZE < 32:
        print("Skipping configuration due to incompatible q_dtype and block_size.")
        continue
    ATTENTION_CONFIGS_BF16.append((
        *model_config,
        seq_len,
        sliding_window,
        soft_cap,
        num_blocks,
        PAGED_ATTENTION_MMAP_SIZE,
    ))

# ATTENTION_CONFIGS_FP8 = [
#     # FP8 configurations
#     (1, 64, 512, 8, 8, 128, 32, torch.float8_e4m3fn, None, None),
#     (4, 128, 1024, 16, 4, 128, 32, torch.float8_e4m3fn, None, None),
#     (8, 256, 2048, 32, 8, 256, 32, torch.float8_e4m3fn, None, None),
# ]

DEVICE_NAME = torch.xpu.get_device_name()
# DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory

# def is_enough_memory(x_val):
#     num_seqs, max_query_len, max_kv_len, num_query_heads, num_kv_heads, head_size, block_size, dtype, sliding_window, soft_cap = x_val
#     # Query: (total_query_tokens, num_query_heads, head_size)
#     # Key/Value cache: (num_blocks, block_size, num_kv_heads, head_size) each
#     # Output: (total_query_tokens, num_query_heads, head_size)

#     total_query_tokens = num_seqs * max_query_len
#     num_blocks = (num_seqs * max_kv_len + block_size - 1) // block_size

#     n_bytes = dtype.itemsize if hasattr(dtype, 'itemsize') else 2

#     required_memory = (
#         total_query_tokens * num_query_heads * head_size * n_bytes +  # Query
#         2 * num_blocks * block_size * num_kv_heads * head_size * n_bytes +  # Key + Value cache
#         total_query_tokens * num_query_heads * head_size * 2 +  # Output (bf16)
#         num_seqs * 128  # Metadata overhead
#     )

#     enough_memory = required_memory < DEVICE_TOTAL_MEMORY * 0.8  # Use 80% of memory
#     if not enough_memory:
#         print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
#     return enough_memory

# ATTENTION_CONFIGS_BF16 = [x_val for x_val in ATTENTION_CONFIGS_BF16 if is_enough_memory(x_val)]


def get_unified_attention_benchmark(
    providers_filter: Optional[list[str]] = None,
    fp8=False,
    plot_name: str = 'unified-attention-performance',
):
    """
    Returns a Mark object containing a Benchmark object for unified attention.
    """
    supported_providers = {
        'triton': 'triton',
        'triton-td': 'triton-td',
        'pytorch': 'pytorch',
    }
    if os.getenv("TRITON_INTERPRET", "0") == "1":
        # Skip triton providers if interpreter is used
        del supported_providers['triton']

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    ATTENTION_CONFIGS_FP8 = []
    configs = ATTENTION_CONFIGS_FP8 if fp8 else ATTENTION_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=[
                'q_heads', 'k_heads', 'head_size', 'dtype', 'qdtype', 'seq_lens', 'sliding_window', 'soft_cap',
                'num_blocks', 'block_size'
            ],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('orange', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name=plot_name,
            args={},
        ))
    def benchmark(q_heads, k_heads, head_size, dtype, qdtype, seq_lens, sliding_window, soft_cap, num_blocks,
                  block_size, provider):
        # Set default device like in the test
        current_platform.seed_everything(0)  # Use same seed as test
        n_warmup = 100
        quantiles = [0.5, 0.0, 1.0]

        torch.set_default_device("xpu")

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        assert q_heads % k_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = ((sliding_window - 1, 0) if sliding_window is not None else (-1, -1))
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), q_heads, head_size, dtype=dtype)
        key_cache = torch.randn(num_blocks, block_size, k_heads, head_size, dtype=dtype)
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
        kv_lens_list = kv_lens  # Preserve the original list
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

        output = torch.empty_like(query)

        maybe_quantized_query = query
        maybe_quantized_key_cache = key_cache
        maybe_quantized_value_cache = value_cache
        q_descale = None
        k_descale = None
        v_descale = None
        if qdtype is not None:
            # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
            maybe_quantized_query = query.to(qdtype)
            maybe_quantized_key_cache = key_cache.to(qdtype)
            maybe_quantized_value_cache = value_cache.to(qdtype)

            scale_shape = (num_seqs, k_heads)
            q_descale = None  # Not yet supported
            k_descale = torch.rand(scale_shape, dtype=torch.float32)
            v_descale = torch.rand(scale_shape, dtype=torch.float32)

        def torch_fn():
            return ref_paged_attn(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=query_lens,
                kv_lens=kv_lens,
                block_tables=block_tables,
                scale=scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
            )

        if provider == 'pytorch':
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                torch_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider.startswith('triton'):
            # First run unified_attention exactly like in the test
            if provider == 'triton':
                fn = unified_attention
            elif provider == 'triton-td':
                fn = unified_attention_td
            else:
                raise ValueError(f'Unsupported provider {provider}')

            def triton_fn():
                fn(
                    q=maybe_quantized_query,
                    k=maybe_quantized_key_cache,
                    v=maybe_quantized_value_cache,
                    out=output,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=kv_lens,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=True,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output

            atol, rtol = 1.5e-2, 1e-2
            if dtype != torch.bfloat16:
                atol, rtol = 1.5e-1, 1.5e-1
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=atol, rtol=rtol, err_msg='triton to torch')

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )
        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        # Calculate performance metrics
        def gbps(ms):
            # n_bytes = dtype.itemsize if hasattr(dtype, 'itemsize') else 2
            # Memory: Query + Key cache + Value cache + Output
            # total_bytes = (
            #     total_query_tokens * num_query_heads * head_size * n_bytes +  # Query
            #     sum(kv_lens) * num_kv_heads * head_size * n_bytes * 2 +      # KV cache accessed
            #     total_query_tokens * num_query_heads * head_size * 2          # Output
            # )
            total_bytes = 1
            return total_bytes * (1e-9) / (ms * 1e-3)

        def tflops(ms):
            # Attention FLOPs: Q@K (2*d*seq_len*kv_len) + Softmax (~seq_len*kv_len) + Attn@V (2*d*seq_len*kv_len)
            total_flops = 0
            for i, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens_list)):
                # Q@K^T and Attn@V operations
                flops_per_head = 2 * head_size * q_len * kv_len * 2  # 2 matmuls
                total_flops += flops_per_head * q_heads
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    # Run unified attention benchmark
    print('Running unified attention benchmark...')
    _benchmark_attention = get_unified_attention_benchmark(fp8=(os.getenv('FP8', '0') == '1'), )
    _benchmark_attention.run(show_plots=False, print_data=True)
