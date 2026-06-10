# Based on vLLM unified attention kernel with Intel XPU tensor descriptor optimizations baked in.
# Source: vllm/v1/attention/ops/triton_unified_attention.py
# Patch: benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch
# Full-featured: alibi, softcap, qq_bias, sinks, mm_prefix, kv_quant, chunk_lookback.

import torch
import triton
import triton.language as tl

float8_info = torch.finfo(torch.float8_e4m3fn)

KV_QUANT_NONE = 0
KV_QUANT_FP8_PER_TENSOR = 1
KV_QUANT_INT8_PER_TOKEN_HEAD = 2
KV_QUANT_FP8_PER_TOKEN_HEAD = 3


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def fast_exp(x):
    return tl.math.exp2(x * 1.4426950408889634)


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = fast_exp(Sdiv)
    p2 = fast_exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def _prepare_kv_tile(
    data,
    Q,
    tensor_scale,
    scale_cache_ptr,
    physical_block_idx,
    seq_offset,
    kv_head_idx,
    stride_s_blk,
    stride_s_slot,
    stride_s_head,
    tile_mask,
    BLOCK_SIZE: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
):
    unused_scales = tile_mask.to(tl.float32)

    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype), unused_scales
        return (data.to(tl.float32) * tensor_scale).to(Q.dtype), unused_scales
    if KV_QUANT_MODE >= 2:
        scale_idx = (
            physical_block_idx * stride_s_blk
            + (seq_offset % BLOCK_SIZE) * stride_s_slot
            + kv_head_idx * stride_s_head
        )
        token_head_scales = tl.load(
            scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
        ).to(tl.float32)
        return data.to(Q.dtype), token_head_scales
    return data.to(Q.dtype), unused_scales


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
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


# ---------------------------------------------------------------------------
# 2D kernel: used for prefill, large decode batches, and sliding-window <=1024
# From t83/t10: _prepare_kv_tile for ALL KQM modes (no inline bypass)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=[],
)
@triton.jit
def kernel_unified_attention_2d(
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
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    stride_ks_blk=0,
    stride_ks_slot=0,
    stride_ks_head=0,
    stride_vs_blk=0,
    stride_vs_slot=0,
    stride_vs_head=0,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    tl.static_assert(BLOCK_SIZE % TILE_SIZE == 0, "BLOCK_SIZE must be multiple of TILE_SIZE")
    tl.static_assert(BLOCK_SIZE >= TILE_SIZE, "BLOCK_SIZE must be >= TILE_SIZE")
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    q_block_local_len = min(BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q)

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED) via tensor descriptor
    q_base = (query_ptr
              + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
              + (kv_head_idx * num_queries_per_kv) * query_stride_1)
    if HEAD_SIZE == HEAD_SIZE_PADDED:
        q_desc = tl.make_tensor_descriptor(
            base=q_base,
            shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
            strides=(query_stride_0, 1),
            block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED))
        Q_raw = q_desc.load([0, 0])
    else:
        q_desc = tl.make_tensor_descriptor(
            base=q_base,
            shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
            strides=(query_stride_0, query_stride_1, 1),
            block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
        Q_raw = q_desc.load([0, 0, 0])
    Q = Q_raw.reshape(BLOCK_M, HEAD_SIZE_PADDED)

    Q = (Q * scale).to(Q.dtype)

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

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    if USE_MM_PREFIX:
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    query_abs_pos = context_len + query_pos[:, None]

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + j // (BLOCK_SIZE // TILE_SIZE)
        ).to(tl.int64)

        offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE

        # K via tensor descriptor: (HEAD_SIZE_PADDED, TILE_SIZE) after transpose
        k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
        k_desc = tl.make_tensor_descriptor(
            base=k_base, shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_k_cache_1, stride_k_cache_3),
            block_shape=(TILE_SIZE, HEAD_SIZE_PADDED))
        K_load = k_desc.load([offset_in_block, 0]).T

        # From t83/t10: use _prepare_kv_tile for ALL KQM modes (no inline bypass)
        K, k_token_head_scales = _prepare_kv_tile(
            K_load, Q, k_scale, k_scale_cache_ptr,
            physical_block_idx, seq_offset, kv_head_idx,
            stride_ks_blk, stride_ks_slot, stride_ks_head,
            tile_mask, BLOCK_SIZE, KV_QUANT_MODE,
        )

        # V via tensor descriptor: (TILE_SIZE, HEAD_SIZE_PADDED)
        v_base = value_cache_ptr + physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
        v_desc = tl.make_tensor_descriptor(
            base=v_base, shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_v_cache_1, stride_v_cache_3),
            block_shape=(TILE_SIZE, HEAD_SIZE_PADDED))
        V_load = v_desc.load([offset_in_block, 0])

        V, v_token_head_scales = _prepare_kv_tile(
            V_load, Q, v_scale, v_scale_cache_ptr,
            physical_block_idx, seq_offset, kv_head_idx,
            stride_vs_blk, stride_vs_slot, stride_vs_head,
            tile_mask, BLOCK_SIZE, KV_QUANT_MODE,
        )

        seq_mask = seq_offset[None, :] <= query_abs_pos

        if CHUNK_LOOKBACK > -1:
            seq_mask = seq_mask & (
                (
                    query_abs_pos // CHUNK_SIZE
                    - (seq_offset[None, :] // CHUNK_SIZE)
                )
                <= CHUNK_LOOKBACK
            )
        elif SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )
                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        if KV_QUANT_MODE >= 2:
            S += tl.dot(Q, K) * k_token_head_scales[None, :]
        else:
            S += tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - query_abs_pos
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],
                other=0.0,
            )
            S += qq_bias

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = fast_exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = fast_exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if KV_QUANT_MODE >= 2:
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # From t83/t10: standard epilogue (_prepare_kv_tile already applied scales)
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # Output store via tensor descriptor
    output_offset = (
        (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * output_stride_0
        + (kv_head_idx * num_queries_per_kv) * output_stride_1
    )
    output_base = output_ptr + output_offset
    output_desc = tl.make_tensor_descriptor(
        base=output_base,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(output_stride_0, output_stride_1, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    acc = acc.to(output_ptr.dtype.element_ty)
    output_desc.store([0, 0, 0], acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))


# ---------------------------------------------------------------------------
# 3D kernel: used for small-batch decode (parallel softmax segments)
# From t60: broad autotune (18 configs w/ warp_size=16 + grf_mode=256)
# From t83/t79: KQM=1 inline bypass with deferred v_scale in epilogue
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({'grf_mode': '256'}, num_warps=2, num_stages=2),
        triton.Config({'grf_mode': '256'}, num_warps=2, num_stages=3),
        triton.Config({'grf_mode': '256'}, num_warps=4, num_stages=2),
        triton.Config({'grf_mode': '256'}, num_warps=4, num_stages=3),
        triton.Config({'grf_mode': '256'}, num_warps=8, num_stages=2),
        triton.Config({'grf_mode': '256'}, num_warps=8, num_stages=3),
        triton.Config({'warp_size': 16}, num_warps=4, num_stages=2),
        triton.Config({'warp_size': 16}, num_warps=4, num_stages=3),
        triton.Config({'warp_size': 16}, num_warps=8, num_stages=2),
        triton.Config({'warp_size': 16}, num_warps=8, num_stages=3),
        triton.Config({'grf_mode': '256', 'warp_size': 16}, num_warps=4, num_stages=2),
        triton.Config({'grf_mode': '256', 'warp_size': 16}, num_warps=8, num_stages=2),
    ],
    key=[],
)
@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,  # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
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
    softcap,  # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    KV_QUANT_MODE: tl.constexpr = 0,
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    stride_ks_blk=0,
    stride_ks_slot=0,
    stride_ks_head=0,
    stride_vs_blk=0,
    stride_vs_slot=0,
    stride_vs_head=0,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    tl.static_assert(BLOCK_SIZE % TILE_SIZE == 0, "BLOCK_SIZE must be multiple of TILE_SIZE")
    tl.static_assert(BLOCK_SIZE >= TILE_SIZE, "BLOCK_SIZE must be >= TILE_SIZE")
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    q_block_local_len = tl.minimum(BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED) via tensor descriptor
    q_base = (query_ptr
              + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
              + (kv_head_idx * num_queries_per_kv) * query_stride_1)
    if HEAD_SIZE == HEAD_SIZE_PADDED:
        q_desc = tl.make_tensor_descriptor(
            base=q_base,
            shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
            strides=(query_stride_0, 1),
            block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED))
        Q_raw = q_desc.load([0, 0])
    else:
        q_desc = tl.make_tensor_descriptor(
            base=q_base,
            shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
            strides=(query_stride_0, query_stride_1, 1),
            block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
        Q_raw = q_desc.load([0, 0, 0])
    Q = Q_raw.reshape(BLOCK_M, HEAD_SIZE_PADDED)

    Q = (Q * scale).to(Q.dtype)

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

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    query_abs_pos = context_len + query_pos[:, None]

    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + j // (BLOCK_SIZE // TILE_SIZE)
        ).to(tl.int64)

        offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE

        # K via tensor descriptor: (HEAD_SIZE_PADDED, TILE_SIZE) after transpose
        k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
        k_desc = tl.make_tensor_descriptor(
            base=k_base, shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_k_cache_1, stride_k_cache_3),
            block_shape=(TILE_SIZE, HEAD_SIZE_PADDED))
        K_load = k_desc.load([offset_in_block, 0]).T

        # From t83/t79: KQM=1 inline bypass for 3D (deferred v_scale in epilogue)
        if KV_QUANT_MODE == 1:
            K = K_load.to(Q.dtype)
            k_token_head_scales = tile_mask.to(tl.float32)
        else:
            K, k_token_head_scales = _prepare_kv_tile(
                K_load, Q, k_scale, k_scale_cache_ptr,
                physical_block_idx, seq_offset, kv_head_idx,
                stride_ks_blk, stride_ks_slot, stride_ks_head,
                tile_mask, BLOCK_SIZE, KV_QUANT_MODE,
            )

        # V via tensor descriptor: (TILE_SIZE, HEAD_SIZE_PADDED)
        v_base = value_cache_ptr + physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
        v_desc = tl.make_tensor_descriptor(
            base=v_base, shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_v_cache_1, stride_v_cache_3),
            block_shape=(TILE_SIZE, HEAD_SIZE_PADDED))
        V_load = v_desc.load([offset_in_block, 0])

        if KV_QUANT_MODE == 1:
            V = V_load.to(Q.dtype)
            v_token_head_scales = tile_mask.to(tl.float32)
        else:
            V, v_token_head_scales = _prepare_kv_tile(
                V_load, Q, v_scale, v_scale_cache_ptr,
                physical_block_idx, seq_offset, kv_head_idx,
                stride_vs_blk, stride_vs_slot, stride_vs_head,
                tile_mask, BLOCK_SIZE, KV_QUANT_MODE,
            )

        seq_mask = seq_offset[None, :] <= query_abs_pos

        if CHUNK_LOOKBACK > -1:
            seq_mask = seq_mask & (
                (
                    query_abs_pos // CHUNK_SIZE
                    - (seq_offset[None, :] // CHUNK_SIZE)
                )
                <= CHUNK_LOOKBACK
            )
        elif SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )
                is_valid = range_start < range_end
                q_in_range = (
                    (query_abs_pos >= range_start)
                    & (query_abs_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (seq_offset[None, :] >= range_start)
                    & (seq_offset[None, :] <= range_end)
                    & is_valid
                )
                seq_mask |= q_in_range & k_in_range

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        if KV_QUANT_MODE >= 2:
            S += tl.dot(Q, K) * k_token_head_scales[None, :]
        else:
            S += tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            if USE_ALIBI_SQRT:
                relative_pos = seq_offset - query_abs_pos
                alibi_offset = tl.where(
                    relative_pos <= 0,
                    -tl.sqrt((-relative_pos).to(tl.float32)),
                    0.0,
                )
            else:
                alibi_offset = seq_offset - context_len
            S += alibi_slope[:, None] * alibi_offset

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],
                other=0.0,
            )
            S += qq_bias

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = fast_exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = fast_exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if KV_QUANT_MODE >= 2:
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # Segment output store: apply deferred v_scale for KQM=1 (from t83/t79)
    if KV_QUANT_MODE == 1:
        acc = acc * v_scale
    acc = acc.to(segm_output_ptr.dtype.element_ty)
    segm_output_offset = (
        (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + (kv_head_idx * num_queries_per_kv)
        * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
    )
    segm_output_desc = tl.make_tensor_descriptor(
        base=segm_output_ptr + segm_output_offset,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(
            num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
            NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
            1,
        ),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))
    segm_output_desc.store([0, 0, 0], acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED))

    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# ---------------------------------------------------------------------------
# Reduce kernel: merges parallel softmax segments into final output
# ---------------------------------------------------------------------------
@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,  # [num_tokens, num_query_heads, max_num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    num_query_heads: tl.constexpr,
    out_scale_inv,  # float32
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    block_table_stride: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * fast_exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= fast_exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------
def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    if _is_gemma3_attention(head_size, sliding_window):
        return 32
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


# ---------------------------------------------------------------------------
# Launcher — crossover t60+t83:
#   t60: broad 3D autotune (18 configs w/ warp_size=16 + grf_mode=256)
#   t83: 2D _prepare_kv_tile for all KQM modes + standard epilogue,
#        KQM-aware segment dispatch, 3D KQM=1 inline + deferred v_scale
# ---------------------------------------------------------------------------
def unified_attention(
    q, k, v, out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    block_table,
    softmax_scale,
    window_size,
    softcap=0.0,
    k_descale=1.0,
    v_descale=1.0,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    kv_quant_mode=KV_QUANT_NONE,
    k_scale_cache=None,
    v_scale_cache=None,
    chunk_lookback=-1,
):
    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    # XPU simplified tile sizes
    if q.element_size() == 1:
        TILE_SIZE_PREFILL = TILE_SIZE_DECODE = 32
    else:
        TILE_SIZE_PREFILL = TILE_SIZE_DECODE = 16
    assert TILE_SIZE_PREFILL <= block_size, "TILE_SIZE must be <= block_size"
    assert block_size % TILE_SIZE_PREFILL == 0, "block_size must be multiple of TILE_SIZE"

    seq_threshold_3D = 32
    use_2d = (max_seqlen_q > 1 or num_seqs > seq_threshold_3D
              or 0 < sliding_window_val <= 1024)
    BLOCK_M = max(32 if use_2d else 16, triton.next_power_of_2(num_queries_per_kv))
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    if use_2d:
        # 2D kernel path
        kernel_unified_attention_2d[(total_num_q_blocks, num_kv_heads)](
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
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=(1 + window_size[0]),
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
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
            KV_QUANT_MODE=kv_quant_mode,
            k_scale_cache_ptr=k_scale_cache,
            v_scale_cache_ptr=v_scale_cache,
            stride_ks_blk=k_scale_cache.stride(0) if k_scale_cache is not None else 0,
            stride_ks_slot=k_scale_cache.stride(1) if k_scale_cache is not None else 0,
            stride_ks_head=k_scale_cache.stride(2) if k_scale_cache is not None else 0,
            stride_vs_blk=v_scale_cache.stride(0) if v_scale_cache is not None else 0,
            stride_vs_slot=v_scale_cache.stride(1) if v_scale_cache is not None else 0,
            stride_vs_head=v_scale_cache.stride(2) if v_scale_cache is not None else 0,
            CHUNK_LOOKBACK=chunk_lookback,
            CHUNK_SIZE=chunk_size,
        )
    else:
        # 3D kernel path: from t83/t79, KQM-aware adaptive segment count
        if num_kv_heads <= 4 or kv_quant_mode == 0:
            num_par_softmax_segments = 8
        else:
            num_par_softmax_segments = 16

        softmax_segm_output = torch.empty(
            q.shape[0], num_query_heads, num_par_softmax_segments,
            triton.next_power_of_2(head_size),
            dtype=torch.float32, device=q.device,
        )
        softmax_segm_max = torch.empty(
            q.shape[0], num_query_heads, num_par_softmax_segments,
            dtype=torch.float32, device=q.device,
        )
        softmax_segm_expsum = torch.empty(
            q.shape[0], num_query_heads, num_par_softmax_segments,
            dtype=torch.float32, device=q.device,
        )

        kernel_unified_attention_3d[(
            total_num_q_blocks, num_kv_heads, num_par_softmax_segments
        )](
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
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
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_ALIBI_SQRT=use_alibi_sqrt,
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
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_MM_PREFIX=use_mm_prefix,
            MAX_MM_RANGES=max_mm_ranges,
            mm_prefix_range_ptr=mm_prefix_range,
            KV_QUANT_MODE=kv_quant_mode,
            k_scale_cache_ptr=k_scale_cache,
            v_scale_cache_ptr=v_scale_cache,
            stride_ks_blk=k_scale_cache.stride(0) if k_scale_cache is not None else 0,
            stride_ks_slot=k_scale_cache.stride(1) if k_scale_cache is not None else 0,
            stride_ks_head=k_scale_cache.stride(2) if k_scale_cache is not None else 0,
            stride_vs_blk=v_scale_cache.stride(0) if v_scale_cache is not None else 0,
            stride_vs_slot=v_scale_cache.stride(1) if v_scale_cache is not None else 0,
            stride_vs_head=v_scale_cache.stride(2) if v_scale_cache is not None else 0,
            CHUNK_LOOKBACK=chunk_lookback,
            CHUNK_SIZE=chunk_size,
        )

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )


# ---------------------------------------------------------------------------
# KernelBench Model
# ---------------------------------------------------------------------------
class Model(torch.nn.Module):
    def __init__(self, QH: int, KH: int, D: int, BS: int, NS: int, TQ: int, MKV: int, NB: int,
                 SW: int = 0, SCAP: int = 0, KQM: int = 0):
        super().__init__()
        self.scale = D ** -0.5
        self.sliding_window = SW
        self.softcap = SCAP / 100.0 if SCAP > 0 else 0.0
        self.kv_quant_mode = KQM  # 0=none, 1=fp8_per_tensor, 2=int8_per_token_head, 3=fp8_per_token_head

        query_lens = []
        base = TQ // NS
        remainder = TQ % NS
        for i in range(NS):
            query_lens.append(base + (1 if i < remainder else 0))

        self.max_seqlen_q = max(query_lens)

        cu_query_lens = torch.zeros(NS + 1, dtype=torch.int32)
        for i in range(NS):
            cu_query_lens[i + 1] = cu_query_lens[i] + query_lens[i]

        kv_lens_tensor = torch.full((NS,), MKV, dtype=torch.int32)

        max_num_blocks_per_seq = (MKV + BS - 1) // BS
        torch.manual_seed(42)
        block_tables = torch.randint(0, NB, (NS, max_num_blocks_per_seq), dtype=torch.int32)

        self.register_buffer('cu_query_lens', cu_query_lens)
        self.register_buffer('kv_lens_tensor', kv_lens_tensor)
        self.register_buffer('block_tables', block_tables)

        if KQM >= 2:
            self.register_buffer('k_scale_cache',
                                 torch.ones(NB, BS, KH, dtype=torch.float32))
            self.register_buffer('v_scale_cache',
                                 torch.ones(NB, BS, KH, dtype=torch.float32))

    def forward(self, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor):
        if self.kv_quant_mode in (1, 3):
            key_cache = key_cache.to(torch.float8_e4m3fn)
            value_cache = value_cache.to(torch.float8_e4m3fn)
        elif self.kv_quant_mode == 2:
            key_cache = key_cache.to(torch.int8)
            value_cache = value_cache.to(torch.int8)

        out = torch.empty_like(query)
        window_val = self.sliding_window - 1 if self.sliding_window > 0 else -1
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=self.cu_query_lens,
            max_seqlen_q=self.max_seqlen_q,
            seqused_k=self.kv_lens_tensor,
            block_table=self.block_tables,
            softmax_scale=self.scale,
            window_size=(window_val, window_val),
            softcap=self.softcap,
            kv_quant_mode=self.kv_quant_mode,
            k_scale_cache=getattr(self, 'k_scale_cache', None),
            v_scale_cache=getattr(self, 'v_scale_cache', None),
        )
        return out
