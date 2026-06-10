import torch
import triton
import triton.language as tl

float8_info = torch.finfo(torch.float8_e4m3fn)


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
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


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


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
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

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    seq_mask = seq_offset[None, :] <= query_abs_pos

    if CHUNK_LOOKBACK > -1:
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
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
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            seq_mask |= q_in_range & k_in_range
    return seq_mask


@triton.jit
def apply_alibi_to_score(
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = tl.where(
            relative_pos <= 0,
            -tl.sqrt((-relative_pos).to(tl.float32)),
            0.0,
        )
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset


@triton.jit
def load_qq_bias_tile(
    qq_bias_row_ptrs,
    seq_offset,
    context_len,
    qq_bias_stride_0,
):
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return tl.load(
        qq_bias_row_ptrs + key_rel_pos[None, :],
        mask=is_query_key[None, :],
        other=0.0,
    )


@triton.jit
def softmax_step(S, M, L, USE_EXP2: tl.constexpr = False):
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    if USE_EXP2:
        P = tl.math.exp2(S - m_j[:, None])
        alpha = tl.math.exp2(M - m_j)
    else:
        P = tl.exp(S - m_j[:, None])
        alpha = tl.exp(M - m_j)
    l_j = tl.sum(P, axis=1)
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


@triton.jit
def _load_q_td(
    query_ptr,
    q_block_local_len,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    cur_batch_in_all_start_index,
    q_block_local_idx,
    kv_head_idx,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    q_base = (
        query_ptr
        + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
        + (kv_head_idx * num_queries_per_kv) * query_stride_1
    )
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
        strides=(query_stride_0, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED),
    )
    return q_desc.load([0, 0]).reshape(BLOCK_M, HEAD_SIZE_PADDED)


@triton.jit
def _load_kv_tile_td(
    cache_ptr,
    physical_block_idx_scalar,
    kv_head_idx,
    offset_in_block,
    stride_cache_0: tl.int64,
    stride_cache_1: tl.int64,
    stride_cache_2: tl.int64,
    stride_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    base = (
        cache_ptr
        + physical_block_idx_scalar * stride_cache_0
        + kv_head_idx * stride_cache_2
    )
    desc = tl.make_tensor_descriptor(
        base=base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_cache_1, stride_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
    )
    return desc.load([offset_in_block, 0])


@triton.jit
def _store_output_td(
    base_ptr,
    acc,
    q_block_local_len,
    stride_token: tl.int64,
    stride_head: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    acc = acc.to(base_ptr.dtype.element_ty)
    output_desc = tl.make_tensor_descriptor(
        base=base_ptr,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(stride_token, stride_head, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )
    output_desc.store(
        [0, 0, 0],
        acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )


# ---------------------------------------------------------------------------
# Decode kernel — t2's autotune (warps 4/8, no nqpkv key) for non-quant decode
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['IS_3D', 'BLOCK_M', 'HEAD_SIZE', 'TILE_SIZE', 'KV_QUANT_MODE', 'SLIDING_WINDOW'],
)
@triton.jit
def kernel_attention_decode(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
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
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_3D: tl.constexpr,
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    stride_ks_blk: tl.int64 = None,
    stride_ks_slot: tl.int64 = None,
    stride_ks_head: tl.int64 = None,
    stride_vs_blk: tl.int64 = None,
    stride_vs_slot: tl.int64 = None,
    stride_vs_head: tl.int64 = None,
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_TD: tl.constexpr = True,
    USE_TD_QO: tl.constexpr = True,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        if USE_TD:
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
            if USE_PER_TOKEN_HEAD_SCALES:
                physical_block_idx = tl.load(
                    block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
                ).to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if USE_PER_TOKEN_HEAD_SCALES:
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    if IS_3D:
        if USE_TD_QO:
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            output_offset = (
                query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1
                + offs_d[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )


# ---------------------------------------------------------------------------
# Compute kernel — t8's autotune (warps 2-16, nqpkv key) for prefill + quant decode
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['IS_3D', 'BLOCK_M', 'HEAD_SIZE', 'TILE_SIZE', 'KV_QUANT_MODE', 'num_queries_per_kv', 'IS_DECODE'],
)
@triton.jit
def kernel_attention_compute(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
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
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_3D: tl.constexpr,
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    stride_ks_blk: tl.int64 = None,
    stride_ks_slot: tl.int64 = None,
    stride_ks_head: tl.int64 = None,
    stride_vs_blk: tl.int64 = None,
    stride_vs_slot: tl.int64 = None,
    stride_vs_head: tl.int64 = None,
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_TD: tl.constexpr = True,
    USE_TD_QO: tl.constexpr = True,
    IS_DECODE: tl.constexpr = False,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2
    USE_EXP2_ATTN: tl.constexpr = (not IS_DECODE) and (not USE_SOFTCAP) and (not USE_ALIBI_SLOPES) and (not USE_QQ_BIAS)

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    if USE_EXP2_ATTN:
        attn_scale = scale * 1.4426950408889634
    else:
        attn_scale = scale

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        if USE_PER_TOKEN_HEAD_SCALES or not USE_TD:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

        if USE_TD:
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            S += tl.dot(Q, K) * (attn_scale * k_token_head_scales[None, :])
        else:
            S += attn_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L, USE_EXP2=USE_EXP2_ATTN)
        acc = acc * alpha[:, None]

        if USE_PER_TOKEN_HEAD_SCALES:
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    if IS_3D:
        if USE_TD_QO:
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            output_offset = (
                query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1
                + offs_d[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )


# ---------------------------------------------------------------------------
# reduce_segments — combines 3D partial results
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=['NUM_SEGMENTS_PER_SEQ', 'HEAD_SIZE'],
)
@triton.jit
def reduce_segments(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    out_scale_inv,
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
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
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
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
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
# Python launcher with dual-kernel dispatch
# ---------------------------------------------------------------------------


def unified_attention(
    q,
    k,
    v,
    out,
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
    kv_quant_mode=0,
    k_scale_cache=None,
    v_scale_cache=None,
    chunk_lookback=-1,
):
    assert softcap >= 0.0

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1]

    use_per_token_head_scales = kv_quant_mode in (2, 3)
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None

    if not isinstance(k_descale, torch.Tensor):
        k_descale = torch.tensor(k_descale, dtype=torch.float32, device=q.device)
    if not isinstance(v_descale, torch.Tensor):
        v_descale = torch.tensor(v_descale, dtype=torch.float32, device=q.device)

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}")

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    if q.element_size() == 1:
        TILE_SIZE_PREFILL = TILE_SIZE_DECODE = 32
    else:
        TILE_SIZE_PREFILL = TILE_SIZE_DECODE = 16
    assert TILE_SIZE_PREFILL <= block_size
    assert block_size % TILE_SIZE_PREFILL == 0

    _is_pow2_nq = (num_queries_per_kv & (num_queries_per_kv - 1)) == 0
    _is_pow2_hs = head_size == head_size_padded
    use_td_qo = _is_pow2_nq and _is_pow2_hs

    if use_td_qo:
        assert q.stride(1) == head_size
        assert out.stride(1) == head_size

    seq_threshold_3D = 32
    use_3d = not (max_seqlen_q > 1 or num_seqs > seq_threshold_3D
                  or 0 < sliding_window_val <= 1024)

    if use_3d and kv_quant_mode == 0 and num_queries_per_kv < 16:
        use_3d = False

    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = 0
        vs_blk = vs_slot = vs_head = 0
        k_scale_ptr = None
        v_scale_ptr = None

    use_decode_kernel = (max_seqlen_q == 1 and kv_quant_mode == 0
                         and num_seqs <= seq_threshold_3D
                         and (num_queries_per_kv < 8 or num_queries_per_kv >= 16))

    if use_3d:
        num_par_softmax_segments = 4 if kv_quant_mode == 0 else 16
        softmax_segm_output = torch.empty(
            q.shape[0], num_query_heads, num_par_softmax_segments, head_size_padded,
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
    else:
        num_par_softmax_segments = 1
        softmax_segm_output = None
        softmax_segm_max = None
        softmax_segm_expsum = None

    segm_output_ptr = softmax_segm_output if use_3d else None
    segm_max_ptr = softmax_segm_max if use_3d else None
    segm_expsum_ptr = softmax_segm_expsum if use_3d else None

    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    kernel_args = dict(
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
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
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
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        KV_QUANT_MODE=kv_quant_mode,
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        USE_TD=True,
        USE_TD_QO=use_td_qo,
    )

    if use_decode_kernel:
        kernel_attention_decode[grid](**kernel_args)
    else:
        kernel_attention_compute[grid](**kernel_args, IS_DECODE=(max_seqlen_q == 1))

    if use_3d:
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
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )


class Model(torch.nn.Module):
    def __init__(self, QH: int, KH: int, D: int, BS: int, NS: int, TQ: int, MKV: int, NB: int,
                 SW: int = 0, SCAP: int = 0, KQM: int = 0):
        super().__init__()
        self.scale = D ** -0.5
        self.sliding_window = SW
        self.softcap = SCAP / 100.0 if SCAP > 0 else 0.0
        self.kv_quant_mode = KQM

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
