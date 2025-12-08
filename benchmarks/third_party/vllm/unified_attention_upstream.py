from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


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
    # import pdb; pdb.set_trace()
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q = q * scale

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
        attn = torch.softmax(attn, dim=-1, ).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


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
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
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
    # WTF: Switching to tl.int64 fixes this issue
    stride_k_cache_3: tl.constexpr,  # int
    # stride_k_cache_3: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = 0,
    FP8_MAX: tl.constexpr = 1,
    debug_ptr = None,
):
    # import pdb; pdb.set_trace()
    kv_head_idx = 0
    seq_idx = 0
    q_block_local_idx = 0
    cur_batch_in_all_start_index = 0

    cur_batch_query_len = 1

    offs_m = tl.arange(0, BLOCK_M).to(tl.int32)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED).to(tl.int32)
    offs_t = tl.arange(0, TILE_SIZE).to(tl.int32)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.bfloat16)

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # WTF: setting value to 2 fixes the issue
    # seq_len = 2

    # WTF: stopping everything non-2 fixes the issue
    # if seq_len != 2:
    #     return

    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    # num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
    # tl.device_print("num_tiles:", num_tiles)
    # WTF: stopping everything non-1 fixes the issue
    # if num_tiles > 1:
    #     return

    # iterate through tiles
    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset +
                                     seq_offset // BLOCK_SIZE).to(tl.int32)

        k_offset = (physical_block_idx[None, :] * stride_k_cache_0 +
                    kv_head_idx * stride_k_cache_2 +
                    offs_d[:, None].to(tl.int32) * stride_k_cache_3 +
                    # offs_d[:, None] * my_const +
                    (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1)

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset,
                         mask=dim_mask[:, None] & tile_mask[None, :],
                         other=0.0)
        K = K_load
        V = tl.zeros(shape=(TILE_SIZE, HEAD_SIZE_PADDED), dtype=tl.bfloat16)
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                     S, float("-inf"))

        # WTF: if we print here, the issue disappears
        # tl.device_print("S:", S)

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # is_gt_neg_inf = m_j > float("-inf")
        # m_j_sanitized = tl.where(is_gt_neg_inf, m_j, 0.0)

        # WTF: if we print here, the issue disappears
        # tl.device_print("is_nan_input:", m_j)

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where((m_j > float("-inf")) & (m_j != float("nan")), m_j, 0.0)


        # P : (BLOCK_M, TILE_SIZE)
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
        # DEBUG changing V dtype to float32 fixes the issue
        # acc += tl.dot(P, V.to(tl.float32))

    # epilogue
    acc = acc / L[:, None]

    output_offset = (query_offset_0[:, None] * output_stride_0 +
                     query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def unified_attention(
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

    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

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
    head_size = q.shape[2]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(
        num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    print("BLOCK_M:", BLOCK_M, " BLOCK_Q:", BLOCK_Q)

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

    # Assigning default tile sizes for prefill and decode.
    # Note: each tile size must be at least 32 for "fp8" (q.element_size() == 1)
    # and at least 16 for all other data types.
    TILE_SIZE_PREFILL = 16
    print("Total", total_num_q_blocks, num_kv_heads)

    debug_tensor = torch.zeros(128, dtype=torch.float32, device="xpu")
    # if batch contains a prefill
    if True:
        kernel_unified_attention_2d[(
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
            TILE_SIZE=TILE_SIZE_PREFILL,
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
            debug_ptr=debug_tensor,
        )
        print("Debug flag:", debug_tensor[0].item())

def main():
    q_heads = 1
    k_heads = 1
    head_size = 64
    dtype = torch.bfloat16
    qdtype = None
    seq_lens = [(1, 1)]
    block_size = 16
    sliding_window = None
    soft_cap = None
    num_blocks = 1
    # Set default device like in the test
    print("Config: ", q_heads, k_heads, head_size, dtype, qdtype, seq_lens, sliding_window, soft_cap, num_blocks,)

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
    query[:] = 0
    key_cache = torch.randn(num_blocks, block_size, k_heads, head_size, dtype=dtype)
    key_cache[:] = 0 # + key_cache * 1e-4
    value_cache = torch.randn_like(key_cache)
    value_cache[:] = 0
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

    output = torch.empty_like(query)
    output[:] = -1

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None

    def torch_fn():
        return ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens_tensor,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

    def triton_fn():
        unified_attention(
            q=maybe_quantized_query,
            k=maybe_quantized_key_cache,
            v=maybe_quantized_value_cache,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens_tensor,
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
    triton_out = triton_fn()
    torch_out = torch_fn()

    atol, rtol = 1.5e-2, 1e-2
    if dtype != torch.bfloat16:
        atol, rtol = 1.5e-1, 1.5e-1
    diff = (triton_out - torch_out).abs().max()
    print("Max diff:", diff)
    print("Min, max: ", triton_out.min().item(), triton_out.max().item())
        # try:
    print("Checking correctness...")
    if not torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol):
        print("Triton output:\n", triton_out)
        print("Torch output:\n", torch_out)
        raise ValueError("Outputs are not close!")
    print("Correctness check passed!")  


if __name__ == '__main__':
    main()
