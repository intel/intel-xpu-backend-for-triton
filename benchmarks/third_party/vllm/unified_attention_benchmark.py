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
from typing import Optional, List

import torch
import triton

import triton_kernels_benchmark as benchmark_suite

# Import vLLM attention functions
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.platforms import current_platform


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

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                             (query_len + sliding_window) +
                                             1).bool().logical_not()
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
NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16]
QDTYPES = [None, torch.float8_e4m3fn] if not current_platform.is_rocm() else [
    None, torch.float8_e4m3fnuz
]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
SEQ_LENS = [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
NUM_BLOCKS = [32768, 2048]
ATTENTION_CONFIGS_BF16 = []
for seq_lens, num_heads, head_size, block_size, sliding_window, dtype, soft_cap, num_blocks, q_dtype in product(
    SEQ_LENS, NUM_HEADS, HEAD_SIZES, BLOCK_SIZES, [None, 256], DTYPES, [None, 50.0], NUM_BLOCKS, QDTYPES
):
    if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
        print("Skipping configuration due to incompatible q_dtype and block_size.")
        continue
    ATTENTION_CONFIGS_BF16.append((
        seq_lens,
        num_heads,
        head_size,
        block_size,
        sliding_window,
        dtype,
        soft_cap,
        num_blocks,
        q_dtype,
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
        'pytorch': 'pytorch',
    }

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    ATTENTION_CONFIGS_FP8 = []
    configs = ATTENTION_CONFIGS_FP8 if fp8 else ATTENTION_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['seq_lens', 'num_heads', 'head_size', 'block_size', 'sliding_window', 'dtype', 'soft_cap', 'num_blocks', 'q_dtype'],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--')],
            ylabel=['GB/s', 'TFlops'],
            plot_name=plot_name,
            args={},
        ))
    def benchmark(seq_lens, num_heads, head_size, block_size, sliding_window, dtype, soft_cap, num_blocks, q_dtype, provider):
        # Set default device like in the test
        current_platform.seed_everything(0)  # Use same seed as test
        n_warmup = 100
        quantiles = [0.5, 0.0, 1.0]

        torch.set_default_device("xpu")


        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = ((sliding_window - 1, 0) if sliding_window is not None else
                    (-1, -1))
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens),
                            num_query_heads,
                            head_size,
                            dtype=dtype)
        key_cache = torch.randn(num_blocks,
                                block_size,
                                num_kv_heads,
                                head_size,
                                dtype=dtype)
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor([0] + query_lens,
                                    dtype=torch.int32).cumsum(dim=0,
                                                            dtype=torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0,
                                    num_blocks,
                                    (num_seqs, max_num_blocks_per_seq),
                                    dtype=torch.int32)

        output = torch.empty_like(query)

        maybe_quantized_query = query
        maybe_quantized_key_cache = key_cache
        maybe_quantized_value_cache = value_cache
        q_descale = None
        k_descale = None
        v_descale = None
        if q_dtype is not None:
            # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
            maybe_quantized_query = query.to(q_dtype)
            maybe_quantized_key_cache = key_cache.to(q_dtype)
            maybe_quantized_value_cache = value_cache.to(q_dtype)

            scale_shape = (num_seqs, num_kv_heads)
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
            
        elif provider == 'triton':
            # First run unified_attention exactly like in the test
            def triton_fn():
                unified_attention(
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
            n_bytes = dtype.itemsize if hasattr(dtype, 'itemsize') else 2
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
            for i, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
                # Q@K^T and Attn@V operations
                flops_per_head = 2 * head_size * q_len * kv_len * 2  # 2 matmuls
                total_flops += flops_per_head * num_query_heads
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    # Run unified attention benchmark
    print('Running unified attention benchmark...')
    _benchmark_attention = get_unified_attention_benchmark(fp8=(os.getenv('FP8', '0') == '1'), )
    _benchmark_attention.run(show_plots=False, print_data=True)
