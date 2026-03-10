# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Based on this commit:
# b5545d9d5cab2625ac04a19f552631a2034c8f47
# https://github.com/vllm-project/vllm/blob/b5545d9d5cab2625ac04a19f552631a2034c8f47/vllm/attention/ops/triton_unified_attention.py
"""
Unified Attention benchmark
==========================

This benchmark is based on the test_triton_unified_attention.py tests
"""
import os
from itertools import product
from typing import Optional

import torch

import triton_kernels_benchmark as benchmark_suite

# This supports both current upstream and pinned version
try:
    from vllm.attention.ops.triton_unified_attention import unified_attention
except ImportError:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
except ImportError as e:
    raise ImportError(
        "Could not import unified_attention from vLLM. Please ensure vLLM is installed and accessible.") from e
from vllm.platforms import current_platform

float8_info = torch.finfo(current_platform.fp8_dtype())

TOTAL_MEMORY_BYTES = benchmark_suite.get_total_gpu_memory_bytes()

IS_FP8 = (os.getenv('FP8', '0') == '1')


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
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
            sliding_window_mask = (torch.triu(empty_mask,
                                              diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not())
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _dtype_size(dtype):
    """Return element size in bytes for a torch dtype."""
    if dtype is None:
        return 0
    if hasattr(dtype, 'itemsize'):
        return dtype.itemsize
    # Fallback for dtype objects without itemsize
    return torch.tensor([], dtype=dtype).element_size()


def is_enough_memory(x_val, safety_factor=0.80):
    """Check whether all tensors created by the benchmark fit in GPU memory.

    Mirrors every allocation inside ``benchmark()`` so the estimate is
    precise.  Returns *True* when the total fits within
    ``safety_factor`` of the device's total memory.
    """
    q_heads, k_heads, head_size, qdtype, seq_lens, sliding_window, soft_cap, num_blocks, block_size = x_val
    dtype = torch.bfloat16

    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_query_tokens = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    d_bytes = _dtype_size(dtype)  # e.g. 2 for bfloat16

    # --- tensors allocated in benchmark() ---
    # query:       (total_query_tokens, q_heads, head_size)  dtype
    query_mem = total_query_tokens * q_heads * head_size * d_bytes
    # key_cache:   (num_blocks, block_size, k_heads, head_size) dtype
    kv_cache_mem = 2 * num_blocks * block_size * k_heads * head_size * d_bytes
    # output:      same shape/dtype as query  (empty_like)
    output_mem = query_mem
    # cu_query_lens: (num_seqs + 1,) int32
    cu_mem = (num_seqs + 1) * 4
    # kv_lens_tensor: (num_seqs,) int32
    kvl_mem = num_seqs * 4
    # block_tables: (num_seqs, max_num_blocks_per_seq) int32
    bt_mem = num_seqs * max_num_blocks_per_seq * 4

    # quantised duplicates (when qdtype is set the originals are *kept*)
    q_quant_mem = 0
    if qdtype is not None:
        qd_bytes = _dtype_size(qdtype)
        # maybe_quantized_query, _key_cache, _value_cache
        q_quant_mem += total_query_tokens * q_heads * head_size * qd_bytes
        q_quant_mem += 2 * num_blocks * block_size * k_heads * head_size * qd_bytes
        # k_descale, v_descale: (num_seqs, k_heads) float32
        q_quant_mem += 2 * num_seqs * k_heads * 4

    total_memory = (query_mem + kv_cache_mem + output_mem + cu_mem + kvl_mem + bt_mem + q_quant_mem)

    threshold = TOTAL_MEMORY_BYTES * safety_factor
    enough = total_memory < threshold

    return enough


# There is an error right now, for FP8 matmul with block size 16 is not supported
# Input shapes should have M >= 1, N >= 16 and K >= 32
MMAP_BLOCK_SIZES = [64] if IS_FP8 else [16, 64]
NUM_BLOCKS = [32768, 2048]
SEQ_LENS = [
    # One 4k input prefill
    [(4096, 4096)],
    # Chunked prefill: 4 batches
    [(512, 512), (512, 512), (512, 512), (512, 512)],
    # End of chunked prefill and some decoding
    [(1, 1328), (5, 178), (129, 463)],
    # Pure decoding, 8 batches
    [(1, k) for k in [1513, 4100, 530, 123, 4803, 434, 3015, 34]]
]
# Models: (q_heads, k_heads, head_size, qdtype, sliding_window, soft_cap)
# sliding_window: None = full attention, int = sliding window size.
# soft_cap: None = disabled, float = soft_cap value.
# Models that use both attention types appear twice (one entry each).
MODELS_BF16 = [
    # llama3.1-8B - full attention
    (32, 8, 128, None, None, None),
    # llama3.1-8B - just to test soft caps kernel path, real model doesn't use it, it's relevant for gemma2
    (32, 8, 128, None, None, 50.0),
    # llama3.3-70B - full attention
    (64, 8, 128, None, None, None),
    # llama4 Scout - sliding window attention (window size 8192)
    (64, 8, 128, None, 8192, None),
    # Qwen2.5-235B - full attention
    (64, 4, 128, None, None, None),
    # Qwen2.5-235B - sliding window attention (window size 256)
    (64, 4, 128, None, 256, None),
    # gpt-oss-120b - full attention
    # (64, 8, 64, None, None, None),
    # gpt-oss-120b - sliding window attention (window size 128)
    # (64, 8, 64, None, 128, None),
]

MODELS_FP8 = [
    # llama4 scout - full attention
    (64, 8, 128, torch.float8_e4m3fn, None, None),
    # llama4 scout - sliding window attention (window size 8192)
    (64, 8, 128, torch.float8_e4m3fn, 8192, None),
]


def _build_attention_configs(model_configs):
    configs = []
    for model_config in model_configs:
        *base_config, sliding_window, soft_cap = model_config
        for seq_lens, num_blocks, block_size in product(SEQ_LENS, NUM_BLOCKS, MMAP_BLOCK_SIZES):
            x_val = (*base_config, seq_lens, sliding_window, soft_cap, num_blocks, block_size)
            qdtype = x_val[3]
            if qdtype is not None and qdtype.itemsize < 2 and x_val[-1] < 32:
                print("Skipping configuration due to incompatible q_dtype and block_size.")
                continue
            if is_enough_memory(x_val=x_val):
                configs.append(x_val)
            else:
                print(f"Skipping configuration due to memory constraints: {x_val}")
    return configs


ATTENTION_CONFIGS_BF16 = _build_attention_configs(MODELS_BF16)
ATTENTION_CONFIGS_FP8 = _build_attention_configs(MODELS_FP8)

# To debug if the benchmark runs at all, without waiting for all configurations to run
if os.getenv('DEBUG_BENCH', '0') == '1':
    ATTENTION_CONFIGS_BF16 = ATTENTION_CONFIGS_BF16[:1]
    ATTENTION_CONFIGS_FP8 = ATTENTION_CONFIGS_FP8[:1]


def get_unified_attention_benchmark(
    providers_filter: Optional[list[str]] = None,
    is_fp8=False,
    is_td_patched=False,
):
    supported_providers = {
        'triton' + ('-td' if is_td_patched else ''): 'triton' + ('-td' if is_td_patched else ''),
        'pytorch': 'pytorch',
    }
    if os.getenv("TRITON_INTERPRET", "0") == "1" and is_td_patched:
        # Skip triton providers if interpreter is used because if fails
        del supported_providers['triton']

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    configs = ATTENTION_CONFIGS_FP8 if is_fp8 else ATTENTION_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=[
                'q_heads', 'k_heads', 'head_size', 'qdtype', 'seq_lens', 'sliding_window', 'soft_cap', 'num_blocks',
                'block_size'
            ],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('orange', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name='unified-attention-performance' + ('-td' if is_td_patched else ''),
            args={},
        ))
    def benchmark(q_heads, k_heads, head_size, qdtype, seq_lens, sliding_window, soft_cap, num_blocks, block_size,
                  provider):
        print("Config:", q_heads, k_heads, head_size, qdtype, seq_lens, sliding_window, soft_cap, num_blocks,
              block_size, provider)
        dtype = torch.bfloat16
        torch.manual_seed(20)
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
        kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

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
                kv_lens=kv_lens_tensor,
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

            atol, rtol = 2.5e-2, 1e-2
            if qdtype is not None:
                atol, rtol = 3 / 8 + 1e-6, 1.5e-1
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
            # Input Q/K/V use quantized dtype if available, output is always bf16
            in_bytes = qdtype.itemsize if qdtype is not None else dtype.itemsize
            out_bytes = dtype.itemsize  # output is always bf16
            total_query_tokens = sum(query_lens)
            # Memory: Query (in) + Key cache (in) + Value cache (in) + Output (out)
            total_bytes = (
                total_query_tokens * q_heads * head_size * in_bytes +  # Query
                sum([min(l, sliding_window) if sliding_window is not None else l
                     for l in kv_lens]) * k_heads * head_size * in_bytes * 2 +  # KV cache accessed
                total_query_tokens * q_heads * head_size * out_bytes  # Output
            )
            return total_bytes * (1e-9) / (ms * 1e-3)

        def tflops(ms):
            # Attention FLOPs: Q@K (2*d*seq_len*kv_len) + Softmax (~seq_len*kv_len) + Attn@V (2*d*seq_len*kv_len)
            total_flops = 0
            for i, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
                if sliding_window is not None:
                    kv_len = min(kv_len, sliding_window)
                # Q@K^T and Attn@V operations
                flops_per_head = 2 * head_size * q_len * kv_len * 2  # 2 matmuls
                total_flops += flops_per_head * q_heads
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    _benchmark_attention = get_unified_attention_benchmark(is_fp8=IS_FP8, is_td_patched=is_td_patched)
    _benchmark_attention.run(show_plots=False, print_data=True)
