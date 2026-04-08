# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused MoE benchmark
===================

This benchmark measures the performance of the fused MoE grouped GEMM kernel
and follows the framework from gemm_benchmark.py to compare performance of
different implementations using vLLM kernels.

"""
import os
from typing import Optional

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
from triton_grouped_gemm import (
    _resize_cache,
    torch_moe_align_block_size,
    get_default_config,
)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

DEVICE_TOTAL_MEMORY_BYTES = benchmark_suite.get_total_gpu_memory_bytes()


def is_enough_memory(x_val, safety_factor=0.80):
    M, N, K, num_experts, topk, dtype, has_bias = x_val

    n_bytes = 2  # bfloat16
    # A: input activations (M, K)
    a_mem = M * K * n_bytes
    # B: weight matrices (num_experts, K, N)
    b_mem = num_experts * K * N * n_bytes
    # C: output workspace (M, topk, N)
    out_mem = M * topk * N * n_bytes

    required_memory = a_mem + b_mem + out_mem
    return required_memory < DEVICE_TOTAL_MEMORY_BYTES * safety_factor


def filter_by_memory(configs):
    result = []
    for x_val in configs:
        if is_enough_memory(x_val):
            result.append(x_val)
        else:
            print(f"'{x_val}' combination skipped, OOM expected")
    return result


# Benchmark configurations: (M, N, K, num_experts, topk, dtype, has_bias)
# Each tuple represents one fused MoE GEMM call with the given token count M.
MM_CONFIGS = [
    # Qwen3-30B-A3B-Instruct w13 with MNK factors (80, 768 * 2 // 4, 2048), num_experts=128, topk=8
    [80, 768 * 2 // 4, 2048, 128, 8, torch.bfloat16, False],
    # Qwen3-30B-A3B-Instruct w2 with MNK factors (80, 2048, 768 * 2 // 2 // 4), num_experts=128, topk=8
    [80, 2048, 768 * 2 // 2 // 4, 128, 8, torch.bfloat16, False],
    # Qwen3-30B-A3B-Instruct w13 with MNK factors (8192, 768 * 2 // 4, 2048), num_experts=128, topk=8
    [8192, 768 * 2 // 4, 2048, 128, 8, torch.bfloat16, False],
    # Qwen3-30B-A3B-Instruct w2 with MNK factors (8192, 2048, 768 * 2 // 2 // 4), num_experts=128, topk=8
    [8192, 2048, 768 * 2 // 2 // 4, 128, 8, torch.bfloat16, False],
    # Llama-4-scout with MNK factors (30, 8192 * 2, 5120), num_experts=16, topk=1
    [30, 8192 * 2, 5120, 16, 1, torch.bfloat16, False],
    # Llama-4-scout with MNK factors (8192, 8192 * 2, 5120), num_experts=16, topk=1
    [8192, 8192 * 2, 5120, 16, 1, torch.bfloat16, False],
]

MM_CONFIGS = filter_by_memory(MM_CONFIGS)

# To debug if the benchmark runs at all, without waiting for all configurations to run
if os.getenv('DEBUG_BENCH', '0') == '1':
    MM_CONFIGS = MM_CONFIGS[:1]


def get_fused_moe_benchmark(
    providers_filter: Optional[list[str]] = None,
    is_td_patched=False,
):
    supported_providers = {
        'triton-regular-ptr' + ('-td' if is_td_patched else ''): 'triton' + ('-td' if is_td_patched else ''),
    }

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['num_tokens', 'output_hidden_size', 'hidden_size', 'num_experts', 'topk', 'dtype', 'has_bias'],
            x_vals=MM_CONFIGS,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('red', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name='fused-moe-gemm-performance' + ('-td' if is_td_patched else ''),
            args={},
        ))
    def benchmark(num_tokens, output_hidden_size, K, num_experts, topk, dtype, has_bias, provider):
        seed = 7
        torch.manual_seed(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

        M = num_tokens
        N = output_hidden_size
        scores = torch.randn((M, num_experts), device=DEVICE, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
        hidden_states = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16) / 16
        input_B = torch.randn((num_experts, K, N), dtype=dtype, device=DEVICE)
        input_A = hidden_states
        m = input_A.shape[0]
        k = input_A.shape[1]
        n = input_B.shape[-1]
        input_B = input_B.transpose(1, 2).contiguous()
        ws_shape = (m, topk, max(n, k))
        workspace = _resize_cache(
            torch.empty(
                ws_shape[0] * ws_shape[1] * ws_shape[2],
                dtype=input_A.dtype,
                device=input_A.device,
            ),
            (m, topk, n),
        )
        config = get_default_config(m, num_experts, n, k, topk)
        sorted_token_ids, expert_ids, num_tokens_post_padded = torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=config["BLOCK_SIZE_M"],
            num_experts=num_experts,
            pad_sorted_ids=True,
        )

        # Number of unique experts actually used in this batch
        num_activated_experts = topk_ids.unique().numel()
        # Total number of (token, expert) route pairs
        num_routed_tokens = m * topk

        def triton_fn():
            invoke_fused_moe_triton_kernel(
                input_A,
                input_B,
                workspace,
                None,  # input scales
                None,  # weight scales
                None,  # topk_weights
                sorted_token_ids,  # sorted_token_ids
                expert_ids,  # expert_ids
                num_tokens_post_padded,  # num_tokens_post_padded
                False,  # mul_routed_weights
                topk,
                config,
                tl.bfloat16,
                False,
                False,
                False,
                False,
                False,
                None,
                None,
            )
            return workspace

        quantiles = [0.5, 0.0, 1.0]
        n_warmup = 600

        if provider.startswith('triton'):
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )
        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        def gbps(ms):
            n_bytes = 2  # bfloat16
            total_bytes = (
                # B matrix: only load weights for activated experts
                num_activated_experts * K * N * n_bytes +
                # A matrix: input activations
                m * K * n_bytes +
                # C matrix: output (one entry per routed token)
                num_routed_tokens * N * n_bytes)
            return total_bytes * 1e-9 / (ms * 1e-3)

        def tflops(ms):
            # Each (token, expert) route pair performs a K×N matrix-vector product
            total_flops = num_routed_tokens * N * K * 2
            return total_flops * 1e-12 / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    print('Running fused MoE benchmark...')
    _benchmark_mm = get_fused_moe_benchmark(is_td_patched=is_td_patched)
    _benchmark_mm.run(show_plots=False, print_data=True)
