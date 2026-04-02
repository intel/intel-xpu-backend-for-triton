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
    torch_moe_align_block_size,
    get_default_config,
)
from triton_grouped_gemm import invoke_fused_moe_triton_kernel as invoke_tdesc_fused_moe_triton_kernel
from math import prod

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


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(), (f"Requested view {v} with {prod(v)} elements exceeds tensor size {x.numel()}")
    return x.flatten()[:prod(v)].view(*v)


RECIPE_TO_DTYPE = {
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu),
    "fp8block": (torch.float8_e4m3fn, torch.float32),
    "mxfp4": (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu),
}


def ref_prologue(
    x,
    scales,
    flat_expert_indices,
    num_per_tok,
    num_experts,
    recipe,
    ep_rank=0,
    ep_size=1,
):
    expert_start_id = num_experts * ep_rank
    expert_end_id = expert_start_id + num_experts

    idxs = flat_expert_indices.argsort(stable=True)
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))
    if recipe in ["mxfp8", "mxfp4"]:
        scales = scales.view(torch.uint8)

    expand_input = []
    expand_scales = []
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if ((start_idx == end_idx) or (expert_id < expert_start_id) or (expert_id >= expert_end_id)):
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]
        expand_input.append(expert_tokens)
        if scales is not None:
            expert_scales = scales[exp_token_idxs]
            expand_scales.append(expert_scales)

    expert_first_token_offset = torch.Tensor(tokens_per_expert).to(torch.int64)

    expand_input = torch.cat(expand_input, dim=0)
    if recipe == "mxfp4":
        expand_input = expand_input.to(torch.uint8).view(torch.float4_e2m1fn_x2)
    else:
        expand_input = expand_input.to(data_dtype)
    expand_scales = (None if scales is None else torch.cat(expand_scales, dim=0).to(DEVICE))

    return expert_first_token_offset.to(DEVICE), expand_input.to(DEVICE), expand_scales


def ref_grouped_gemm(input_A, input_B, topk_ids, topk):
    num_experts = input_B.shape[0]
    dtype = input_A.dtype

    flat_expert_indices = topk_ids.view(-1)

    ref_expert_offset, ref_expand_input, ref_expand_scales = ref_prologue(input_A, None, flat_expert_indices, topk,
                                                                          num_experts, "bf16")

    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = ref_expert_offset[i]
        if cur_token_num == 0:
            continue
        input = ref_expand_input[pre_token_sum:cur_token_num, :].to(torch.float32)
        weight = input_B[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum = cur_token_num
    ref = torch.cat(ref, dim=0)
    return ref


def get_fused_moe_benchmark(
    providers_filter: Optional[list[str]] = None,
):
    supported_providers = {
        'triton-regular-ptr' : 'triton-regular-ptr',
        'triton-tdesc' : 'triton-tdesc',
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
            plot_name='fused-moe-gemm-performance',
            args={},
        ))
    def benchmark(num_tokens, output_hidden_size, hidden_size, num_experts, topk, dtype, has_bias, provider):
        seed = 7
        torch.manual_seed(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

        M = num_tokens
        N = output_hidden_size
        K = hidden_size
        scores = torch.randn((M, num_experts), device=DEVICE, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)
        hidden_states = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16) / 16
        input_B = torch.randn((num_experts, K, N), dtype=dtype, device=DEVICE)

        # Reference output
        output_ref = ref_grouped_gemm(input_A=hidden_states, input_B=input_B, topk_ids=topk_ids, topk=topk)

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


        if provider.startswith('triton-tdesc'):
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
        elif provider.startswith('triton'):
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
        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        # Triton output
        triton_fn()  # workspace updated in-place
        output = workspace.clone().view(-1, n)
        num_tokens = m * topk
        valid_sorted_token_ids = sorted_token_ids[sorted_token_ids < num_tokens].to(torch.long)
        assert valid_sorted_token_ids.numel() == num_tokens
        output_triton_grouped = output[valid_sorted_token_ids]

        # Correctness check
        torch.testing.assert_close(output_triton_grouped, output_ref, rtol=2e-2, atol=1e-2)

        quantiles = [0.5, 0.0, 1.0]
        n_warmup = 600

        _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
            triton_fn,
            n_warmup=n_warmup,
            n_repeat=10,
            quantiles=quantiles,
        )

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
            # Each (token, expert) route pair performs a K×N matrix-vector product;
            # *2 accounts for the multiply-add operations in the matrix multiplication.
            total_flops = num_routed_tokens * N * K * 2
            return total_flops * 1e-12 / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    print('Running fused MoE benchmark...')
    _benchmark_mm = get_fused_moe_benchmark()
    _benchmark_mm.run(show_plots=False, print_data=True)
