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
import random
from math import prod
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite

from tests.kernels.moe.utils import make_quantized_test_activations, make_test_weight
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, get_default_config
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm as sycl_tla_grouped_gemm

DEVICE = triton.runtime.driver.active.get_active_torch_device()

DEVICE_TOTAL_MEMORY_BYTES = benchmark_suite.get_total_gpu_memory_bytes()


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = ((original_count + block_size - 1) // block_size) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded, ),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos:current_pos + num_expert_tokens] = expert_tokens

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size
            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block:current_block + expert_blocks_needed] = expert_id_new

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor([total_padded_tokens], dtype=torch.int32, device=topk_ids.device)

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def is_enough_memory(x_val, safety_factor=0.80):
    M, N, K, num_experts, topk, dtype, has_bias = x_val

    fp8 = dtype == torch.float8_e4m3fn

    # A and B bf16 originals (always allocated, freed later in fp8 case)
    a_mem = M * K * 2
    b_mem = num_experts * K * N * 2

    if fp8:
        # fp8 copies + scales
        a_mem += M * K + 4
        b_mem += num_experts * K * N + num_experts * 4

    # C: output workspace (M, topk, N)
    out_mem = M * topk * N * 2

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
MM_CONFIGS_BF16 = [
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

MM_CONFIGS_FP8 = [[M, N, K, num_experts, topk, torch.float8_e4m3fn, has_bias]
                  for M, N, K, num_experts, topk, _, has_bias in MM_CONFIGS_BF16]

MM_CONFIGS_BF16 = filter_by_memory(MM_CONFIGS_BF16)
MM_CONFIGS_FP8 = filter_by_memory(MM_CONFIGS_FP8)

# To debug if the benchmark runs at all, without waiting for all configurations to run
if os.getenv('DEBUG_BENCH', '0') == '1':
    MM_CONFIGS_BF16 = MM_CONFIGS_BF16[:1]
    MM_CONFIGS_FP8 = MM_CONFIGS_FP8[:1]


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(), f"Requested view {v} with {prod(v)} elements exceeds tensor size {x.numel()}"
    return x.flatten()[:prod(v)].view(*v)


def _normalize_fp8_scale(scale: torch.Tensor | None) -> torch.Tensor | None:
    if scale is None:
        return None
    return scale.reshape(1) if scale.numel() == 1 else scale


def _dequantize_fp8(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    dequantized = tensor.to(torch.float32)
    if scale is not None:
        dequantized = dequantized * scale.to(torch.float32)
    return dequantized


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


def get_fused_moe_benchmark(providers_filter: Optional[list[str]] = None, is_fp8=False, is_td_patched=False):
    supported_providers = {
        'triton' + ('-td' if is_td_patched else ''): 'triton' + ('-td' if is_td_patched else ''),
        'sycl-tla': 'sycl-tla',
    }

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    configs = MM_CONFIGS_FP8 if is_fp8 else MM_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['num_tokens', 'output_hidden_size', 'hidden_size', 'num_experts', 'topk', 'dtype', 'has_bias'],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('red', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name='fused-moe-gemm-performance' + ('-fp8' if is_fp8 else '') + ('-td' if is_td_patched else ''),
            args={},
        ))
    def benchmark(num_tokens, output_hidden_size, hidden_size, num_experts, topk, dtype, has_bias, provider):
        seed = 7
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        M = num_tokens
        N = output_hidden_size
        K = hidden_size
        use_fp8 = dtype == torch.float8_e4m3fn
        output_dtype = torch.bfloat16
        scores = torch.randn((M, num_experts), device=DEVICE, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

        if use_fp8:
            hidden_states, input_A, input_scales = make_quantized_test_activations(
                1,
                M,
                K,
                in_dtype=output_dtype,
                quant_dtype=dtype,
            )
            hidden_states = hidden_states.squeeze(0)
            input_A = input_A.squeeze(0)
            input_scales = _normalize_fp8_scale(input_scales)

            input_B_orig, input_B, weight_scales, _ = make_test_weight(
                num_experts,
                N,
                K,
                in_dtype=output_dtype,
                quant_dtype=dtype,
            )
            weight_scales = _normalize_fp8_scale(weight_scales)
            input_B_ref = _dequantize_fp8(input_B, weight_scales).transpose(1, 2).contiguous()
            hidden_states_ref = _dequantize_fp8(input_A, input_scales)
        else:
            hidden_states = torch.randn((M, K), device=DEVICE, dtype=output_dtype) / 16
            input_A = hidden_states
            input_scales = None
            input_B_ref = torch.randn((num_experts, K, N), dtype=output_dtype, device=DEVICE)
            input_B = input_B_ref.transpose(1, 2).contiguous()
            weight_scales = None

        # Reference output
        output_ref = ref_grouped_gemm(
            input_A=(hidden_states_ref if use_fp8 else hidden_states).to(output_dtype),
            input_B=input_B_ref.to(output_dtype),
            topk_ids=topk_ids,
            topk=topk,
        )
        if use_fp8:
            # Free unused bf16 originals in the fp8 case.
            del hidden_states, input_B_orig

        m = M
        n = N
        k = K
        ws_shape = (m, topk, max(n, k))
        workspace = _resize_cache(
            torch.empty(
                ws_shape[0] * ws_shape[1] * ws_shape[2],
                dtype=output_dtype,
                device=input_A.device,
            ),
            (m, topk, n),
        )
        config = get_default_config(m, num_experts, n, k, topk, "fp8_w8a8" if use_fp8 else "bf16")
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

        quantiles = [0.5, 0.0, 1.0]
        n_warmup = 600

        if provider.startswith('triton'):

            def triton_fn():
                invoke_fused_moe_triton_kernel(
                    input_A,
                    input_B,
                    workspace,
                    input_scales,
                    weight_scales,
                    None,  # topk_weights
                    sorted_token_ids,  # sorted_token_ids
                    expert_ids,  # expert_ids
                    num_tokens_post_padded,  # num_tokens_post_padded
                    False,  # mul_routed_weights
                    topk,
                    config,
                    tl.bfloat16,
                    use_fp8,
                    False,
                    False,
                    False,
                    False,
                    None,
                    None,
                )
                return workspace

            triton_fn()  # workspace updated in-place
            output = workspace.clone().view(-1, n)
            num_output_tokens = m * topk
            valid_sorted_token_ids = sorted_token_ids[sorted_token_ids < num_output_tokens].to(torch.long)
            assert valid_sorted_token_ids.numel() == num_output_tokens
            output_triton_grouped = output[valid_sorted_token_ids]

            # Correctness check
            torch.testing.assert_close(
                output_triton_grouped,
                output_ref,
                rtol=6e-2 if use_fp8 else 2e-2,
                atol=6e-2 if use_fp8 else 1e-2,
            )

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'sycl-tla':
            # Gather tokens by expert inside the timed region to match Triton's in-kernel
            # gather; the permutation depends only on routing, so compute it once.
            flat_expert_indices = topk_ids.view(-1)
            gather_idx = flat_expert_indices.argsort(stable=True) // topk
            rows_per_expert = flat_expert_indices.bincount(minlength=num_experts).to(torch.int32).tolist()
            input_B_grouped = input_B.transpose(1, 2).contiguous()
            output_sycl = torch.empty((gather_idx.shape[0], n), dtype=input_A.dtype, device=DEVICE)

            def sycl_tla_fn():
                input_A_grouped = input_A.index_select(0, gather_idx)
                sycl_tla_grouped_gemm(input_A_grouped, input_B_grouped, None, output_sycl, rows_per_expert, n, k,
                                      num_experts)
                return output_sycl

            sycl_tla_fn()
            torch.testing.assert_close(output_sycl, output_ref, rtol=2e-2, atol=1e-2)

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                sycl_tla_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        def gbps(ms):
            n_bytes = 1 if use_fp8 else 2
            total_bytes = (
                # B matrix: only load weights for activated experts
                num_activated_experts * K * N * n_bytes +
                # A matrix: each token is read once per expert assignment (topk times total)
                num_routed_tokens * K * n_bytes +
                # C matrix: output (one entry per routed token)
                num_routed_tokens * N * 2)
            return total_bytes * 1e-9 / (ms * 1e-3)

        def tflops(ms):
            # Each (token, expert) route pair performs a K×N matrix-vector product;
            # *2 accounts for the multiply-add operations in the matrix multiplication.
            total_flops = num_routed_tokens * N * K * 2
            return total_flops * 1e-12 / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


def get_benchmark(providers_filter: Optional[list[str]] = None, is_fp8=False, is_td_patched=None):
    if is_td_patched is None:
        is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    return get_fused_moe_benchmark(
        providers_filter=providers_filter,
        is_fp8=is_fp8,
        is_td_patched=is_td_patched,
    )


if __name__ == '__main__':
    is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    print('Running fused MoE benchmark...')
    _benchmark_mm = get_fused_moe_benchmark(is_fp8=(os.getenv('FP8', '0') == '1'), is_td_patched=is_td_patched)
    _benchmark_mm.run(show_plots=False, print_data=True)
