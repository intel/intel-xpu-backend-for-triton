# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batched MoE benchmark
=====================

This benchmark is based on the test_batched_moe.py tests and follows
the framework from gemm_benchmark.py to compare performance of different
batched MoE implementations using vLLM kernels.

"""
import os
from typing import Optional

import torch
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite

from vllm.model_executor.layers.fused_moe.fused_batched_moe import invoke_moe_batched_triton_kernel

# Import utility functions from vLLM tests
from tests.kernels.moe.utils import make_quantized_test_activations, make_test_weight
from tests.kernels.quant_utils import native_batched_masked_quant_matmul

# Benchmark shapes for batched MoE
# (E: num_experts, M: max_tokens_per_expert, K: hidden_dim, N: intermediate_dim, fp8, block_quant)
# BATCHED_MM_X_VALS = [(E, M, K, N, False, False) for E in [8, 32] for M in [32, 224, 512] for K in [128, 1024] for N in [128, 1024]]
# Each pair represent transformation for SwiGLU and then output transformation
MM_CONFIGS_BF16 = sum([[(E, M, hidden_size, int_size * 2, fp8, block_quant),  # input -> swiglu input
                        (E, M, int_size, hidden_size, fp8, block_quant)]  # swiglu output -> final output
                       for M in [1, 8, 64, 256]
                       for E, hidden_size, int_size, fp8, block_quant in [
                           # llama4 scout
                           (16, 5120, 8192, False, False),
                           # gpt-oss 20b
                           (32, 2880, 2880, False, False),
                           # gpt-oss 120b
                           (128, 2880, 2880, False, False),
                           # qwen3-235b-A22B
                           (128, 4096, 1536, False, False),
                           # qwen3-30b-A3B
                           (128, 2048, 768, False, False),
                           # qwen3-next-80B
                           (512, 2048, 512, False, False),
                       ]], [])

MM_CONFIGS_FP8 = sum([[(E, M, hidden_size, int_size * 2, fp8, block_quant),
                       (E, M, int_size, hidden_size, fp8, block_quant)]
                      for M in [1, 8, 64, 256]
                      for E, hidden_size, int_size, fp8, block_quant in [
                          # deepseek V3 (R1 is the same)
                          # 3.5 GBs of weights!
                          (256, 7168, 2048, True, True),
                          #  # qwen3-235b-A22B
                          (128, 4096, 1536, True, False),
                          # qwen3-30b-A3B
                          (128, 2048, 768, True, False),
                      ]], [])

DEVICE_TOTAL_MEMORY_BYTES = benchmark_suite.get_total_gpu_memory_bytes()


def is_enough_memory(x_val, safety_factor=0.80):
    E, M, K, N, fp8, block_quant = x_val

    # A and B bf16 originals (always allocated, freed later in fp8 case)
    # A memory is doubled because make_quantized_test_activations uses out-of-place / 15
    a_mem = E * M * K * 2 * 2
    b_mem = E * N * K * 2

    if fp8:
        # fp8 copies + scales
        a_mem += E * M * K
        b_mem += E * N * K
        if block_quant:
            bk_n, bk_k = 128, 128
            a_mem += E * ((M + bk_n - 1) // bk_n) * ((K + bk_k - 1) // bk_k) * 4
            b_mem += E * ((N + bk_n - 1) // bk_n) * ((K + bk_k - 1) // bk_k) * 4
        else:
            a_mem += E * 4
            b_mem += E * 4

    # C, ref (E, M, N) bf16 each + num_expert_tokens (E,) int32
    out_mem = 2 * E * M * N * 2 + E * 4

    # Peak is before bf16 originals are freed
    required_memory = a_mem + b_mem + out_mem
    print(f"Estimated memory for {x_val}: {required_memory * 1e-9:.2f} GB", flush=True)
    return required_memory < DEVICE_TOTAL_MEMORY_BYTES * safety_factor


def filter_by_memory(configs):
    result = []
    for x_val in configs:
        if is_enough_memory(x_val):
            result.append(x_val)
        else:
            print(f"'{x_val}' combination skipped, OOM expected")
    return result


MM_CONFIGS_BF16 = filter_by_memory(MM_CONFIGS_BF16)
MM_CONFIGS_FP8 = filter_by_memory(MM_CONFIGS_FP8)


def get_batched_mm_benchmark(
    providers_filter: Optional[list[str]] = None,
    per_act_token_quant: bool = False,
    fp8=False,
    is_td_patched=False,
):
    """
    Returns a Mark object containing a Benchmark object for batched matrix multiplication.
    """
    supported_providers = {
        'triton' + ('-td' if is_td_patched else ''): 'triton' + ('-td' if is_td_patched else ''),
        'pytorch': 'pytorch',
    }
    if fp8:
        # pytorch is very slow with fp8 case, for (8, 64, 1024, 2048) case it has ~0.15 TFlops vs 1.5 for triton
        del supported_providers['pytorch']

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    configs = MM_CONFIGS_FP8 if fp8 else MM_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['num_experts', 'max_tokens_per_expert', 'K', 'N', 'fp8', 'block_quant'],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('red', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name='moe-gemm-performance' + ('-td' if is_td_patched else ''),
            args={},
        ))
    def benchmark(num_experts, max_tokens_per_expert, K, N, fp8, block_quant, provider):
        torch.manual_seed(20)
        n_warmup = 600

        act_dtype = torch.bfloat16
        use_fp8_w8a8 = fp8
        quant_dtype = torch.float8_e4m3fn if fp8 else None

        block_shape = (128, 128) if block_quant else None

        # Create random number of expert tokens
        num_expert_tokens = torch.randint(low=0, high=max_tokens_per_expert + 1, size=(num_experts, ), device='xpu',
                                          dtype=torch.int32)
        out_shape = (num_experts, max_tokens_per_expert, N)

        # Create quantized test activations
        A, A_q, A_scale = make_quantized_test_activations(
            num_experts,
            max_tokens_per_expert,
            K,
            in_dtype=act_dtype,
            quant_dtype=quant_dtype,
            block_shape=block_shape,
            per_act_token_quant=per_act_token_quant,
        )

        # Create test weights (only need B matrix for batched MM)
        B, B_q, B_scale, _ = make_test_weight(
            num_experts,
            N,
            K,
            in_dtype=act_dtype,
            quant_dtype=quant_dtype,
            block_shape=block_shape,
            per_out_ch_quant=per_act_token_quant,
        )

        # Free unused bf16 originals in the fp8 case
        if quant_dtype is not None:
            del A, B
        quantiles = [0.5, 0.0, 1.0]

        C = torch.zeros(out_shape, device='xpu', dtype=act_dtype)
        compute_tl_dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[C.dtype]
        rtol = 6e-2 if act_dtype == torch.bfloat16 else 1e-2
        atol = 6e-2 if act_dtype == torch.bfloat16 else 1e-2
        ref = torch.zeros(out_shape, device='xpu', dtype=act_dtype)

        def torch_fn():
            native_batched_masked_quant_matmul(A_q, B_q, ref, num_expert_tokens, A_scale, B_scale, block_shape,
                                               per_act_token_quant)
            return ref

        if provider == 'pytorch':
            # PyTorch reference implementation using native_batched_masked_quant_matmul
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                torch_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider.startswith('triton'):

            def triton_fn():
                invoke_moe_batched_triton_kernel(
                    A_q,
                    B_q,
                    C,
                    num_expert_tokens,
                    compute_tl_dtype,
                    A_scale,
                    B_scale,
                    None,
                    use_fp8_w8a8,
                    False,
                    False,
                    config={'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32 if fp8 else 16},
                    # config={'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32 if dtype.itemsize > 1 else 32},
                    per_act_token_quant=per_act_token_quant,
                    block_shape=block_shape,
                )
                return C

            # Verify correctness against reference
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
        # Memory bandwidth: A (E*M*K*2) + B (E*K*N*2) + C (E*M*N*4) bytes
        # Compute: E * M * N * K * 2 FLOPs (multiply-add)
        num_activated_experts = num_expert_tokens.ne(0).sum().item()
        num_tokens = num_expert_tokens.sum().item()

        def gbps(ms):
            n_bytes = 1 if fp8 else 2
            # In practice due to the uniform distribution of lengths, on average half of the tokens are used,
            # let's take that into account
            total_bytes = (
                # B matrix, we only have to load activated experts
                num_activated_experts * (K * N * n_bytes) +
                # A matrix - activations, we only load part of tokens
                num_tokens * K * n_bytes +
                # C matrix - outputs, we only load/store part of tokens
                num_tokens * N * 2)
            return total_bytes * (1e-9) / (ms * 1e-3)

        def tflops(ms):
            total_flops = num_experts * max_tokens_per_expert * N * K * 2
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    # Run batched MM benchmark
    is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    print('Running batched MM benchmark...')
    _benchmark_mm = get_batched_mm_benchmark(fp8=(os.getenv('FP8', '0') == '1'), is_td_patched=is_td_patched)
    _benchmark_mm.run(show_plots=False, print_data=True)
