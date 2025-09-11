# This benchmark requires a Pytorch version with FlexAttention support for XPU available
from functools import lru_cache
import os
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

import torch
import torch._inductor
import torch._inductor.lowering
import torch._inductor.kernel
import torch._inductor.kernel.flex.flex_attention as flex_attn
from torch._inductor.template_heuristics.triton import FlexConfig, FlexDecodeConfig

import triton_kernels_benchmark as benchmark_suit
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Use TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 or uncomment the following line to print the auto-tune results.
# torch._inductor.config.max_autotune_gemm = True


def get_flex_attn_fwd_configs(*args, **kwargs):  # pylint: disable=unused-argument
    configs = [
        FlexConfig(32, 16, 2, 4),
        FlexConfig(128, 64, 2, 16),
        FlexConfig(128, 64, 2, 8),
        FlexConfig(128, 32, 2, 16),
        FlexConfig(128, 32, 2, 8),
    ]
    return configs


def get_flex_decode_configs(*args, **kwargs):  # pylint: disable=unused-argument
    configs = [
        FlexDecodeConfig(32, 1, 2),
        FlexDecodeConfig(32, 1, 1),
        FlexDecodeConfig(32, 2, 2),
        FlexDecodeConfig(32, 2, 1),
        FlexDecodeConfig(64, 1, 2),
        FlexDecodeConfig(64, 1, 1),
        FlexDecodeConfig(64, 2, 2),
        FlexDecodeConfig(64, 2, 1),
    ]
    return configs


# There is a auto-tuning requirement to get the best configuration for the flex attention.
# The pytorch flex attention doesn't support auto-tuning by user by default.
# Overriding the get_flex_attention_fwd_configs method to provide custom configurations for auto-tuning on XPU.
flex_attn.V.choices.get_flex_attention_fwd_configs = get_flex_attn_fwd_configs
flex_attn.V.choices.get_flex_decode_configs = get_flex_decode_configs

torch._dynamo.config.recompile_limit = 100  # pylint: disable=protected-access

# Compile the flex_attention function
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device=DEVICE):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def causal_mask(_, __, q_idx, kv_idx):
    return q_idx >= kv_idx


throughput_test = os.getenv('THROUGHPUT_TEST', '0') == '1'
batch_size = int(os.getenv('BATCH_SIZE', '1'))
batch_sizes = [16, 32, 64] if throughput_test else [batch_size]
fa_kernel_mode = os.getenv('FA_KERNEL_MODE', 'fwd')


# Kernel profiling for Backward mode is not working as expected:
# For details: https://github.com/pytorch/pytorch/issues/144778
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        x_names=['Z', 'H_q', 'H_kv', 'N_CTX_q', 'N_CTX_kv', 'D_HEAD_qk', 'D_HEAD_v', 'MODE'],
        x_vals=
        # Multi-head attention. H_q equals H_kv
        # Prefill shapes of Phi3-mini-3.8B
        [[z, 32, 32, 1024, 1024, 96, 96, fa_kernel_mode] for z in batch_sizes] +
        # Prefill shapes of Deepseek-v3
        [[z, 128, 128, 1024, 1024, 192, 128, fa_kernel_mode] for z in batch_sizes] +
        # Append shapes of Phi3-mini-3.8B
        [[z, 32, 32, 512, 1024 + 128 + 512, 96, 96, fa_kernel_mode] for z in batch_sizes] +

        # Multi-query attention. H_kv equals 1.
        # Append shapes of Deepseek-v3 (Nope)
        [[z, 128, 1, 512, 1024 + 128 + 512, 64, 512, fa_kernel_mode] for z in batch_sizes] +
        # Append shapes of Deepseek-v3 (Rope)
        [] +

        # Grouped-query attention. H_q / H_kv > 1
        # Prefill shapes of Llama-3.1-8B
        [[z, 32, 8, 1024, 1024, 128, 128, fa_kernel_mode] for z in batch_sizes] +
        # Prefill shapes of Qwen2-7B
        [[z, 28, 4, 1024, 1024, 128, 128, fa_kernel_mode] for z in batch_sizes] +
        # Append shapes of Llama-3.1-8B
        [[z, 32, 8, 512, 1024 + 128 + 512, 128, 128, fa_kernel_mode] for z in batch_sizes] +
        # Append shapes of Qwen2-7B
        [[z, 28, 4, 512, 1024 + 128 + 512, 128, 128, fa_kernel_mode] for z in batch_sizes] +

        # FlexDecoding configuration. N_CTX_q equals 1. N_CTX_kv >= 1k
        # Decode shapes of Llama-3.1-8B
        [[z, 32, 8, 1, 1024 + 64, 128, 128, fa_kernel_mode] for z in batch_sizes] +
        # Decode shapes of Phi3-mini-3.8B
        [
            # acc = acc.reshape(G, BLOCK_M_PER_HQ, V_HEAD_DIM)
            # ValueError: Shape element 2 must be a power of 2
            # [z, 32, 32, 1, 1024 + 64, 96, 96, fa_kernel_mode] for z in batch_sizes
        ] +
        # Decode shapes of Qwen2-7B
        [
            # torch._inductor.exc.InductorError: LoweringException: ValueError: Number of shared query heads sharing the same KV head must be power of 2.
            # [z, 28, 4, 1, 1024 + 64, 128, 128, fa_kernel_mode] for z in batch_sizes
        ] +
        # Decode shapes of Deepseek-v3 (Nope)
        [
            # There is an known issue in IGC for kernel with extreme register pressure.
            # Enable this case later with new IGC.
            # RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME
            # [z, 128, 1, 1, 1024, 64, 512, fa_kernel_mode] for z in batch_sizes
        ] +
        # Decode shapes of Deepseek-v3 (Rope)
        [],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],
        plot_name='flexAttnCausal-performance',
        args={},
    ))
def benchmark(Z, H_q, H_kv, N_CTX_q, N_CTX_kv, D_HEAD_qk, D_HEAD_v, MODE, provider):
    if MODE not in ('fwd', 'bwd'):
        raise ValueError(f"Invalid MODE: {MODE}. Expected 'fwd' or 'bwd'.")
    dtype = torch.float16
    q = torch.randn((Z, H_q, N_CTX_q, D_HEAD_qk), device=DEVICE, dtype=dtype, requires_grad=MODE == 'bwd')
    k = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD_qk), device=DEVICE, dtype=dtype, requires_grad=MODE == 'bwd')
    v = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD_v), device=DEVICE, dtype=dtype, requires_grad=MODE == 'bwd')
    sm_scale = 0.125

    quantiles = [0.5, 0.0, 1.0]
    block_mask = create_block_mask_cached(causal_mask, 1, 1, N_CTX_q, N_CTX_kv, device=DEVICE)
    torch_fn = lambda: flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=not H_q == H_kv)

    if provider == 'torch':
        if MODE == 'bwd':
            min_ms = float('nan')
            max_ms = float('nan')
            mean = float('nan')
            cv = float('nan')
        else:
            _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(torch_fn, n_warmup=10, n_repeat=10,
                                                                  quantiles=quantiles, device=DEVICE)

    elif provider == 'triton':
        kernel_options = {'BLOCKS_ARE_CONTIGUOUS': True}
        triton_fn = lambda: compiled_flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=(
            not H_q == H_kv), kernel_options=kernel_options)
        if MODE == 'bwd':
            torch_o = torch_fn()
            backwards_grad = torch.randn_like(torch_o)
            torch_grads = torch.autograd.grad((torch_o, ), (q, k, v), backwards_grad, retain_graph=True)
            eager_tensors = (torch_o, *torch_grads)
            triton_o = triton_fn()
            triton_grads = torch.autograd.grad((triton_o, ), (q, k, v), backwards_grad, retain_graph=True)
            compiled_tensors = (triton_o, *triton_grads)

            tensor_names = ['out', 'grad_query', 'grad_key', 'grad_value']
            for eager, compiled, name in zip(eager_tensors, compiled_tensors, tensor_names):
                benchmark_suit.assert_close(lambda: eager, lambda: compiled, atol=1e-2, rtol=1e-3,  # pylint: disable=cell-var-from-loop
                                            err_msg=f'Error comparing {name} between triton and torch')

            triton_fn = lambda: torch.autograd.grad((triton_o, ), (q, k, v), backwards_grad, retain_graph=True)
        else:
            benchmark_suit.assert_close(triton_fn, torch_fn, atol=1e-2, rtol=1e-3, err_msg='triton to torch')

        # Needs more warmup on B580 for some reason
        benchmark_suit.do_prewarmup(triton_fn)
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(
            triton_fn, n_warmup=200, n_repeat=10, quantiles=quantiles, device=DEVICE, grad_to_none=(q, k, v),
            benchmark_label=None if MODE == 'fwd' else 'CompiledFunctionBackward')

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    qk_flops = H_q * N_CTX_q * N_CTX_kv * D_HEAD_qk  # mul + add, causal=True. Only the lower triangle is computed.
    pv_flops = H_q * N_CTX_q * D_HEAD_v * N_CTX_kv  # mul + add, causal=True. Only the lower triangle is computed.
    tflops = lambda mean: Z * (qk_flops + pv_flops) * (1e-12) / (mean * 1e-3)

    q_elems = H_q * N_CTX_q * D_HEAD_qk
    k_elems = H_kv * N_CTX_kv * D_HEAD_qk
    v_elems = H_kv * N_CTX_kv * D_HEAD_v
    gbps = lambda mean: Z * (q_elems + k_elems + v_elems) * 2 * (1e-9) / (mean * 1e-3)  # float16 2 bytes

    if MODE == 'bwd':
        # The tflops and gbps are aligned to the one in flash_attention_benchmark.
        tflops = lambda mean: 2.5 * Z * (qk_flops + pv_flops) * (1e-12) / (mean * 1e-3)
        gbps = lambda mean: 2.5 * Z * (q_elems + k_elems + v_elems) * 2 * (1e-9) / (mean * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
