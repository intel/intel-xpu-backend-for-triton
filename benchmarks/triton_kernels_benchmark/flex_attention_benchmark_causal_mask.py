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

import triton_kernels_benchmark as benchmark_suite
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

if 'B580' in torch.xpu.get_device_name():
    old_count = len(batch_sizes)
    batch_sizes = [size for size in batch_sizes if size < 16]
    if len(batch_sizes) != old_count:
        print('Skipping running batch_sizes >= 16 on b580')


# Kernel profiling for Backward mode is not working as expected:
# For details: https://github.com/pytorch/pytorch/issues/144778
@benchmark_suite.perf_report(
    benchmark_suite.Benchmark(
        x_names=['Z', 'H_q', 'H_kv', 'N_CTX_q', 'N_CTX_kv', 'D_HEAD_qk', 'D_HEAD_v', 'MODE'],
        x_vals=[[z, *params, fa_kernel_mode] for z in batch_sizes for params in [
            # Multi-head attention. H_q equals H_kv
            (32, 32, 1024, 1024, 96, 96),  # Prefill shapes of Phi3-mini-4k-instruct
        ]],
        line_arg='provider',
        line_vals=['torch'],
        line_names=['Torch'],
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],
        plot_name='flexAttnCausal-performance',
        args={},
    ))
def benchmark(Z, H_q, H_kv, N_CTX_q, N_CTX_kv, D_HEAD_qk, D_HEAD_v, MODE, provider):
    do_bench = benchmark_suite.get_do_bench(n_warmup=600, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
    dtype = torch.float16
    q = torch.randn((Z, H_q, N_CTX_q, D_HEAD_qk), device=DEVICE, dtype=dtype, requires_grad=MODE == 'bwd')
    k = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD_qk), device=DEVICE, dtype=dtype, requires_grad=MODE == 'bwd')
    v = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD_v), device=DEVICE, dtype=dtype, requires_grad=MODE == 'bwd')
    sm_scale = 0.125

    block_mask = create_block_mask_cached(causal_mask, 1, 1, N_CTX_q, N_CTX_kv, device=DEVICE)
    torch_fn = lambda: flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=not H_q == H_kv)
    #torch_fn()
    #min_ms = float('nan')
    #max_ms = float('nan')
    #mean = float('nan')
    #cv = float('nan')
    _, min_ms, max_ms, mean, cv = do_bench(torch_fn, device=DEVICE)

    qk_flops = H_q * N_CTX_q * N_CTX_kv * D_HEAD_qk  # mul + add, causal=True. Only the lower triangle is computed.
    pv_flops = H_q * N_CTX_q * D_HEAD_v * N_CTX_kv  # mul + add, causal=True. Only the lower triangle is computed.
    tflops = lambda mean: Z * (qk_flops + pv_flops) * (1e-12) / (mean * 1e-3)

    q_elems = H_q * N_CTX_q * D_HEAD_qk
    k_elems = H_kv * N_CTX_kv * D_HEAD_qk
    v_elems = H_kv * N_CTX_kv * D_HEAD_v
    gbps = lambda mean: Z * (q_elems + k_elems + v_elems) * 2 * (1e-9) / (mean * 1e-3)  # float16 2 bytes

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
