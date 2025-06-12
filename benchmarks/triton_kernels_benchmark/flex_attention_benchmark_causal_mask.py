# This benchmark requires a Pytorch version with FlexAttention support for XPU available
from functools import lru_cache
import os
from torch.nn.attention.flex_attention import (
    create_block_mask,
    create_mask,
    flex_attention,
)

import torch
import torch.nn.functional as F
import torch._inductor
import torch._inductor.lowering
import torch._inductor.kernel
import torch._inductor.kernel.flex_attention as flex_attn
import torch._inductor.virtualized

import triton_kernels_benchmark as benchmark_suit

# Use TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 or uncomment the following line to print the auto-tune results.
# torch._inductor.config.max_autotune_gemm = True


def get_xpu_config(*args, **kwargs):  # pylint: disable=unused-argument
    # BLOCK_M, BLOCK_N, num_warps, num_stages
    configs = [
        (32, 16, 4, 2),
        (128, 64, 16, 2),
        (128, 64, 8, 2),
        (128, 32, 16, 2),
        (128, 32, 8, 2),
    ]
    return configs


# There is a auto-tuning requirement to get the best configuration for the flex attention.
#The pytorch flex attention doesn't support auto-tuning by user in default.
#Overriding the _get_xpu_config method to provide custom configurations for auto-tuning on XPU.
flex_attn._get_xpu_config = get_xpu_config  # pylint: disable=protected-access

torch._dynamo.config.recompile_limit = 100  # pylint: disable=protected-access

# Compile the flex_attention function
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device='xpu'):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def causal_mask(_, __, q_idx, kv_idx):
    return q_idx >= kv_idx


throughput_test = os.getenv('THROUGHPUT_TEST', '0') == '1'
batch_sizes = [16, 32, 64] if throughput_test else [1]


# Kernel profiling for Backward mode is not working as expected:
# For details: https://github.com/pytorch/pytorch/issues/144778
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        x_names=['Z', 'H_q', 'H_kv', 'N_CTX_q', 'N_CTX_kv', 'D_HEAD_qk', 'D_HEAD_v', 'MODE'],
        x_vals=
        # Multi-head attention. H_q equals H_kv
        # Prefill shapes of Phi3-mini-3.8B
        [[z, 32, 32, 1024, 1024, 96, 96, 'fwd'] for z in batch_sizes] +
        # Prefill shapes of Deepseek-v3
        [[z, 128, 128, 1024, 1024, 192, 128, 'fwd'] for z in batch_sizes] +
        # Append shapes of Phi3-mini-3.8B
        [[z, 32, 32, 512, 1024 + 128 + 512, 96, 96, 'fwd'] for z in batch_sizes] +

        # Multi-query attention. H_kv equals 1.
        # Append shapes of Deepseek-v3 (Nope)
        [
            # RuntimeError: No valid triton configs. OutOfResources: out of resource: shared memory, Required: 133120, Hardware limit: 131072.
            # [z, 128, 1, 512, 1024 + 128 + 512, 64, 512, 'fwd'] for z in batch_sizes
        ] +
        # Append shapes of Deepseek-v3 (Rope)
        [] +

        # Grouped-query attention. H_q / H_kv > 1
        # Prefill shapes of Llama-3.1-8B
        [[z, 32, 8, 1024, 1024, 128, 128, 'fwd'] for z in batch_sizes] +
        # Prefill shapes of Qwen2-7B
        [[z, 28, 4, 1024, 1024, 128, 128, 'fwd'] for z in batch_sizes] +
        # Append shapes of Llama-3.1-8B
        [[z, 32, 8, 512, 1024 + 128 + 512, 128, 128, 'fwd'] for z in batch_sizes] +
        # Append shapes of Qwen2-7B
        [[z, 28, 4, 512, 1024 + 128 + 512, 128, 128, 'fwd'] for z in batch_sizes] +

        # FlexDecoding configuration. N_CTX_q equals 1. N_CTX_kv >= 1k
        # Decode shapes of Llama-3.1-8B
        [[z, 32, 8, 1, 1024 + 64, 128, 128, 'fwd'] for z in batch_sizes] +
        # Decode shapes of Phi3-mini-3.8B
        [
            # acc = acc.reshape(G, BLOCK_M_PER_HQ, V_HEAD_DIM)
            # ValueError: Shape element 2 must be a power of 2
            # [z, 32, 32, 1, 1024 + 64, 96, 96, 'fwd'] for z in batch_sizes
        ] +
        # Decode shapes of Qwen2-7B
        [
            # torch._inductor.exc.InductorError: LoweringException: ValueError: Number of shared query heads sharing the same KV head must be power of 2.
            # [z, 28, 4, 1, 1024 + 64, 128, 128, 'fwd'] for z in batch_sizes
        ] +
        # Decode shapes of Deepseek-v3 (Nope)
        [
            # RuntimeError: No valid triton configs. OutOfResources: out of resource: shared memory, Required: 264192, Hardware limit: 131072.
            # [z, 128, 1, 1, 1024, 64, 512, 'fwd'] for z in batch_sizes
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
    assert MODE in ['fwd']
    dtype = torch.float16
    q = torch.randn((Z, H_q, N_CTX_q, D_HEAD_qk), device='xpu', dtype=dtype, requires_grad=MODE == 'bwd')
    k = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD_qk), device='xpu', dtype=dtype, requires_grad=MODE == 'bwd')
    v = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD_v), device='xpu', dtype=dtype, requires_grad=MODE == 'bwd')
    sm_scale = 0.125
    if MODE == 'bwd':
        sm_scale = 1.3

    quantiles = [0.5, 0.0, 1.0]
    block_mask = create_block_mask_cached(causal_mask, 1, 1, N_CTX_q, N_CTX_kv, device='xpu')
    torch_fn = lambda: flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=not H_q == H_kv)

    if provider == 'torch':
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(torch_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    elif provider == 'triton':
        kernel_options = {'BLOCKS_ARE_CONTIGUOUS': True}
        triton_fn = lambda: compiled_flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=(
            not H_q == H_kv), kernel_options=kernel_options)
        if MODE == 'bwd':
            triton_o = triton_fn()
            triton_do = torch.randn_like(triton_o)
            triton_fn = lambda: triton_o.backward(triton_do, retain_graph=True)

        benchmark_suit.assert_close(triton_fn, torch_fn, atol=1e-2, rtol=1e-3, err_msg='triton to torch')
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    elif provider == 'onednn':
        # OneDNN only supports MHA.
        if H_q == H_kv:
            mask = create_mask(causal_mask, 1, 1, N_CTX_q, N_CTX_kv, device=q.device)
            xformers_fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            if MODE == 'bwd':
                xformers_o = xformers_fn()
                xformers_do = torch.randn_like(xformers_o)
                xformers_fn = lambda: xformers_o.backward(xformers_do, retain_graph=True)
            _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(xformers_fn, n_warmup=10, n_repeat=10,
                                                                  quantiles=quantiles)
        else:
            _, min_ms, max_ms, mean, cv = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    qk_flops = H_q * N_CTX_q * N_CTX_kv * D_HEAD_qk * 2  # mul + add
    pv_flops = H_q * N_CTX_q * D_HEAD_v * N_CTX_kv * 2  # mul + add
    tflops = lambda mean: Z * (qk_flops + pv_flops) * (1e-12) / (mean * 1e-3)

    q_elems = H_q * N_CTX_q * D_HEAD_qk
    k_elems = H_kv * N_CTX_kv * D_HEAD_qk
    v_elems = H_kv * N_CTX_kv * D_HEAD_v
    gbps = lambda mean: Z * (q_elems + k_elems + v_elems) * 2 * (1e-9) / (mean * 1e-3)  # float16 2 bytes

    if MODE == 'bwd':
        tflops = lambda mean: 2.5 * 2 * 2 * Z * H_q * N_CTX_q * N_CTX_kv * D_HEAD_qk * (1e-12) / (mean * 1e-3)
        gbps = lambda mean: 2.5 * Z * H_q * (N_CTX_q * D_HEAD_qk + N_CTX_kv * D_HEAD_qk) * 2 * 2 * (1e-9) / (mean * 1e-3
                                                                                                             )

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
