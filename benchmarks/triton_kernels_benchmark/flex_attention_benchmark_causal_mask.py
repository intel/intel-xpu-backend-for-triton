# This benchmark requires a Pytorch version with FlexAttention support for XPU available
from functools import lru_cache
import os
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

import torch
import triton_kernels_benchmark as benchmark_suit
from triton_kernels_benchmark import xetla_kernel

torch._dynamo.config.recompile_limit = 100  # pylint: disable=protected-access

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device='xpu'):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def causal_mask(_, __, q_idx, kv_idx):
    return q_idx >= kv_idx


# Kernel profiling for Backward mode is not working as expected:
# For details: https://github.com/pytorch/pytorch/issues/144778
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        x_names=['B', 'H_q', 'H_kv', 'N_CTX_q', 'N_CTX_kv', 'D_HEAD', 'MODE'],
        x_vals=
        # Multi-head attention. H_q equals H_kv
        [
            # [z, h, h, n_ctx_q, n_ctx_kv, dhead, mode]
            # for z in [1, 2, 4, 8, 16, 32]
            # for h in [1, 2, 4, 8, 16, 32]
            # for n_ctx_q in [128, 256, 512, 1024]
            # for n_ctx_kv in [128, 256, 512, 1024]
            # for dhead in [64, 128, 256]
            # for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]
        ] +
        # Multi-query attention. H_kv equals 1.
        [] +
        # Grouped-query attention. H_q / H_kv > 1
        [[z, h_q, h_kv, n_ctx_q, n_ctx_kv, dhead, mode]
         for z in [1, 2, 4, 8, 16, 32]
         for h_q, h_kv in [(2, 1), (4, 1), (4, 2), (8, 4), (16, 8)]
         for n_ctx_q in [128, 256, 512, 1024]
         for n_ctx_kv in [128, 256, 512, 1024]
         for dhead in [64, 128, 256]
         for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]] +
        # FlexDecoding configuration. N_CTX_q equals 1. N_CTX_kv >= 1k
        [],
        # [[z, h, 16384 // z, dhead, causal, mode]
        #         for z in [1, 2, 4, 8, 16, 32]
        #         for (h, dhead) in [(16, 128), (32, 64)]
        #         for causal in [True]
        #         for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]]  #
        # + [[4, 48, 1024, 64, True, mode] for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]]  #
        # + [[z, h, 1024, dhead, True, mode]
        #    for z in [1, 2, 4, 8, 16, 32, 64]
        #    for (h, dhead) in [(8, 128), (32, 96), (4, 128)]
        #    for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]],
        line_arg='provider',
        line_vals=['triton', 'xetla'],
        line_names=['Triton', 'XeTLA'],
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],
        plot_name='flexAttnCausal-performance',
        args={},
    ))
def benchmark(Z, H_q, H_kv, N_CTX_q, N_CTX_kv, D_HEAD, MODE, provider):
    assert MODE in ['fwd', 'bwd']
    dtype = torch.float16
    q = torch.randn((Z, H_q, N_CTX_q, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
    k = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
    v = torch.randn((Z, H_kv, N_CTX_kv, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
    sm_scale = 0.125
    if MODE == 'bwd':
        sm_scale = 1.3

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':
        kernel_options = {'num_stages': 2, 'num_warps': 16 if D_HEAD == 128 else 8, 'BLOCKS_ARE_CONTIGUOUS': True}
        block_mask = create_block_mask_cached(causal_mask, 1, 1, N_CTX_q, N_CTX_kv, device='xpu')
        triton_fn = lambda: flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, enable_gqa=(not H_q == H_kv),
                                           kernel_options=kernel_options)
        if MODE == 'bwd':
            triton_o = triton_fn()
            triton_do = torch.randn_like(triton_o)
            triton_fn = lambda: triton_o.backward(triton_do, retain_graph=True)
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    elif provider == 'xetla':
        if (H_q == H_kv) and (N_CTX_q == N_CTX_kv):
            xetla_fn = None
            H = H_q
            N_CTX = N_CTX_q
            if MODE == 'fwd':
                module_name = 'flash_attn_causal_True'.lower()
                func = getattr(xetla_kernel, module_name)
                out = torch.empty_like(q, device='xpu', dtype=dtype)
                size_score = Z * H * N_CTX * N_CTX
                size_attn_mask = Z * N_CTX * N_CTX
                dropout_mask = torch.empty((size_score, ), device='xpu', dtype=torch.uint8)
                bias = torch.empty((size_attn_mask, ), device='xpu', dtype=dtype)
                size_ml = Z * H * N_CTX
                m = torch.empty((size_ml, ), device='xpu', dtype=torch.float)
                l = torch.empty((size_ml, ), device='xpu', dtype=torch.float)
                xetla_fn = lambda: func(q, k, v, out, dropout_mask, bias, m, l, Z, H, D_HEAD, N_CTX, N_CTX, sm_scale)
            if MODE == 'bwd':
                module_name = 'flash_attn_bwd_causal_True'.lower()
                func = getattr(xetla_kernel, module_name)
                grad_out = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                bias = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                dropout = torch.empty_like(q, device='xpu', dtype=torch.uint8)
                out = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                log_sumexp = torch.zeros(q.size(), device='xpu', dtype=dtype, requires_grad=True)
                workspace = torch.zeros(q.size(), device='xpu', dtype=dtype, requires_grad=True)
                grad_q_tmp = torch.zeros(q.size(), device='xpu', dtype=dtype, requires_grad=True)
                alpha = sm_scale
                dropout_prob = 0
                grad_query = torch.empty_like(q, device='xpu', dtype=dtype, requires_grad=True)
                grad_key = torch.empty_like(k, device='xpu', dtype=dtype, requires_grad=True)
                grad_value = torch.empty_like(v, device='xpu', dtype=dtype, requires_grad=True)
                grad_bias = torch.empty_like(bias, device='xpu', dtype=dtype, requires_grad=True)
                bias_strideB = -1
                bias_strideN = -1
                bias_strideF = -1
                attn_mask_padding = 0

                xetla_fn = lambda: func(grad_out, q, k, v, bias, dropout, out, log_sumexp, workspace, grad_q_tmp, alpha,
                                        dropout_prob, grad_query, grad_key, grad_value, grad_bias, Z, H, D_HEAD, N_CTX,
                                        N_CTX, bias_strideB, bias_strideN, bias_strideF, attn_mask_padding)
            _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(xetla_fn, n_warmup=10, n_repeat=10,
                                                                  quantiles=quantiles)
        else:
            _, min_ms, max_ms, mean, cv = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda mean: 2 * 2 * Z * H_q * N_CTX_q * N_CTX_kv * D_HEAD * (1e-12) / (mean * 1e-3)
    gbps = lambda mean: Z * H_q * (N_CTX_q * D_HEAD + N_CTX_kv * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    if MODE == 'bwd':
        tflops = lambda mean: 2.5 * 2 * 2 * Z * H_q * N_CTX_q * N_CTX_kv * D_HEAD * (1e-12) / (mean * 1e-3)
        gbps = lambda mean: 2.5 * Z * H_q * (N_CTX_q * D_HEAD + N_CTX_kv * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
