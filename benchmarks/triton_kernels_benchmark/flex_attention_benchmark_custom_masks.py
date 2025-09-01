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

import triton_kernels_benchmark as benchmark_suit

torch._dynamo.config.recompile_limit = 100  # pylint: disable=protected-access

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device='xpu'):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


# Default values for NATTEN mask:
# Consider a 2D image of size (G_H x G_W) flattened into a sequence of tokens.
# Queries attend to keys in a fixed kernel area (K_H x K_W)
G_H = 128
G_W = 128
K_H = 13
K_W = 13


def get_x_y(idx):
    return idx // G_W, idx % G_W


def natten_mask(_, __, q_idx, kv_idx):
    q_x, q_y = get_x_y(q_idx)
    kv_x, kv_y = get_x_y(kv_idx)
    # kernel nominally attempts to center itself on the query, but kernel center
    # is clamped to a fixed distance (kernel half-length) from the canvas edge
    kernel_x = q_x.clamp(K_W // 2, (G_W - 1) - K_W // 2)
    kernel_y = q_y.clamp(K_H // 2, (G_H - 1) - K_H // 2)
    hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
    vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
    return hori_mask & vert_mask


def alibi_functional(score, _, h, q_idx, kv_idx):
    scale = torch.exp2(-((h + 1) * 8.0 / G_H))
    bias = (kv_idx - q_idx) * scale
    return score + bias


# Kernel profiling for Backward mode is not working as expected:
# For details: https://github.com/pytorch/pytorch/issues/144778
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        x_names=['Z', 'H', 'N_CTX', 'D_HEAD', 'MASK', 'MODE'],
        x_vals=[[z, h, 16384 // z, dhead, mask, mode]
                for z in [4, 8, 16, 32]
                for (h, dhead) in [(16, 128), (32, 64)]
                for mask in ['NATTEN', 'Alibi']
                for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]]  #
        + [[4, 48, 1024, 64, mask, mode]
           for mask in ['NATTEN', 'Alibi']
           for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]]  #
        + [[z, h, 1024, dhead, mask, mode]
           for z in [1, 2, 4, 8, 16, 32, 64]
           for (h, dhead) in [(8, 128), (32, 96), (4, 128)]
           for mask in ['NATTEN', 'Alibi']
           for mode in [os.getenv('FA_KERNEL_MODE', 'fwd')]],
        line_arg='provider',
        line_vals=['triton'] + (['onednn'] if '580' not in torch.xpu.get_device_name() else []),
        line_names=['Triton'] + (['OneDNN'] if '580' not in torch.xpu.get_device_name() else []),
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],
        plot_name='flexAttnMasks-performance',
        args={},
    ))
def benchmark(Z, H, N_CTX, D_HEAD, MASK, MODE, provider):
    assert MODE in ['fwd', 'bwd']
    assert MASK in ['NATTEN', 'Alibi']
    dtype = torch.float16
    q = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
    k = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)
    v = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=True)

    mask_mod = None
    score_mod = None
    if MASK == 'NATTEN':
        mask_mod = natten_mask
    elif MASK == 'Alibi':
        score_mod = alibi_functional

    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, N_CTX, N_CTX, device=q.device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, N_CTX, N_CTX, device=q.device)

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':
        kernel_options = {'num_stages': 2, 'num_warps': 16 if D_HEAD == 128 else 8, 'BLOCKS_ARE_CONTIGUOUS': True}
        triton_fn = lambda: flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, kernel_options=
                                           kernel_options)
        if MODE == 'bwd':
            triton_o = triton_fn()
            triton_do = torch.randn_like(triton_o)
            triton_fn = lambda: triton_o.backward(triton_do, retain_graph=True)
        # Needs more warmup on B580 for some reason
        # benchmark_suit.do_prewarmup(triton_fn)
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=5, quantiles=quantiles)
        # Values checking cannot be implemented for these case as :
        # "The operator 'aten::_scaled_dot_product_flash_attention_for_cpu' is not currently implemented for the XPU device"

    elif provider == 'onednn':
        xformers_fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if MODE == 'bwd':
            xformers_o = xformers_fn()
            xformers_do = torch.randn_like(xformers_o)
            xformers_fn = lambda: xformers_o.backward(xformers_do, retain_graph=True)
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(xformers_fn, n_warmup=10, n_repeat=10,
                                                              quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda mean: 2 * 2 * Z * H * N_CTX * N_CTX * D_HEAD * (1e-12) / (mean * 1e-3)
    gbps = lambda mean: Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    if MODE == 'bwd':
        tflops = lambda mean: 2.5 * 2 * 2 * Z * H * N_CTX * N_CTX * D_HEAD * (1e-12) / (mean * 1e-3)
        gbps = lambda mean: 2.5 * Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
