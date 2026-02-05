# This benchmark requires a Pytorch version with FlexAttention support for XPU available
from functools import lru_cache
import os
from torch.nn.attention.flex_attention import (
    create_block_mask,
    create_mask,
    flex_attention,
    noop_mask,
)
from torch.nn.attention.experimental._paged_attention import PagedAttention

import torch
import torch.nn.functional as F

import triton_kernels_benchmark as benchmark_suite

torch._dynamo.config.recompile_limit = 200  # pylint: disable=protected-access

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device='xpu'):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
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


def count_natten_pairs(N):
    count = 0

    for q_idx in range(N):
        q_x, q_y = get_x_y(q_idx)
        kernel_x = min(max(q_x, K_W // 2), (G_W - 1) - K_W // 2)
        kernel_y = min(max(q_y, K_H // 2), (G_H - 1) - K_H // 2)

        for kv_idx in range(N):
            kv_x, kv_y = get_x_y(kv_idx)
            hori_mask = abs(kernel_x - kv_x) <= K_W // 2
            vert_mask = abs(kernel_y - kv_y) <= K_H // 2
            if hori_mask & vert_mask:
                count += 1

    return count


def alibi_functional(score, _, h, q_idx, kv_idx):
    scale = torch.exp2(-((h + 1) * 8.0 / G_H))
    bias = (kv_idx - q_idx) * scale
    return score + bias


SOFT_CAP = 50


def tanh_softcap_functional(score, _, __, ___, ____):
    return SOFT_CAP * torch.tanh(score / SOFT_CAP)


def run_bench_contiguous(q, k, v, score_mod, block_mask, kernel_options):
    return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, kernel_options=kernel_options)


def run_bench_paged(q, k, v, score_mod, _, __):
    B, H, N, D = q.shape

    logical_block_mask = create_block_mask_cached(noop_mask, B, H, N, N, device=q.device)
    page_size = logical_block_mask.BLOCK_SIZE[1]
    pages_per_seq = (N + page_size - 1) // page_size
    n_pages = pages_per_seq * B

    paged_attn = PagedAttention(n_pages=n_pages, page_size=page_size, max_batch_size=B, device=q.device)

    for b in range(B):
        paged_attn.reserve(torch.tensor(b, device=q.device), torch.tensor(N, device=q.device))

    k_cache = torch.empty((1, H, n_pages * page_size, D), dtype=k.dtype, device=k.device)
    v_cache = torch.empty((1, H, n_pages * page_size, D), dtype=v.dtype, device=v.device)

    batch_idx = torch.arange(B, device=q.device)
    input_pos = torch.arange(N, device=q.device).view(1, -1).expand(B, -1)

    with torch.no_grad():
        paged_attn.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

    phys_block_mask = paged_attn.convert_logical_block_mask(logical_block_mask, batch_idx=batch_idx)
    phys_score_mod = paged_attn.get_score_mod(score_mod)

    kernel_options = {'num_stages': 2, 'num_warps': 16 if D == 128 else 8, 'BLOCKS_ARE_CONTIGUOUS': False}

    return flex_attention(q, k_cache, v_cache, score_mod=phys_score_mod, block_mask=phys_block_mask,
                          kernel_options=kernel_options)


IS_B580 = '580' in torch.xpu.get_device_name()
MASKS = ['NATTEN', 'Alibi', 'Noop', 'Softcap', 'PagedNoop']
fa_kernel_mode = os.getenv('FA_KERNEL_MODE', 'fwd')


# Kernel profiling for Backward mode is not working as expected:
# For details: https://github.com/pytorch/pytorch/issues/144778
@benchmark_suite.perf_report(
    benchmark_suite.Benchmark(
        x_names=['Z', 'H', 'N_CTX', 'D_HEAD', 'MASK', 'MODE'],
        x_vals=[[z, h, 16384 // z, dhead, mask, mode]
                for z in [8, 16, 32]
                for (h, dhead) in [(16, 128), (32, 64)]
                for mask in MASKS
                for mode in [fa_kernel_mode]]  #
        + [[4, 48, 1024, 64, mask, mode] for mask in MASKS for mode in [fa_kernel_mode]]  #
        + [[z, h, 1024, dhead, mask, mode]
           for z in [1, 2, 4, 8, 16, 32]
           for (h, dhead) in [(8, 128), (32, 96), (4, 128)]
           for mask in MASKS
           for mode in [fa_kernel_mode]],
        line_arg='provider',
        line_vals=['triton'] + (['onednn'] if not IS_B580 else []),
        line_names=['Triton'] + (['OneDNN'] if not IS_B580 else []),
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],
        plot_name='flexAttnMasks-performance',
        args={},
    ))
def benchmark(Z, H, N_CTX, D_HEAD, MASK, MODE, provider):
    torch.xpu.empty_cache()

    # There is still performance variance for triton, probably caused by random choice of autotune config
    do_bench = benchmark_suite.get_do_bench(n_warmup=200, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
    assert MODE in ['fwd', 'bwd']
    assert MASK in MASKS

    if MASK == 'PagedNoop' and (provider == 'onednn' or MODE == 'bwd'):
        # Paged attention benchmark not supported for OneDNN or Backward mode
        return (0, 0, 0), (0, 0, 0), 0.0

    mask_mod = None
    score_mod = None
    requires_grad = True
    bench_func = run_bench_contiguous
    num_pairs = None

    # Maps MASK to a tuple: (mask_mod, score_mod, bench_func, requires_grad, num_pairs)
    MASK_CONFIGS = {
        'NATTEN': (natten_mask, None, run_bench_contiguous, True, count_natten_pairs(N_CTX)),
        'Alibi': (None, alibi_functional, run_bench_contiguous, True, N_CTX * N_CTX),
        'Noop': (noop_mask, None, run_bench_contiguous, True, N_CTX * N_CTX),
        'Softcap': (None, tanh_softcap_functional, run_bench_contiguous, True, N_CTX * N_CTX),
        'PagedNoop': (None, None, run_bench_paged, False, N_CTX * N_CTX),
    }

    mask_mod, score_mod, bench_func, requires_grad, num_pairs = MASK_CONFIGS[MASK]

    dtype = torch.float16
    q = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=requires_grad)
    k = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=requires_grad)
    v = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype, requires_grad=requires_grad)

    if MASK != 'PagedNoop':
        if mask_mod is not None:
            block_mask = create_block_mask_cached(mask_mod, 1, 1, N_CTX, N_CTX, device=q.device)
        else:
            block_mask = None
        sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
        mask = create_mask(sdpa_mask_fn, 1, 1, N_CTX, N_CTX, device=q.device)
    else:
        block_mask = None
        mask = None

    if provider == 'triton':
        kernel_options = {'num_stages': 2, 'num_warps': 16 if D_HEAD == 128 else 8, 'BLOCKS_ARE_CONTIGUOUS': True}

        triton_fn = lambda: bench_func(q, k, v, score_mod, block_mask, kernel_options)
        if MODE == 'bwd':
            triton_o = triton_fn()
            triton_do = torch.randn_like(triton_o)
            triton_fn = lambda: triton_o.backward(triton_do, retain_graph=True)
        _, min_ms, max_ms, mean, cv = do_bench(triton_fn)
        # Values checking cannot be implemented for these case as :
        # "The operator 'aten::_scaled_dot_product_flash_attention_for_cpu' is not currently implemented for the XPU device"

    elif provider == 'onednn':
        xformers_fn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if MODE == 'bwd':
            xformers_o = xformers_fn()
            xformers_do = torch.randn_like(xformers_o)
            xformers_fn = lambda: xformers_o.backward(xformers_do, retain_graph=True)
        _, min_ms, max_ms, mean, cv = do_bench(xformers_fn)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda mean: 2 * 2 * Z * H * num_pairs * D_HEAD * (1e-12) / (mean * 1e-3)
    gbps = lambda mean: Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    if MODE == 'bwd':
        tflops = lambda mean: 2.5 * 2 * 2 * Z * H * num_pairs * D_HEAD * (1e-12) / (mean * 1e-3)
        gbps = lambda mean: 2.5 * Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
