# From
# https://github.com/sgl-project/sglang/blob/6d0364681c8b1abc132cc88f1bb0b7a8a352628f/test/srt/quant/test_triton_scaled_mm.py
# https://github.com/sgl-project/sglang/blob/6d0364681c8b1abc132cc88f1bb0b7a8a352628f/python/sglang/srt/layers/quantization/fp8_kernel.py
import os
from typing import Optional, List

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite

from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


def get_matmul_batched_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'grf_mode': '256'}, num_stages=s, num_warps=32)
        for s in [2, 3]
    ] + [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'grf_mode': m}, num_stages=s, num_warps=w)
        for s in [2]
        for (m, w) in ([('256', 32), ('128', 64)])
    ] + [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'grf_mode': '256'}, num_stages=s, num_warps=32)
        for s in [2]
    ] + [
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 512, 'BLOCK_K': 64, 'grf_mode': '256'}, num_stages=s, num_warps=32)
        for s in [2]
    ] + [
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64, 'grf_mode': '256'}, num_stages=s, num_warps=4)
        for s in [2]
    ]
    return configs


@triton.jit
def scaled_mm_kernel_td(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am: tl.int64,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    # offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # masks_bn = offsets_bn < N

    # offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    # offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    # offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = tl.arange(0, BLOCK_SIZE_SCALE_A) + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = tl.arange(0, BLOCK_SIZE_SCALE_B) + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    masks_scale_bn = offsets_scale_bn < N

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))

    # a_ptrs = a_ptr + offsets_a
    # b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    off_k = 0
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # masks_k = offsets_k < K
        # masks_a = masks_am[:, None] & masks_k[None, :]
        # a = tl.load(a_ptrs, mask=masks_a)

        # masks_b = masks_k[:, None] & masks_bn[None, :]
        # b = tl.load(b_ptrs, mask=masks_b)

        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        # accumulator += tl.dot(a, b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
        off_k += BLOCK_SIZE_K

        # offsets_k += BLOCK_SIZE_K
        # a_ptrs += BLOCK_SIZE_K * stride_ak
        # b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    # c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # tl.store(c_ptrs, c, mask=c_mask)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


# input  - [M, K]
# weight - [K, N]
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
def triton_scaled_mm_td(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)
    else:
        raise NotImplementedError('Only heuristic-based tile size selection is supported currently.')

    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel_td[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
    )
    return result


torch.set_default_device('xpu')
device = 'xpu'


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation using float32 for stability"""
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a.to(torch.float32) * out * scale_b.to(torch.float32).T
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)


def _make_inputs(M, K, N, in_dtype):
    if in_dtype == torch.int8:
        a = torch.randint(-8, 8, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-8, 8, (K, N), dtype=in_dtype, device=device)
    else:  # fp8
        # Adding zero help with nan for some reason, without it there will be some accidental nans
        a = (0 + torch.clamp(0.5 * torch.randn((M, K), dtype=torch.float16, device=device), -0.25, 0.25)).to(in_dtype)
        b = 0.5 * torch.randn((K, N), dtype=torch.float16, device=device)
        b = torch.clamp(b, -0.25, 0.25)
        # Adding zero help with nan for some reason, without it there will be some accidental nans
        b = (0 + b).to(in_dtype)
    return a, b


X_VALS = sum([[  #
    # [M, 128, 128],
    [M, 1024, 4096], [M, 4096, 4096], [M, 4096, 4096 * 4]
] for M in [1, 8, 128, 1024, 4096]], [])


def get_scaled_mm_benchmark(
    providers_filter: Optional[list[str]] = None,
    fp8=False,
    plot_name: str = 'scaled_mm_benchmark',
):
    supported_providers = {
        'triton': 'triton',
        'triton-td': 'triton-td',
        'pytorch': 'pytorch-deqmm',
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=X_VALS,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('red', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name=plot_name,
            args={},
        ))
    def benchmark(M, N, K, provider, with_bias=False):
        torch.manual_seed(10)
        n_warmup = 600

        quantiles = [0.5, 0, 1.0]

        if fp8:
            in_dtype, out_dtype = torch.float8_e4m3fn, torch.float32
        else:
            in_dtype, out_dtype = torch.int8, torch.bfloat16

        x, weight = _make_inputs(M, K, N, in_dtype)
        scale_a = 0.1 + 0.05 * torch.rand((M, 1), dtype=torch.float32, device=device)
        scale_b = 0.1 + 0.05 * torch.rand((N, 1), dtype=torch.float32, device=device)
        bias = (0.01 * torch.randn((M, N), dtype=out_dtype, device=device) if with_bias else None)

        def torch_fn():
            return torch_scaled_mm(x, weight, scale_a, scale_b, bias)

        # Use relaxed tolerances
        rtol = 0.15 if in_dtype == torch.int8 else 0.25
        atol = 0.1 if in_dtype == torch.int8 else 0.15

        if provider == 'pytorch':
            # PyTorch reference implementation using native_batched_masked_quant_matmul
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                torch_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider in ('triton', 'triton-td'):
            invoke_kernel = triton_scaled_mm if provider == 'triton' else triton_scaled_mm_td

            def triton_fn():
                return invoke_kernel(x, weight, scale_a, scale_b, out_dtype, bias)

            benchmark_suite.assert_close(triton_fn, torch_fn, atol=atol, rtol=rtol, err_msg='triton to torch')

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        def gbps(ms):
            total_bytes = in_dtype.itemsize * (M * K + K * N) + out_dtype.itemsize * M * N
            return total_bytes * (1e-9) / (ms * 1e-3)

        def tflops(ms):
            total_flops = M * N * K * 2
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    _benchmark_mm = get_scaled_mm_benchmark(fp8=(os.getenv('FP8', '0') == '1'), )
    _benchmark_mm.run(show_plots=False, print_data=True)
