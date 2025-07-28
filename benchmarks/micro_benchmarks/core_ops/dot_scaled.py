import os
import sys

import torch
import triton
import triton.language as tl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from triton_kernels_benchmark import Benchmark, do_bench, perf_report  # pylint: disable=C0413


@triton.jit
def dot_scale_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                     type_b: tl.constexpr):
    DIV_FACTOR_A: tl.constexpr = 2 if type_a == 'e2m1' else 1
    DIV_FACTOR_B: tl.constexpr = 2 if type_b == 'e2m1' else 1
    PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
    PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0, PACKED_BLOCK_K_A)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0, BLOCK_N)[None, :] * stride_b1

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    c = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b)
    out_ptr = out + \
        tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + \
        tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr, c.to(tl.bfloat16))


def dot_scaled(M, N, K, x, y, z, scale_x, scale_y, type_a, type_b, num_warps):
    kernel_kwargs = {'num_warps': num_warps}
    dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b,
                            **kernel_kwargs)


# Benchmark Performance
@perf_report(
    Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M', 'K', 'N', 'col_a', 'col_b', 'rhs_scale', 'mxfp_type', 'normal_type'],
        x_vals=[(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type)
                for M, N, K in [(128, 128, 128)]
                for col_a, col_b in [(True, True), (False, False)]
                for rhs_scale in [True, False]
                for mxfp_type in ['e2m1', 'e4m3']
                for normal_type in ['bf16']],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['triton'],
        # label name for the lines
        line_names=['Triton'],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=('GB/s', ),  # label name for the y-axis
        plot_name='scaled-dot',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, col_a, col_b, rhs_scale, mxfp_type, normal_type, provider):

    device = 'xpu'
    num_warps = 4
    quantiles = [0.5, 0.0, 1.0]

    comp_dtype = torch.float16 if normal_type == 'fp16' else torch.bfloat16
    # The max exponent we use to initialize data in the x/y and associated scale tensor to avoid
    # overflow when scaling.
    comp_dtype_max_exp = 6 if normal_type == 'fp16' else 15

    torch.manual_seed(0)

    def make_arg(shape, ty, col_major=False):
        if col_major:
            shape = shape[:-2] + (shape[-1], shape[-2])
        if ty in ['fp16', 'bf16']:
            ret = torch.randn(shape, dtype=comp_dtype, device=device)
            # Clamp to avoid relative error issues
            ret.clamp_(-2**comp_dtype_max_exp, 2**comp_dtype_max_exp - 1)
        else:
            ret = torch.randint(256, shape, dtype=torch.uint8, device=device)
        if col_major:
            ret = ret.mT
        return ret

    type_a = normal_type if rhs_scale else mxfp_type
    type_b = mxfp_type if rhs_scale else normal_type

    DIV_FACTOR_A = 2 if type_a == 'e2m1' else 1
    DIV_FACTOR_B = 2 if type_b == 'e2m1' else 1
    x = make_arg((M, K // DIV_FACTOR_A), type_a, col_major=col_a)
    y = make_arg((K // DIV_FACTOR_B, N), type_b, col_major=col_b)

    min_scale, max_scale = (0, 142) if comp_dtype == torch.bfloat16 else (124, 131)
    scale_x = torch.randint(min_scale, max_scale + 1, (M, K // 32), dtype=torch.uint8, device=device)
    scale_y = torch.randint(min_scale, max_scale + 1, (N, K // 32), dtype=torch.uint8, device=device)

    def make_finite(x, dtype):
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if dtype not in ('e5m2', 'e4m3'):
            return x
        if dtype == 'e5m2' and comp_dtype == torch.float16:
            x = x & 0xB
        mask = 0x7C if dtype == 'e5m2' else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.uint8).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)
    z = x.new_empty((M, N), dtype=comp_dtype)
    if rhs_scale:
        scale_x = None
    else:
        scale_y = None

    if provider == 'triton':
        triton_fn = lambda: dot_scaled(M, N, K, x, y, z, scale_x, scale_y, type_a, type_b, num_warps)

        _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)
    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    def gbps(ms):

        def size_x(m, n, ty):
            if ty in ['e2m1']:
                return m * n // 2
            if ty in ['e4m3', 'e5m2']:
                return m * n
            if ty in ['fp16', 'bf16']:
                return m * n * 2
            raise NotImplementedError(f'Unsupported type {ty} for scaledot operand')

        tensor_size = size_x(M, K, type_a) + size_x(K, N, type_b)
        scale_size = (M * K // 32) if rhs_scale else (N * K // 32)
        return (tensor_size + scale_size + 4.0 * (M * N)) * (1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


def run_benchmarks():
    benchmark.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    run_benchmarks()
