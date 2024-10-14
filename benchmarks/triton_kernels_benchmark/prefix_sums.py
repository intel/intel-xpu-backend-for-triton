import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suit

if benchmark_suit.USE_IPEX_OPTION:
    import intel_extension_for_pytorch  # type: ignore # noqa: F401


@triton.jit
def scan_kernel(x_ptr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,  #
                AXIS: tl.constexpr):
    range_m = tl.arange(0, BLOCK_SIZE_M)
    range_n = tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(x_ptr + range_m[:, None] * BLOCK_SIZE_N + range_n[None, :])
    x = tl.cumsum(x, axis=AXIS)
    tl.store(x_ptr + range_m[:, None] * BLOCK_SIZE_N + range_n[None, :], x)


@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        x_names=['M', 'N', 'AXIS'],
        x_vals=[(m, n, a) for (m, n) in [  #
            (32, 16),  #
            (32, 32),  #
            (32, 64),  #
            (64, 32)
        ] for a in [  #
            0,  #
            1
        ]],
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel=['GB/s', 'TFlops'],
        plot_name='prefix-sums',
        args={},
    ))
def benchmark(M, N, AXIS, provider):
    quantiles = [0.5, 0.0, 1.0]
    x = torch.rand(M, N, device='xpu', dtype=torch.float32)

    if provider == 'triton':
        triton_fn = lambda: scan_kernel[(1, )](x, BLOCK_SIZE_M=M, BLOCK_SIZE_N=N, AXIS=AXIS)
        _, min_ms, max_ms, mean_ms, cv = benchmark_suit.do_bench(triton_fn, quantiles=quantiles,
                                                                 kernel_name='scan_kernel')
    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: (x.numel() * 1e-12) / (ms * 1e-3)
    gbps = lambda ms: (2 * x.numel() * x.element_size() * 1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(print_data=True)
