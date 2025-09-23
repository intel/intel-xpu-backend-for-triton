from typing import List, Optional

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite


@triton.jit
def scan_kernel(x_ptr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,  #
                AXIS: tl.constexpr):
    range_m = tl.arange(0, BLOCK_SIZE_M)
    range_n = tl.arange(0, BLOCK_SIZE_N)
    x = tl.load(x_ptr + range_m[:, None] * BLOCK_SIZE_N + range_n[None, :])
    x = tl.cumsum(x, axis=AXIS)
    tl.store(x_ptr + range_m[:, None] * BLOCK_SIZE_N + range_n[None, :], x)


def get_benchmark(providers_filter: Optional[List[str]] = None):
    """
    Returns a Mark object containing a Benchmark object constructed at runtime and parameterized by the provided option values.
    The benchmark can then be executed by calling the :code:`.run` method on the return value.
    """

    supported_providers = {
        "triton": "Triton",
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["M", "N", "AXIS"],
            x_vals=[(m, n, a)
                    for (m, n) in [(32, 16), (32, 32), (32, 64), (64, 32)]  #  #  #  #
                    for a in [0, 1]  #  #
                    ],
            line_arg="provider",
            line_vals=providers.keys(),
            line_names=providers.values(),
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel=["GB/s", "TFlops"],
            plot_name="prefix-sums",
            args={},
        ))
    def benchmark(M, N, AXIS, provider):
        n_warmup, n_repeat = benchmark_suite.get_benchmark_setup("prefix_sums")
        quantiles = [0.5, 0.0, 1.0]
        x = torch.rand(M, N, device="xpu", dtype=torch.float32)

        if provider == "triton":
            triton_fn = lambda: scan_kernel[(1, )](x, BLOCK_SIZE_M=M, BLOCK_SIZE_N=N, AXIS=AXIS)
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(triton_fn, quantiles=quantiles, n_warmup=n_warmup,
                                                                      n_repeat=n_repeat)
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        tflops = lambda ms: (x.numel() * 1e-12) / (ms * 1e-3)
        gbps = lambda ms: (2 * x.numel() * x.element_size() * 1e-9) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
