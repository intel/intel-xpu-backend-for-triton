import torch

import triton_kernels_benchmark as benchmark_suit
from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul
from sglang.test.test_block_fp8 import native_w8a8_block_fp8_matmul

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def has_enough_memory(x_val):
    # x_val: (B, M, N, K)
    B, M, N, K = x_val
    # a: (B, M, K) float8_e4m3
    # b: (B, N, K) float8_e4m3
    # c: (B, M, N) bfloat16
    # pytorch reference: (B, M, N) float32
    required_memory = B * M * K * 1 + B * N * K * 1 + B * M * N * 2 * 2
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


X_VALS = [[1024 * i, 1024 * i, 1024 * i] for i in [1, 2, 4, 8]] + [
    [1, 13824, 5120],
    [4, 12288, 4096],
    [512, 8192, 8192],
    [512, 8192, 32768],
    [512, 32768, 8192],
    [1024, 8192, 16384],
    [1024, 8192, 28672],
    [3072, 3072, 4096],
    [4096, 8192, 16384],
    [8192, 1024, 16384],
    [8192, 4096, 16384],
    [16384, 1024, 8192],
    [16384, 4096, 8192],
    [16384, 8192, 1024],
    [16384, 8192, 4096],
    [32768, 128, 4096],
    [32768, 4096, 128],
    [4096, 128, 4096],
    [8, 128, 16384],
    [8, 16384, 128],
]

X_VALS = [x_val for x_val in X_VALS if has_enough_memory(x_val)]


# Benchmark Performance
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["M", "N", "K"],
        # different possible values for `x_name`
        x_vals=X_VALS,
        line_arg="provider",
        # argument name whose value corresponds to a different line in the plot
        line_vals=["triton"],
        # label name for the lines
        line_names=["Triton"],
        # line styles
        ylabel=["GB/s", "TFlops"],  # label name for the y-axis
        plot_name="sglang-fp8-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, M, N, K, provider):
    torch.manual_seed(0)

    block_size = [128, 128]
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="xpu") - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="xpu") - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device="xpu") * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="xpu") * factor_for_scale

    quantiles = [0.5, 0.0, 1.0]

    if provider == "triton":
        triton_fn = lambda: w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size)
        torch_fn = lambda: native_w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size)
        benchmark_suit.assert_close(triton_fn, torch_fn, atol=3e-4, rtol=1e-2, err_msg="triton to torch")
        _, min_ms, max_ms, mean_ms, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10,
                                                                 quantiles=quantiles)

    else:
        raise NotImplementedError(f"Unsupported provider {provider}")

    tflops = lambda ms: 2 * B * M * N * K * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: B * ((M * K + K * N) + 2.0 * (M * N)) * (1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)
