"""
Block FP8 GEMM benchmark
============================
This benchmark uses the SGLang block-wise FP8 GEMM Triton kernel
(``w8a8_block_fp8_matmul_triton``) imported from the installed SGLang package,
validated against a native Torch reference.
"""

from typing import List, Optional

import torch

import triton_kernels_benchmark as benchmark_suite

# This supports both the current dispatcher name and older direct name.
try:
    from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton as w8a8_block_fp8_matmul
except ImportError:
    from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory

# Correctness is validated on a small shape only: native_w8a8_block_fp8_matmul is a slow
# tiled Python reference, so running it on every large (M, N, K) would make CI time out.
# N/K caps are multiples of the block size (128).
VALIDATION_MAX_M = 128
VALIDATION_MAX_N = 256
VALIDATION_MAX_K = 512


# For test
def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """This function performs matrix multiplication with block-wise quantization using native torch.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    """

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def has_enough_memory(x_val):
    # x_val: (M, N, K)
    M, N, K = x_val
    # a: (M, K) float8_e4m3
    # b: (N, K) float8_e4m3
    # c: (M, N) bfloat16
    # pytorch reference: (M, N) float32
    required_memory = M * K * 1 + N * K * 1 + M * N * 2 * 2
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


def _gen_inputs(M, N, K, block_size, factor_for_scale, fp8_min, fp8_max):
    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device='xpu') - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device='xpu') - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device='xpu') * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device='xpu') * factor_for_scale
    return A_fp8, B_fp8, As, Bs


def get_benchmark(providers_filter: Optional[List[str]] = None):
    """Returns a Mark object with the SGLang block FP8 GEMM benchmark."""
    supported_providers = {
        'triton': 'Triton',
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['M', 'N', 'K'],
            # different possible values for `x_name`
            x_vals=X_VALS,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
            plot_name='sglang-fp8-gemm-performance',
            # name for the plot. Used also as a file name for saving the plot.
            args={},
        ))
    def benchmark(M, N, K, provider):
        torch.manual_seed(0)

        block_size = [128, 128]
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        A_fp8, B_fp8, As, Bs = _gen_inputs(M, N, K, block_size, factor_for_scale, fp8_min, fp8_max)

        quantiles = [0.5, 0.0, 1.0]

        if provider == 'triton':
            triton_fn = lambda: w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size)
            # Validate on a small shape: the torch reference is a slow tiled Python
            # implementation, so running it on every large (M, N, K) would make CI time out.
            M_val, N_val, K_val = min(M, VALIDATION_MAX_M), min(N, VALIDATION_MAX_N), min(K, VALIDATION_MAX_K)
            A_val, B_val, As_val, Bs_val = _gen_inputs(M_val, N_val, K_val, block_size, factor_for_scale, fp8_min,
                                                       fp8_max)
            benchmark_suite.assert_close(lambda: w8a8_block_fp8_matmul(A_val, B_val, As_val, Bs_val, block_size),
                                         lambda: native_w8a8_block_fp8_matmul(A_val, B_val, As_val, Bs_val, block_size),
                                         atol=3e-4, rtol=1e-2, err_msg='triton to torch')
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(triton_fn, n_warmup=10, n_repeat=10,
                                                                      quantiles=quantiles)

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        tflops = lambda ms: 2 * M * N * K * (1e-12) / (ms * 1e-3)
        # A/B are fp8 (1 byte each), output C is bf16/fp16 (2 bytes).
        gbps = lambda ms: ((M * K + K * N) + 2.0 * (M * N)) * (1e-9) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    get_benchmark().run(show_plots=False, print_data=True)
