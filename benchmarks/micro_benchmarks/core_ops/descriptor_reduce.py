import torch
import triton
import triton.language as tl
from triton_kernels_benchmark import Benchmark, assert_close, do_bench, perf_report

DEVICE = 'xpu'

# ---------------------------------------------------------------------------
# reduce_segments_2d
# ---------------------------------------------------------------------------

# vLLM's reduce_segments: a [S, D] fp32 tile, weighted per row and summed over the rows.


@triton.jit
def reduce_segments_2d_load_kernel(out_ptr, x_ptr, m_ptr, nseg, H, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.arange(0, S)
    dim = tl.arange(0, DP)
    smask = seg < nseg
    m = tl.load(m_ptr + (t * H + h) * S + seg, mask=smask, other=float('-inf'))
    w = tl.exp(m - tl.max(m, axis=0))
    base = x_ptr + (t * H + h) * (S * DP)
    x = tl.load(base + seg[:, None] * DP + dim[None, :], mask=smask[:, None] & (dim < D)[None, :], other=0.0)
    tl.store(out_ptr + (t * H + h) * D + dim, tl.sum(x * w[:, None], axis=0).to(tl.bfloat16), mask=dim < D)


@triton.jit
def reduce_segments_2d_desc_kernel(out_ptr, x_ptr, m_ptr, nseg, H, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.arange(0, S)
    dim = tl.arange(0, DP)
    m = tl.load(m_ptr + (t * H + h) * S + seg, mask=seg < nseg, other=float('-inf'))
    w = tl.exp(m - tl.max(m, axis=0))
    base = x_ptr + (t * H + h) * (S * DP)
    x = tl.make_tensor_descriptor(base, shape=[nseg, D], strides=[DP, 1], block_shape=[S, DP]).load([0, 0])
    tl.store(out_ptr + (t * H + h) * D + dim, tl.sum(x * w[:, None], axis=0).to(tl.bfloat16), mask=dim < D)


@perf_report(
    Benchmark(
        x_names=['T', 'D'],
        x_vals=[(T, D) for T in (1, 8) for D in (64, 96, 128, 256)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='reduce-segments-2d',
        args={},
    ))
def benchmark_reduce_segments_2d(T, D, provider):
    quantiles = [0.5, 0.0, 1.0]
    H, S = 32, 16
    DP = triton.next_power_of_2(D)
    torch.manual_seed(0)

    x = torch.randn(T, H, S, DP, dtype=torch.float32, device=DEVICE)
    m = torch.randn(T, H, S, dtype=torch.float32, device=DEVICE)
    y = torch.empty(T, H, D, dtype=torch.bfloat16, device=DEVICE)
    y_ref = torch.empty_like(y)
    grid = (T, H)

    assert_close(
        lambda: (reduce_segments_2d_load_kernel[grid](y_ref, x, m, S, H, D, DP, S), y_ref)[1],
        lambda: (reduce_segments_2d_desc_kernel[grid](y, x, m, S, H, D, DP, S), y)[1],
    )

    load_fn = lambda: reduce_segments_2d_load_kernel[grid](y, x, m, S, H, D, DP, S)
    desc_fn = lambda: reduce_segments_2d_desc_kernel[grid](y, x, m, S, H, D, DP, S)
    fn = load_fn if provider == 'load' else desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        return (T * H * S * (D + 1) * 4 + T * H * D * 2) * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# loop_tile_sum_2d
# ---------------------------------------------------------------------------

# Each program sums K consecutive [16, 256] fp32 tiles: 1 KB row pitch, 256 MB in total.


@triton.jit
def loop_tile_sum_2d_load_kernel(out_ptr, x_ptr, N: tl.constexpr, BM: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * (K * BM) + tl.arange(0, BM)
    cols = tl.arange(0, N)
    acc = tl.zeros([N], dtype=tl.float32)
    for i in range(K):
        acc += tl.sum(tl.load(x_ptr + (rows[:, None] + i * BM) * N + cols[None, :]), axis=0)
    tl.store(out_ptr + pid * N + cols, acc)


@triton.jit
def loop_tile_sum_2d_desc_kernel(out_ptr, x_ptr, M, N: tl.constexpr, BM: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    row0 = pid * (K * BM)
    cols = tl.arange(0, N)
    desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, N])
    acc = tl.zeros([N], dtype=tl.float32)
    for i in range(K):
        acc += tl.sum(desc.load([row0 + i * BM, 0]), axis=0)
    tl.store(out_ptr + pid * N + cols, acc)


@perf_report(
    Benchmark(
        x_names=['K', 'P'],
        x_vals=[(K, 16384 // K) for K in (8, 16, 32, 64, 128)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='loop-tile-sum-2d',
        args={},
    ))
def benchmark_loop_tile_sum_2d(K, P, provider):
    quantiles = [0.5, 0.0, 1.0]
    N, BM = 256, 16
    M = P * K * BM
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
    y = torch.empty(P, N, dtype=torch.float32, device=DEVICE)
    y_ref = torch.empty_like(y)
    grid = (P, )

    assert_close(
        lambda: (loop_tile_sum_2d_load_kernel[grid](y_ref, x, N, BM, K), y_ref)[1],
        lambda: (loop_tile_sum_2d_desc_kernel[grid](y, x, M, N, BM, K), y)[1],
        atol=1e-3,
        rtol=1e-4,
    )

    load_fn = lambda: loop_tile_sum_2d_load_kernel[grid](y, x, N, BM, K)
    desc_fn = lambda: loop_tile_sum_2d_desc_kernel[grid](y, x, M, N, BM, K)
    fn = load_fn if provider == 'load' else desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        return (M * N + P * N) * 4 * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


def run_benchmarks():
    benchmark_reduce_segments_2d.run(show_plots=False, print_data=True)
    benchmark_loop_tile_sum_2d.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    run_benchmarks()
