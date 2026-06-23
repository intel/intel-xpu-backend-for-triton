import torch
import triton
import triton.language as tl

from triton_kernels_benchmark import Benchmark, do_bench, perf_report, assert_close

DEVICE = 'xpu'

# ---------------------------------------------------------------------------
# copy_1d
# ---------------------------------------------------------------------------


@triton.jit
def copy_1d_load_kernel(in_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)
    mask = x0 < N
    tmp = tl.load(in_ptr + x0, mask)
    tl.store(out_ptr + x0, tmp, mask)


@triton.jit
def copy_1d_desc_kernel(in_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    in_desc = tl.make_tensor_descriptor(in_ptr, shape=[N], strides=[1], block_shape=[XBLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[XBLOCK])
    tmp = in_desc.load([xoffset])
    out_desc.store([xoffset], tmp)


@perf_report(
    Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(20, 27)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='copy-1d',
        args={},
    ))
def benchmark_copy_1d(N, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    x = torch.randn(N, dtype=dtype, device=DEVICE)
    y = torch.empty(N, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(N, dtype=dtype, device=DEVICE)

    XBLOCK = 1024
    grid = (triton.cdiv(N, XBLOCK), )

    assert_close(
        lambda: (copy_1d_load_kernel[grid](x, y_ref, N, XBLOCK), y_ref)[1],
        lambda: (copy_1d_desc_kernel[grid](x, y, N, XBLOCK), y)[1],
    )

    load_fn = lambda: copy_1d_load_kernel[grid](x, y, N, XBLOCK)
    desc_fn = lambda: copy_1d_desc_kernel[grid](x, y, N, XBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        return 2 * N * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# gelu_1d
# ---------------------------------------------------------------------------


@triton.jit
def gelu_1d_load_kernel(in_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)
    mask = x0 < N
    tmp = tl.load(in_ptr + x0, mask).to(tl.float32)
    tmp = tmp * tl.sigmoid(1.702 * tmp)
    tl.store(out_ptr + x0, tmp.to(tl.float16), mask)


@triton.jit
def gelu_1d_desc_kernel(in_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    in_desc = tl.make_tensor_descriptor(in_ptr, shape=[N], strides=[1], block_shape=[XBLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[XBLOCK])
    tmp = in_desc.load([xoffset]).to(tl.float32)
    tmp = tmp * tl.sigmoid(1.702 * tmp)
    out_desc.store([xoffset], tmp.to(tl.float16))


@perf_report(
    Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(20, 27)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='gelu-1d',
        args={},
    ))
def benchmark_gelu_1d(N, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    x = torch.randn(N, dtype=dtype, device=DEVICE)
    y = torch.empty(N, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(N, dtype=dtype, device=DEVICE)

    XBLOCK = 1024
    grid = (triton.cdiv(N, XBLOCK), )

    assert_close(
        lambda: (gelu_1d_load_kernel[grid](x, y_ref, N, XBLOCK), y_ref)[1],
        lambda: (gelu_1d_desc_kernel[grid](x, y, N, XBLOCK), y)[1],
    )

    load_fn = lambda: gelu_1d_load_kernel[grid](x, y, N, XBLOCK)
    desc_fn = lambda: gelu_1d_desc_kernel[grid](x, y, N, XBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        return 2 * N * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# copy_2d
# ---------------------------------------------------------------------------


@triton.jit
def copy_2d_load_kernel(in_ptr, out_ptr, M, N, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK
    y0 = yoffset + tl.arange(0, YBLOCK)
    x0 = xoffset + tl.arange(0, XBLOCK)
    ymask = y0[:, None] < M
    xmask = x0[None, :] < N
    tmp = tl.load(in_ptr + y0[:, None] * N + x0[None, :], ymask & xmask)
    tl.store(out_ptr + y0[:, None] * N + x0[None, :], tmp, ymask & xmask)


@triton.jit
def copy_2d_desc_kernel(in_ptr, out_ptr, M, N, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    yoffset = tl.program_id(1) * YBLOCK
    xoffset = tl.program_id(0) * XBLOCK
    in_desc = tl.make_tensor_descriptor(in_ptr, shape=[M, N], strides=[N, 1], block_shape=[YBLOCK, XBLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1], block_shape=[YBLOCK, XBLOCK])
    tmp = in_desc.load([yoffset, xoffset])
    out_desc.store([yoffset, xoffset], tmp)


@perf_report(
    Benchmark(
        x_names=['M', 'N'],
        x_vals=[(1024, 1024), (2048, 2048), (4096, 4096)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='copy-2d',
        args={},
    ))
def benchmark_copy_2d(M, N, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=dtype, device=DEVICE)
    y = torch.empty(M, N, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(M, N, dtype=dtype, device=DEVICE)

    YBLOCK = 32
    XBLOCK = 32

    grid = (triton.cdiv(N, XBLOCK), triton.cdiv(M, YBLOCK))

    assert_close(
        lambda: (copy_2d_load_kernel[grid](x, y_ref, M, N, YBLOCK, XBLOCK), y_ref)[1],
        lambda: (copy_2d_desc_kernel[grid](x, y, M, N, YBLOCK, XBLOCK), y)[1],
    )

    load_fn = lambda: copy_2d_load_kernel[grid](x, y, M, N, YBLOCK, XBLOCK)
    desc_fn = lambda: copy_2d_desc_kernel[grid](x, y, M, N, YBLOCK, XBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    total_elements = M * N

    def gbps(ms):
        return 2 * total_elements * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# permute_3d
# ---------------------------------------------------------------------------


@triton.jit
def permute_3d_load_kernel(in_ptr, out_ptr, D0, D1, D2, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
                           ZBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK
    zoffset = tl.program_id(2) * ZBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)
    y0 = yoffset + tl.arange(0, YBLOCK)
    z0 = zoffset + tl.arange(0, ZBLOCK)
    xmask = x0 < D0
    ymask = y0 < D1
    zmask = z0 < D2
    tmp = tl.load(in_ptr + x0[:, None, None] * (D1 * D2) + y0[None, :, None] * D2 + z0[None, None, :],
                  xmask[:, None, None] & ymask[None, :, None] & zmask[None, None, :])
    tl.store(out_ptr + y0[:, None, None] * (D2 * D0) + z0[None, :, None] * D0 + x0[None, None, :], tmp.trans(1, 2, 0),
             ymask[:, None, None] & zmask[None, :, None] & xmask[None, None, :])


@triton.jit
def permute_3d_desc_kernel(in_ptr, out_ptr, D0, D1, D2, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
                           ZBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    yoffset = tl.program_id(1) * YBLOCK
    zoffset = tl.program_id(2) * ZBLOCK
    in_desc = tl.make_tensor_descriptor(in_ptr, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1],
                                        block_shape=[XBLOCK, YBLOCK, ZBLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[D1, D2, D0], strides=[D2 * D0, D0, 1],
                                         block_shape=[YBLOCK, ZBLOCK, XBLOCK])
    tmp = in_desc.load([xoffset, yoffset, zoffset])
    out_desc.store([yoffset, zoffset, xoffset], tl.trans(tmp, [1, 2, 0]))


@perf_report(
    Benchmark(
        x_names=['D0', 'D1', 'D2'],
        x_vals=[(28, 64, 28), (56, 128, 56)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='permute-3d',
        args={},
    ))
def benchmark_permute_3d(D0, D1, D2, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    x = torch.randn(D0, D1, D2, dtype=dtype, device=DEVICE)
    y = torch.empty(D1, D2, D0, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(D1, D2, D0, dtype=dtype, device=DEVICE)

    XBLOCK = 8
    YBLOCK = 8
    ZBLOCK = 8

    grid = (triton.cdiv(D0, XBLOCK), triton.cdiv(D1, YBLOCK), triton.cdiv(D2, ZBLOCK))

    assert_close(
        lambda: (permute_3d_load_kernel[grid](x, y_ref, D0, D1, D2, XBLOCK, YBLOCK, ZBLOCK), y_ref)[1],
        lambda: (permute_3d_desc_kernel[grid](x, y, D0, D1, D2, XBLOCK, YBLOCK, ZBLOCK), y)[1],
    )

    load_fn = lambda: permute_3d_load_kernel[grid](x, y, D0, D1, D2, XBLOCK, YBLOCK, ZBLOCK)
    desc_fn = lambda: permute_3d_desc_kernel[grid](x, y, D0, D1, D2, XBLOCK, YBLOCK, ZBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    total_elements = D0 * D1 * D2

    def gbps(ms):
        return 2 * total_elements * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# row_sum_2d
# ---------------------------------------------------------------------------


@triton.jit
def row_sum_2d_load_kernel(in_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    row = tl.program_id(0)
    _acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, N, XBLOCK):
        cols = off + tl.arange(0, XBLOCK)
        _acc += tl.load(in_ptr + row * N + cols, cols < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(_acc, axis=0).to(tl.float16))


@triton.jit
def row_sum_2d_desc_kernel(in_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    row = tl.program_id(0)
    in_desc = tl.make_tensor_descriptor(in_ptr + row * N, shape=[N], strides=[1], block_shape=[XBLOCK])
    _acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, N, XBLOCK):
        _acc += in_desc.load([off]).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(_acc, axis=0).to(tl.float16))


@perf_report(
    Benchmark(
        x_names=['M', 'N'],
        x_vals=[(1024, 1024), (1024, 4096), (1024, 16384), (4096, 4096)],  # N must be multiple of XBLOCK=1024
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='row-sum-2d',
        args={},
    ))
def benchmark_row_sum_2d(M, N, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    x = torch.randn(M, N, dtype=dtype, device=DEVICE)
    y = torch.empty(M, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(M, dtype=dtype, device=DEVICE)

    XBLOCK = 1024
    grid = (M, )

    assert_close(
        lambda: (row_sum_2d_load_kernel[grid](x, y_ref, N, XBLOCK), y_ref)[1],
        lambda: (row_sum_2d_desc_kernel[grid](x, y, N, XBLOCK), y)[1],
    )

    load_fn = lambda: row_sum_2d_load_kernel[grid](x, y, N, XBLOCK)
    desc_fn = lambda: row_sum_2d_desc_kernel[grid](x, y, N, XBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        return (M * N + M) * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# layer_norm_2d
# ---------------------------------------------------------------------------


@triton.jit
def layer_norm_2d_load_kernel(in_ptr, out_ptr, N, eps, XBLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * N
    # pass 1: mean
    _mean = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, N, XBLOCK):
        cols = off + tl.arange(0, XBLOCK)
        _mean += tl.load(in_ptr + row_start + cols, cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(_mean, axis=0) / N
    # pass 2: variance
    _var = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, N, XBLOCK):
        cols = off + tl.arange(0, XBLOCK)
        x = tl.load(in_ptr + row_start + cols, cols < N, other=0.0).to(tl.float32) - mean
        _var += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(_var, axis=0) / N + eps)
    # pass 3: normalize and store
    for off in range(0, N, XBLOCK):
        cols = off + tl.arange(0, XBLOCK)
        mask = cols < N
        x = tl.load(in_ptr + row_start + cols, mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row_start + cols, ((x - mean) * rstd).to(tl.float16), mask)


@triton.jit
def layer_norm_2d_desc_kernel(in_ptr, out_ptr, N, eps, XBLOCK: tl.constexpr):
    row = tl.program_id(0)
    in_desc = tl.make_tensor_descriptor(in_ptr + row * N, shape=[N], strides=[1], block_shape=[XBLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr + row * N, shape=[N], strides=[1], block_shape=[XBLOCK])
    # pass 1: mean
    _mean = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, N, XBLOCK):
        _mean += in_desc.load([off]).to(tl.float32)
    mean = tl.sum(_mean, axis=0) / N
    # pass 2: variance
    _var = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, N, XBLOCK):
        x = in_desc.load([off]).to(tl.float32) - mean
        _var += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(_var, axis=0) / N + eps)
    # pass 3: normalize and store
    for off in range(0, N, XBLOCK):
        x = in_desc.load([off]).to(tl.float32)
        out_desc.store([off], ((x - mean) * rstd).to(tl.float16))


@perf_report(
    Benchmark(
        x_names=['M', 'N'],
        x_vals=[(1024, 1024), (1024, 4096), (1024, 16384), (4096, 4096)],  # N must be multiple of XBLOCK=1024
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='layer-norm-2d',
        args={},
    ))
def benchmark_layer_norm_2d(M, N, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)
    eps = 1e-5

    x = torch.randn(M, N, dtype=dtype, device=DEVICE)
    y = torch.empty(M, N, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(M, N, dtype=dtype, device=DEVICE)

    XBLOCK = 1024
    grid = (M, )

    assert_close(
        lambda: (layer_norm_2d_load_kernel[grid](x, y_ref, N, eps, XBLOCK), y_ref)[1],
        lambda: (layer_norm_2d_desc_kernel[grid](x, y, N, eps, XBLOCK), y)[1],
    )

    load_fn = lambda: layer_norm_2d_load_kernel[grid](x, y, N, eps, XBLOCK)
    desc_fn = lambda: layer_norm_2d_desc_kernel[grid](x, y, N, eps, XBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        # 3 reads (mean pass, var pass, normalize pass) + 1 write
        return 4 * M * N * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


# ---------------------------------------------------------------------------
# matmul_desc
# ---------------------------------------------------------------------------


@triton.jit
def matmul_ptr_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.float16))


@triton.jit
def matmul_desc_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a_desc.load([pid_m * BM, k]), b_desc.load([k, pid_n * BN]), acc)
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.float16))


@perf_report(
    Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)],
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['TFlops'],
        plot_name='matmul-desc',
        args={},
    ))
def benchmark_matmul_desc(M, N, K, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    torch.manual_seed(0)

    a = torch.randn(M, K, dtype=dtype, device=DEVICE)
    b = torch.randn(K, N, dtype=dtype, device=DEVICE)
    c = torch.empty(M, N, dtype=dtype, device=DEVICE)
    c_ref = torch.empty(M, N, dtype=dtype, device=DEVICE)

    BM = BN = BK = 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    assert_close(
        lambda: (matmul_ptr_kernel[grid](a, b, c_ref, M, N, K, BM, BN, BK), c_ref)[1],
        lambda: (matmul_desc_kernel[grid](a, b, c, M, N, K, BM, BN, BK), c)[1],
        atol=1e-2,
        rtol=1e-2,
    )

    load_fn = lambda: matmul_ptr_kernel[grid](a, b, c, M, N, K, BM, BN, BK)
    desc_fn = lambda: matmul_desc_kernel[grid](a, b, c, M, N, K, BM, BN, BK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def tflops(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


# ---------------------------------------------------------------------------
# cat_reduce_2d  (covers #7103, #7104)
# ---------------------------------------------------------------------------
# Pattern: reduce rows of a conceptual concatenation of two input tensors
# [A: M×(N/2), B: M×(N/2)] → out: M
# Tests multi-source descriptor usage in a reduction loop.


@triton.jit
def cat_reduce_2d_load_kernel(a_ptr, b_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    row = tl.program_id(0)
    half_n = N // 2
    _acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, half_n, XBLOCK):
        cols = off + tl.arange(0, XBLOCK)
        _acc += tl.load(a_ptr + row * half_n + cols, cols < half_n, other=0.0).to(tl.float32)
    for off in range(0, half_n, XBLOCK):
        cols = off + tl.arange(0, XBLOCK)
        _acc += tl.load(b_ptr + row * half_n + cols, cols < half_n, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(_acc, axis=0).to(tl.float16))


@triton.jit
def cat_reduce_2d_desc_kernel(a_ptr, b_ptr, out_ptr, N, XBLOCK: tl.constexpr):
    row = tl.program_id(0)
    half_n = N // 2
    a_desc = tl.make_tensor_descriptor(a_ptr + row * half_n, shape=[half_n], strides=[1], block_shape=[XBLOCK])
    b_desc = tl.make_tensor_descriptor(b_ptr + row * half_n, shape=[half_n], strides=[1], block_shape=[XBLOCK])
    _acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for off in range(0, half_n, XBLOCK):
        _acc += a_desc.load([off]).to(tl.float32)
    for off in range(0, half_n, XBLOCK):
        _acc += b_desc.load([off]).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(_acc, axis=0).to(tl.float16))


@perf_report(
    Benchmark(
        x_names=['M', 'N'],
        x_vals=[(1024, 2048), (1024, 8192), (1024, 32768), (4096, 8192)],  # N must be multiple of 2*XBLOCK=2048
        line_arg='provider',
        line_vals=['load', 'descriptor'],
        line_names=['tl.load', 'Tensor Descriptor'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='cat-reduce-2d',
        args={},
    ))
def benchmark_cat_reduce_2d(M, N, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    half_n = N // 2
    a = torch.randn(M, half_n, dtype=dtype, device=DEVICE)
    b = torch.randn(M, half_n, dtype=dtype, device=DEVICE)
    y = torch.empty(M, dtype=dtype, device=DEVICE)
    y_ref = torch.empty(M, dtype=dtype, device=DEVICE)

    XBLOCK = 1024
    grid = (M, )

    assert_close(
        lambda: (cat_reduce_2d_load_kernel[grid](a, b, y_ref, N, XBLOCK), y_ref)[1],
        lambda: (cat_reduce_2d_desc_kernel[grid](a, b, y, N, XBLOCK), y)[1],
    )

    load_fn = lambda: cat_reduce_2d_load_kernel[grid](a, b, y, N, XBLOCK)
    desc_fn = lambda: cat_reduce_2d_desc_kernel[grid](a, b, y, N, XBLOCK)

    if provider == 'load':
        fn = load_fn
    else:
        fn = desc_fn

    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        # read N elements (N/2 from A + N/2 from B) + write M elements
        return (M * N + M) * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


def _gpu_warmup():
    """Run a simple kernel a few times to bring the GPU to steady-state frequency."""
    n = 1024 * 1024
    x = torch.randn(n, dtype=torch.float16, device=DEVICE)
    y = torch.empty_like(x)
    xblock = 1024
    grid = (triton.cdiv(n, xblock), )
    for _ in range(5):
        copy_1d_load_kernel[grid](x, y, n, xblock)
    torch.xpu.synchronize()


def run_benchmarks():
    _gpu_warmup()
    benchmark_copy_1d.run(show_plots=False, print_data=True)
    benchmark_gelu_1d.run(show_plots=False, print_data=True)
    benchmark_copy_2d.run(show_plots=False, print_data=True)
    benchmark_permute_3d.run(show_plots=False, print_data=True)
    benchmark_row_sum_2d.run(show_plots=False, print_data=True)
    benchmark_layer_norm_2d.run(show_plots=False, print_data=True)
    benchmark_matmul_desc.run(show_plots=False, print_data=True)
    benchmark_cat_reduce_2d.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    run_benchmarks()
