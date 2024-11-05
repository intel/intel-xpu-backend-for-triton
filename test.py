import torch
import triton
import triton.language as tl
from functools import partial

device = 'xpu'
backend = getattr(torch, device)


def compute_time(
    fn,
    warmup=1,
    rep=5,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
):
    assert return_mode in ["min", "max", "mean", "median"]

    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    backend.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device)

    # compute number of warmup and repeat

    start_event = [backend.Event(enable_timing=True) for i in range(rep)]
    end_event = [backend.Event(enable_timing=True) for i in range(rep)]
    # Warm-up
    for _ in range(warmup):
        fn()
    # Benchmark
    for i in range(rep):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                if hasattr(x, 'grad'):
                    x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    backend.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float
    )
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config(kwargs={'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=32),
        # triton.Config(kwargs={'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=32),
        # triton.Config(kwargs={'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=32),
        # triton.Config(kwargs={'BLOCK_SIZE_M':   8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=2, num_warps=32),
    ],
    key=['M', 'N', 'K'],)
@triton.jit
def matmul_kernel_with_block_pointers(
        # Pointers to matrices
        a_ptr, b_ptr, bias_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BIAS_REQD: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    #tl.device_print("pid", pid_m)

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(tl.float32)
    # add bias to accumulator
    if BIAS_REQD:
        offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + offs_yn, mask=offs_yn < N, other=0.0).to(tl.float32)
        c += bias[None, :]
    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c.to(tl.float16), boundary_check=(0, 1))


def triton_mm(X, Y, b=None, transpose_x=False, transpose_y=False):
    if transpose_x:
        K, M = X.shape
        Xstride0, Xstride1 = X.stride(1), X.stride(0)
    else:
        M, K = X.shape
        Xstride0, Xstride1 = X.stride(0), X.stride(1)
    if transpose_y:
        N, _ = Y.shape
        Wstride0, Wstride1 = Y.stride(1), Y.stride(0)
    else:
        _, N = Y.shape
        Wstride0, Wstride1 = Y.stride(0), Y.stride(1)
    # Allocates output.
    Z = torch.empty((M, N), device=X.device, dtype=X.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel_with_block_pointers[grid](
        X, Y, b, Z,
        M, N, K,
        Xstride0, Xstride1,
        Wstride0, Wstride1,
        Z.stride(0), Z.stride(1),
        BIAS_REQD=b is not None,
    )

    return Z


M = 1024
K = 5120
N = 4096
dtype  = torch.float16
torch.manual_seed(0)

AxB = True
AxBT = True
ATxB = True
ATxBT = True

if AxB:
    print('Compute A x B')
    X = torch.randn((M, K), device=device, dtype=dtype, requires_grad=False)
    Y = torch.randn((K, N), device=device, dtype=dtype, requires_grad=False)

    fn_tor = partial(torch.mm, X, Y)
    fn_tri = partial(triton_mm, X, Y)

    rtol = 1e-3
    result_tor = fn_tor()
    result_tri = fn_tri()
    if torch.allclose(result_tri, result_tor, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        exit("❌ Triton and Torch differ")

    t_tor = compute_time(fn_tor, warmup=5, rep=100)
    t_tri = compute_time(fn_tri, warmup=5, rep=100)
    print(f"Time for torch: {t_tor} ms")
    print(f"Time for triton: {t_tri} ms")


if AxBT:
    torch.manual_seed(0)
    print('Compute A x B.T')
    X = torch.randn((M, K), device=device, dtype=dtype, requires_grad=False)
    Y = torch.randn((N, K), device=device, dtype=dtype, requires_grad=False)

    fn_tor = partial(torch.mm, X, Y.T)
    fn_tri = partial(triton_mm, X, Y, transpose_y=True)

    rtol = 1e-3
    result_tor = fn_tor()
    result_tri = fn_tri()
    if torch.allclose(result_tri, result_tor, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        exit("❌ Triton and Torch differ")

    t_tor = compute_time(fn_tor, warmup=5, rep=100)
    t_tri = compute_time(fn_tri, warmup=5, rep=100)
    print(f"Time for torch: {t_tor} ms")
    print(f"Time for triton: {t_tri} ms")

if ATxB:
    torch.manual_seed(0)
    print('Compute A.T x B')
    X = torch.randn((K, M), device=device, dtype=dtype, requires_grad=False)
    Y = torch.randn((K, N), device=device, dtype=dtype, requires_grad=False)

    fn_tor = partial(torch.mm, X.T, Y)
    fn_tri = partial(triton_mm, X, Y, transpose_x=True)

    rtol = 1e-3
    result_tor = fn_tor()
    result_tri = fn_tri()
    if torch.allclose(result_tri, result_tor, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        exit("❌ Triton and Torch differ")

    t_tor = compute_time(fn_tor, warmup=5, rep=100)
    t_tri = compute_time(fn_tri, warmup=5, rep=100)
    print(f"Time for torch: {t_tor} ms")
    print(f"Time for triton: {t_tri} ms")

if ATxBT:
    torch.manual_seed(0)
    print('Compute A.T x B.T')
    X = torch.randn((K, M), device=device, dtype=dtype, requires_grad=False)
    Y = torch.randn((N, K), device=device, dtype=dtype, requires_grad=False)
    
    fn_tor = partial(torch.mm, X.T, Y.T)
    fn_tri = partial(triton_mm, X, Y, transpose_x=True, transpose_y=True)
    
    rtol = 1e-3
    result_tor = fn_tor()
    result_tri = fn_tri()
    if torch.allclose(result_tri, result_tor, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        exit("❌ Triton and Torch differ")

    t_tor = compute_time(fn_tor, warmup=5, rep=100)
    t_tri = compute_time(fn_tri, warmup=5, rep=100)
    print(f"Time for torch: {t_tor} ms")
    print(f"Time for triton: {t_tri} ms")
