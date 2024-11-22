import torch
import triton
import triton.language as tl
from functools import partial
import tempfile

device = 'xpu'
backend = getattr(torch, device)


def compute_time(
    fn,
    warmup=1,
    rep=5,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
):
    #assert return_mode in ["min", "max", "mean", "median"]

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
    mean_ms = torch.mean(times).item()
    max_ms = torch.max(times).item()
    min_ms = torch.min(times).item()

    return (mean_ms, max_ms, min_ms)

ir = f"""
#blocked = #triton_gpu.blocked<{{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}}>
#blocked1 = #triton_gpu.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 32], order = [0, 1]}}>
#loc = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":96:0)
#mma = #triton_intel_gpu.dpas<{{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [8, 2], A = [64, 16], B = [16, 32], C = [64, 32]}}>
module attributes {{"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block}} {{
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}} loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":96:0), %arg1: !tt.ptr<f16> {{tt.divisibility = 16 : i32}} loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":96:0), %arg2: !tt.ptr<f16> {{tt.divisibility = 16 : i32}} loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":96:0)) attributes {{noinline = false}} {{
    %c4_i32 = arith.constant 4 : i32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc1)
    %c5120_i64 = arith.constant 5120 : i64 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c4096_i64 = arith.constant 4096 : i64 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c5120_i32 = arith.constant 5120 : i32 loc(#loc1)
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.divsi %0, %c64_i32 : i32 loc(#loc3)
    %2 = arith.muli %1, %c4_i32 : i32 loc(#loc4)
    %3 = arith.subi %c4_i32, %2 : i32 loc(#loc5)
    %4 = arith.minsi %3, %c4_i32 : i32 loc(#loc6)
    %5 = arith.remsi %0, %4 : i32 loc(#loc7)
    %6 = arith.addi %2, %5 : i32 loc(#loc8)
    %7 = arith.remsi %0, %c64_i32 : i32 loc(#loc9)
    %8 = arith.divsi %7, %4 : i32 loc(#loc10)
    %9 = arith.muli %6, %c256_i32 : i32 loc(#loc11)
    %10 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c5120_i64], [%c5120_i64, %c1_i64], [%9, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<256x32xf16, #triton_gpu.dot_op<{{opIdx = 0, parent = #mma, kWidth = 2}}>>> loc(#loc12)
    %11 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c5120_i64], [%c5120_i64, %c1_i64], [%9, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<256x32xf16, #blocked>> loc(#loc12)
    %12 = arith.muli %8, %c256_i32 : i32 loc(#loc13)
    %13 = tt.make_tensor_ptr %arg1, [%c5120_i64, %c4096_i64], [%c1_i64, %c5120_i64], [%c0_i32, %12] {{order = array<i32: 1, 0>}} : <tensor<32x256xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>> loc(#loc14)
    %14 = tt.make_tensor_ptr %arg1, [%c5120_i64, %c4096_i64], [%c1_i64, %c5120_i64], [%c0_i32, %12] {{order = array<i32: 1, 0>}} : <tensor<32x256xf16, #blocked1>> loc(#loc14)
    triton_intel_gpu.prefetch %11 {{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, triton_intel_gpu.block_io = "row_major"}} : !tt.ptr<tensor<256x32xf16, #blocked>> loc(#loc15)
    triton_intel_gpu.prefetch %14 {{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, triton_intel_gpu.block_io = "column_major"}} : !tt.ptr<tensor<32x256xf16, #blocked1>> loc(#loc16)
    %15:5 = scf.for %arg3 = %c0_i32 to %c5120_i32 step %c32_i32 iter_args(%arg4 = %cst, %arg5 = %11, %arg6 = %14, %arg7 = %10, %arg8 = %13) -> (tensor<256x256xf32, #mma>, !tt.ptr<tensor<256x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{{opIdx = 0, parent = #mma, kWidth = 2}}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>>)  : i32 {{
      %18 = tt.advance %arg7, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #triton_gpu.dot_op<{{opIdx = 0, parent = #mma, kWidth = 2}}>>> loc(#loc18)
      %19 = tt.advance %arg5, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #blocked>> loc(#loc18)
      %20 = tt.advance %arg8, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>> loc(#loc19)
      %21 = tt.advance %arg6, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #blocked1>> loc(#loc19)
      triton_intel_gpu.prefetch %19 {{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, triton_intel_gpu.block_io = "row_major"}} : !tt.ptr<tensor<256x32xf16, #blocked>> loc(#loc15)
      triton_intel_gpu.prefetch %21 {{boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>, triton_intel_gpu.block_io = "column_major"}} : !tt.ptr<tensor<32x256xf16, #blocked1>> loc(#loc16)
      %22 = tt.load %arg7 {{boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"}} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{{opIdx = 0, parent = #mma, kWidth = 2}}>>> loc(#loc15)
      %23 = tt.load %arg8 {{boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "column_major"}} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>> loc(#loc16)
      %24 = tt.dot %22, %23, %arg4, inputPrecision = tf32 : tensor<256x32xf16, #triton_gpu.dot_op<{{opIdx = 0, parent = #mma, kWidth = 2}}>> * tensor<32x256xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>> -> tensor<256x256xf32, #mma> loc(#loc20)
      scf.yield %24, %19, %21, %18, %20 : tensor<256x256xf32, #mma>, !tt.ptr<tensor<256x32xf16, #blocked>>, !tt.ptr<tensor<32x256xf16, #blocked1>>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{{opIdx = 0, parent = #mma, kWidth = 2}}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>> loc(#loc17)
    }} loc(#loc17)
    %16 = tt.make_tensor_ptr %arg2, [%c1024_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %12] {{order = array<i32: 1, 0>}} : <tensor<256x256xf16, #mma>> loc(#loc21)
    %17 = arith.truncf %15#0 : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma> loc(#loc22)
    tt.store %16, %17 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<256x256xf16, #mma>> loc(#loc23)
    tt.return loc(#loc24)
  }} loc(#loc)
}} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":117:24)
#loc3 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":121:22)
#loc4 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":122:29)
#loc5 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":123:35)
#loc6 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":123:48)
#loc7 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":124:33)
#loc8 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":124:27)
#loc9 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":125:19)
#loc10 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":125:40)
#loc11 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":133:53)
#loc12 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":134:36)
#loc13 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":136:56)
#loc14 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":137:36)
#loc15 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":151:20)
#loc16 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":152:20)
#loc17 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":145:25)
#loc18 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":157:46)
#loc19 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":158:46)
#loc20 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":154:33)
#loc21 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":170:78)
#loc22 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":171:31)
#loc23 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":171:26)
#loc24 = loc("/home/abaden/Projects/intel-xpu-backend-for-triton/test_local.py":171:4)

"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
    f.write(ir)
    f.flush()
    matmul_kernel_with_block_pointers = triton.compile(f.name, options={"grf_mode": "large"})

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
    #grid = lambda META: (triton.cdiv(M, 256) * triton.cdiv(N, 256), )
    grid = (64, 1, 1)

    matmul_kernel_with_block_pointers[grid](
        X, Y, Z
    )

    return Z


M = 1024
K = 5120
N = 4096
dtype  = torch.float16
torch.manual_seed(0)

AxB = False
AxBT = True
ATxB = False
ATxBT = False

tflops = lambda ms: 2 * M * N * K * (1e-12) / (ms * 1e-3)
gbps = lambda ms: (2 * (M * K + K * N) + 4.0 * (M * N)) * (1e-9) / (ms * 1e-3)

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
    #print(f"Time for torch: {t_tor} (mean / max / min ms)")
    #print(f"Time for triton: {t_tri} (mean / max / min ms)")

    print(f"AxB      \ttorch\t\t\ttriton")
    print(f"gbps mean\t{gbps(t_tor[0])}\t{gbps(t_tri[0])}")
    print(f"gbps min\t{gbps(t_tor[1])}\t{gbps(t_tri[1])}")
    print(f"gbps max\t{gbps(t_tor[2])}\t{gbps(t_tri[2])}")
    print(f"tflops mean\t{tflops(t_tor[0])}\t{tflops(t_tri[0])}")
    print(f"tflops min\t{tflops(t_tor[1])}\t{tflops(t_tri[1])}")
    print(f"tflops max\t{tflops(t_tor[2])}\t{tflops(t_tri[2])}")
    



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
    # print(f"Time for torch: {t_tor} (mean / max / min ms)")
    # print(f"Time for triton: {t_tri} (mean / max / min ms)")

    print(f"AxBT      \ttorch\t\t\ttriton")
    print(f"gbps mean\t{gbps(t_tor[0])}\t{gbps(t_tri[0])}")
    print(f"gbps min\t{gbps(t_tor[1])}\t{gbps(t_tri[1])}")
    print(f"gbps max\t{gbps(t_tor[2])}\t{gbps(t_tri[2])}")
    print(f"tflops mean\t{tflops(t_tor[0])}\t{tflops(t_tri[0])}")
    print(f"tflops min\t{tflops(t_tor[1])}\t{tflops(t_tri[1])}")
    print(f"tflops max\t{tflops(t_tor[2])}\t{tflops(t_tri[2])}")
    

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
    # print(f"Time for torch: {t_tor} (mean / max / min ms)")
    # print(f"Time for triton: {t_tri} (mean / max / min ms)")
    print(f"ATxB      \ttorch\t\t\ttriton")
    print(f"gbps mean\t{gbps(t_tor[0])}\t{gbps(t_tri[0])}")
    print(f"gbps min\t{gbps(t_tor[1])}\t{gbps(t_tri[1])}")
    print(f"gbps max\t{gbps(t_tor[2])}\t{gbps(t_tri[2])}")
    print(f"tflops mean\t{tflops(t_tor[0])}\t{tflops(t_tri[0])}")
    print(f"tflops min\t{tflops(t_tor[1])}\t{tflops(t_tri[1])}")
    print(f"tflops max\t{tflops(t_tor[2])}\t{tflops(t_tri[2])}")
    

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
    # print(f"Time for torch: {t_tor} (mean / max / min ms)")
    # print(f"Time for triton: {t_tri} (mean / max / min ms)")
    
    print(f"ATxBT      \ttorch\t\t\ttriton")
    print(f"gbps mean\t{gbps(t_tor[0])}\t{gbps(t_tri[0])}")
    print(f"gbps min\t{gbps(t_tor[1])}\t{gbps(t_tri[1])}")
    print(f"gbps max\t{gbps(t_tor[2])}\t{gbps(t_tri[2])}")
    print(f"tflops mean\t{tflops(t_tor[0])}\t{tflops(t_tri[0])}")
    print(f"tflops min\t{tflops(t_tor[1])}\t{tflops(t_tri[1])}")
    print(f"tflops max\t{tflops(t_tor[2])}\t{tflops(t_tri[2])}")
    