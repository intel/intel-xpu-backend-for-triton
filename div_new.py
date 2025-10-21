import triton
import triton.language as tl
import torch
import os

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_DEBUG'] = '1'
os.environ['TRITON_INTERPRET'] = '0'
# Minimal debug output
os.environ['MLIR_ENABLE_DUMP'] = '1'
os.environ['LLVM_IR_ENABLE_DUMP'] = '1'

@triton.jit
def div_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)
    
    # Precise division - should use __imf_fdiv_rn (SLOW)
    result = tl.div_rn(x, y)
    
    tl.store(out_ptr + offsets, result, mask=mask)


def benchmark(kernel, x, y, out, n_elements, num_runs=100):
    """Simple benchmark function"""
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Warmup
    for _ in range(10):
        kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256)
    torch.xpu.synchronize()
    
    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(num_runs):
        kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256)
    torch.xpu.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000 / num_runs  # ms per run


if __name__ == '__main__':
    # Setup
    n_elements = 1024 * 1024
    x = torch.randn(n_elements, device='xpu', dtype=torch.float32)
    y = torch.randn(n_elements, device='xpu', dtype=torch.float32) + 1.0
    out = torch.empty_like(x)
    
    print("=" * 70)
    print("PRECISE DIVISION (tl.div_rn) - NEW VERSION")
    print("Expected: __imf_fdiv_rn - External function call")
    print("=" * 70)
    
    # Benchmark
    ms = benchmark(div_kernel, x, y, out, n_elements)
    num_gb = (n_elements * 4 * 3) / 1e9
    print(f"Time:       {ms:.3f} ms")
    print(f"Throughput: {num_gb / (ms / 1e3):.2f} GB/s")
    print("=" * 70)