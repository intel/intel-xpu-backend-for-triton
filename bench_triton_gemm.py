"""
Benchmark Script 2: Triton compiled GEMM (the slow case)
Reproduces the "bad" performance case from issue #6012.
This runs the GEMM through torch.compile with max-autotune which uses Triton.
"""
import torch
import time
import os

# Force triton backend for GEMM
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "TRITON"

print("=" * 60)
print("Triton Compiled GEMM Benchmark (torch.compile max-autotune)")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"XPU device: {torch.xpu.get_device_name(0)}")
print(f"TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS: {os.environ.get('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS', 'not set')}")
print()

# Setup - matching the issue exactly
flush_cache = torch.randn(64, 1024, 1024, dtype=torch.float16, device="xpu")
x = torch.randn(10000, 64, dtype=torch.float16, device="xpu")
w = torch.randn(64, 64, dtype=torch.float16, device="xpu")
b = torch.randn(1, 64, dtype=torch.float16, device="xpu")


def linear(src, wei, bias):
    return torch.addmm(bias, src, wei.t())


# Compile with max-autotune to use Triton GEMM template
linear_compiled = torch.compile(linear, mode="max-autotune")

# Warmup (this triggers compilation)
print("Warming up and compiling (10 iterations)...")
for _ in range(10):
    torch.relu(flush_cache)
    y = linear_compiled(x, w, b)
torch.xpu.synchronize()

# Benchmark
num_iters = 100
print(f"Benchmarking ({num_iters} iterations)...")

torch.xpu.synchronize()
start_time = time.perf_counter_ns()
for _ in range(num_iters):
    torch.relu(flush_cache)
    y = linear_compiled(x, w, b)
torch.xpu.synchronize()
end_time = time.perf_counter_ns()

total_ns = end_time - start_time
avg_ns = total_ns / num_iters

# Also time just the GEMM without flush
torch.xpu.synchronize()
start_time2 = time.perf_counter_ns()
for _ in range(num_iters):
    y = linear_compiled(x, w, b)
torch.xpu.synchronize()
end_time2 = time.perf_counter_ns()

total_ns2 = end_time2 - start_time2
avg_ns2 = total_ns2 / num_iters

# GEMM stats
M, K, N = 10000, 64, 64
flops = 2 * M * N * K  # multiply-add = 2 ops
data_bytes = (M * K + K * N + M * N) * 2  # fp16 = 2 bytes

print()
print("Results (with cache flush):")
print(f"  Total time:   {total_ns/1e6:.3f} ms")
print(f"  Avg per iter: {avg_ns/1e3:.1f} us ({avg_ns:.0f} ns)")
print(f"  GFLOPS:       {flops / (avg_ns):.2f}")
print(f"  BW (GB/s):    {data_bytes / avg_ns:.2f}")
print()
print("Results (GEMM only, no cache flush):")
print(f"  Total time:   {total_ns2/1e6:.3f} ms")
print(f"  Avg per iter: {avg_ns2/1e3:.1f} us ({avg_ns2:.0f} ns)")
print(f"  GFLOPS:       {flops / (avg_ns2):.2f}")
print(f"  BW (GB/s):    {data_bytes / avg_ns2:.2f}")
print()

# Store result for comparison
print(f"KEY_METRIC: triton_avg_ns={avg_ns:.0f}")
