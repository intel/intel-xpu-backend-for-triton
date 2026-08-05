"""
Micro-benchmark for the TritonIntelGPUCanonicalizePointers pass.

Workload: a fused elementwise sum of N input tensors, processed block-by-block
over a long axis (out = sum_k A_k). This is the multi-tensor / foreach pattern
(e.g. torch._foreach_add, gradient accumulation, ensemble averaging) -- every
load feeds the output, so nothing is dead-code-eliminated.

The kernel naturally carries N pointer tensors across the loop. Without the
pass, each loop-carried iter_arg is a `tensor<BLOCK x !tt.ptr<f16>>` (BLOCK x 8
bytes of live i64 per pointer), driving GRF register pressure far above the
128-GRF working set. The pass decomposes each into a scalar base pointer +
recomputed offset, collapsing the loop-carried state to a handful of scalars.

The two "providers" are the SAME kernel compiled with the pass disabled vs
enabled (toggled via TRITON_INTEL_DISABLE_CANONICALIZE_POINTERS). Each provider
forces a recompile with the env var flipped, so the benchmark is independent of
the pass's default setting.
"""

import os
import tempfile

import torch
import triton
import triton.language as tl

from triton_kernels_benchmark import Benchmark, do_bench, perf_report, assert_close

DEVICE = 'xpu'

NUM_TENSORS = 16
NUM_ITERS = 128
GRID = 64

_CANON_ENV = 'TRITON_INTEL_DISABLE_CANONICALIZE_POINTERS'


@triton.jit
def fused_sum_kernel(out_ptr, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, N,
                     BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # 16 pointer tensors -- each becomes a loop-carried iter_arg.
    a0 = p0 + offs
    a1 = p1 + offs
    a2 = p2 + offs
    a3 = p3 + offs
    a4 = p4 + offs
    a5 = p5 + offs
    a6 = p6 + offs
    a7 = p7 + offs
    a8 = p8 + offs
    a9 = p9 + offs
    a10 = p10 + offs
    a11 = p11 + offs
    a12 = p12 + offs
    a13 = p13 + offs
    a14 = p14 + offs
    a15 = p15 + offs
    o = out_ptr + offs

    for _ in tl.range(0, ITERS):
        m = offs < N
        s = (tl.load(a0, m) + tl.load(a1, m) + tl.load(a2, m) + tl.load(a3, m) + tl.load(a4, m) + tl.load(a5, m) +
             tl.load(a6, m) + tl.load(a7, m) + tl.load(a8, m) + tl.load(a9, m) + tl.load(a10, m) + tl.load(a11, m) +
             tl.load(a12, m) + tl.load(a13, m) + tl.load(a14, m) + tl.load(a15, m))
        tl.store(o, s, m)
        a0 += BLOCK
        a1 += BLOCK
        a2 += BLOCK
        a3 += BLOCK
        a4 += BLOCK
        a5 += BLOCK
        a6 += BLOCK
        a7 += BLOCK
        a8 += BLOCK
        a9 += BLOCK
        a10 += BLOCK
        a11 += BLOCK
        a12 += BLOCK
        a13 += BLOCK
        a14 += BLOCK
        a15 += BLOCK
        o += BLOCK


def _select_pass(enabled: bool):
    """Make the next kernel launch recompile with the canonicalize pass on/off.

    The pass toggle is read at compile time and is NOT part of Triton's compile
    cache key, so a plain re-launch would reuse whatever binary was compiled
    first. To actually get a fresh compile with the requested setting we:
      1. set the toggle env var,
      2. force recompilation (TRITON_ALWAYS_COMPILE) into a per-setting cache
         dir so the on-disk caches don't collide, and
      3. clear the JITFunction's in-memory cache.
    """
    os.environ[_CANON_ENV] = '0' if enabled else '1'
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    os.environ['TRITON_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f"canon_bench_{'on' if enabled else 'off'}")
    fused_sum_kernel.device_caches.clear()


@perf_report(
    Benchmark(
        x_names=['BLOCK'],
        x_vals=[256, 512, 1024, 2048],
        line_arg='provider',
        line_vals=['baseline', 'canonicalized'],
        line_names=['pass off', 'pass on'],
        styles=[('blue', '-'), ('orange', '-')],
        ylabel=['GB/s'],
        plot_name='fused-sum-16',
        args={},
    ))
def benchmark_fused_sum(BLOCK, provider):
    quantiles = [0.5, 0.0, 1.0]
    dtype = torch.float16
    element_bytes = torch.finfo(dtype).bits // 8
    torch.manual_seed(0)

    N = BLOCK * GRID
    span = N + NUM_ITERS * BLOCK  # each program streams NUM_ITERS blocks forward
    inputs = [torch.randn(span, dtype=dtype, device=DEVICE) for _ in range(NUM_TENSORS)]
    output = torch.empty(span, dtype=dtype, device=DEVICE)
    grid = (GRID, )

    _select_pass(provider == 'canonicalized')
    fn = lambda: fused_sum_kernel[grid](output, *inputs, N, BLOCK, NUM_ITERS)
    fn()  # compile with the requested pass setting; populates in-memory cache
    _, min_ms, max_ms, mean_ms, cv = do_bench(fn, n_warmup=25, n_repeat=100, quantiles=quantiles)

    def gbps(ms):
        # NUM_TENSORS reads + 1 write, per block, per iteration, per program.
        return GRID * NUM_ITERS * (NUM_TENSORS + 1) * BLOCK * element_bytes * 1e-9 / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv


def _verify():
    """Sanity check: the pass must not change results."""
    dtype = torch.float16
    BLOCK = 1024
    N = BLOCK * GRID
    span = N + NUM_ITERS * BLOCK
    torch.manual_seed(0)
    inputs = [torch.randn(span, dtype=dtype, device=DEVICE) for _ in range(NUM_TENSORS)]
    grid = (GRID, )

    def compute(enabled):
        out = torch.zeros(span, dtype=dtype, device=DEVICE)
        _select_pass(enabled)
        fused_sum_kernel[grid](out, *inputs, N, BLOCK, NUM_ITERS)
        return out

    assert_close(lambda: compute(False), lambda: compute(True), atol=1e-2, rtol=1e-2)


def run_benchmarks():
    _verify()
    benchmark_fused_sum.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    run_benchmarks()
