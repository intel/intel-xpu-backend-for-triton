import os
import sys

import torch
import triton
import triton.language as tl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from triton_kernels_benchmark import Benchmark, do_bench, perf_report  # pylint: disable=C0413

TYPES = {
    tl.float8e4nv: torch.float8_e4m3fn, tl.float8e5: torch.float8_e5m2, tl.float16: torch.float16, tl.bfloat16:
    torch.bfloat16, tl.float32: torch.float32
}


@triton.jit
def float_conversion_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    x_type: tl.constexpr,
    y_type: tl.constexpr,
    rnd: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_itype = tl.int8 if x_type.itemsize == 1 else tl.int16 if x_type.itemsize == 2 else tl.int32
    y_itype = tl.int8 if y_type.itemsize == 1 else tl.int16 if y_type.itemsize == 2 else tl.int32

    x = tl.load(x_ptr + offsets, mask=mask)
    converted = x.to(y_type, fp_downcast_rounding=rnd)
    x = tl.cast(x, x_itype, bitcast=True)
    y = tl.cast(converted, y_itype, bitcast=True)
    for i in range(99):
        x += tl.full(x.shape, i, x_itype)
        converted = tl.cast(x, x_type, bitcast=True).to(y_type, fp_downcast_rounding=rnd)
        y += tl.cast(converted, y_itype, bitcast=True)
    y = tl.cast(y, y_type, bitcast=True)
    tl.store(y_ptr + offsets, y, mask=mask)


def get_bench(x_type, y_type):
    assert x_type.itemsize < y_type.itemsize
    plot_name = f'{x_type}-{y_type}'
    line_vals = [(x_type, y_type, None), (y_type, x_type, 'rtne')]
    line_names = [f'{x_type}->{y_type}', f'{y_type}->{x_type}-rtne']
    if y_type == tl.float32:
        line_vals.append((y_type, x_type, 'rtz'))
        line_names.append(f'{y_type}->{x_type}-rtz')

    @perf_report(
        Benchmark(
            x_names=['N'],
            x_vals=[2**i for i in range(12, 28, 2)],
            line_arg='args',
            line_vals=line_vals,
            line_names=line_names,
            styles=[(c, s) for c in 'bgry' for s in ('-', '--', '-.', ':')],
            ylabel=('GB/s', ),
            plot_name=plot_name,
            args={},
        ))
    def bench(N, args):
        quantiles = [0.5, 0.2, 0.8]
        x_type = args[0]
        y_type = args[1]
        if x_type.itemsize == 1:
            x = torch.rand(N, dtype=torch.float16, device='xpu', requires_grad=True).to(TYPES[x_type])
        else:
            x = torch.rand(N, dtype=TYPES[x_type], device='xpu', requires_grad=True)
        y = torch.empty_like(x, dtype=TYPES[y_type], device='xpu')
        rnd = args[2] if x_type.itemsize > y_type.itemsize else None

        def fwd():
            BLOCK_SIZE = 4096
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
            float_conversion_kernel[grid](x, y, N, BLOCK_SIZE, x_type, y_type, rnd)
            return x

        _, min_ms, max_ms, mean_ms, cv = do_bench(fwd, n_warmup=10, n_repeat=10, quantiles=quantiles)
        gbps = lambda ms: (N * x.element_size() * 1e-9) / (ms * 1e-3)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), cv

    return bench


def get_benchmarks():
    return [get_bench(s, t) for s in TYPES for t in TYPES if s.itemsize < t.itemsize]


def run_benchmarks():
    for bench in get_benchmarks():
        bench.run(print_data=True)


if __name__ == '__main__':
    run_benchmarks()
