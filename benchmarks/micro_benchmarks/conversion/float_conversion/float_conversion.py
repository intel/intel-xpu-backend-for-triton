import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl


@triton.jit
def float_trunc_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    target_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    as_target = x.to(target_type)
    as_f32 = as_target.to(tl.float32)
    for _ in range(100):
        as_f32 += 1  # plus one ensures that there are no redundant conversions that can be removed
        as_target = as_f32.to(target_type)
        as_f32 = as_target.to(tl.float32)

    tl.store(x_ptr + offsets, as_f32, mask=mask)


def launch_conversion(x: torch.Tensor, target_type: type):
    assert x.is_xpu
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    float_trunc_kernel[grid](x, n_elements, BLOCK_SIZE=1024, target_type=target_type)
    return x


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(12, 28, 2)],
        line_arg='target_type',
        line_vals=['bfloat16', 'float16'],
        line_names=['BF16', 'FP16'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='float-conversion',
        args={},
    ))
def benchmark(N, target_type):
    quantiles = [0.5, 0.2, 0.8]
    inputs = torch.rand(N, dtype=torch.float32, device='xpu', requires_grad=True)

    if target_type == "bfloat16":
        fwd = lambda: launch_conversion(inputs, tl.bfloat16)
    elif target_type == "float16":
        fwd = lambda: launch_conversion(inputs, tl.float16)
    else:
        raise NotImplementedError(f'Type {target_type} is not supported')

    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles)
    gbps = lambda ms: (inputs.numel() * inputs.element_size() * 1e-9) / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
