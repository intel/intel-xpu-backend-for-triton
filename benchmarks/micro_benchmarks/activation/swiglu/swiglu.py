import torch
import intel_extension_for_pytorch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def kernel_swiglu_fwd(
    x_ptrs,
    y_ptrs,
    z_ptrs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptrs + offsets, mask).to(tl.float32)
    y = tl.load(y_ptrs + offsets, mask).to(tl.float32)

    u = tl.sigmoid(x)
    v = x * u  # silu
    z = v * y  # swiglu

    z = z.to(tl.float16)
    tl.store(z_ptrs + offsets, z, mask)


@triton.jit
def kernel_swiglu_bwd(
    x_ptrs,
    y_ptrs,
    dz_ptrs,
    dx_ptrs,
    dy_ptrs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptrs + offsets, mask).to(tl.float32)
    y = tl.load(y_ptrs + offsets, mask).to(tl.float32)
    dz = tl.load(dz_ptrs + offsets, mask).to(tl.float32)

    u = tl.sigmoid(x)
    v = x * u  # silu
    dy = dz * v
    dt = dz * y  # temp
    dx = dt * u * (1.0 + x * (1.0 - u))

    dx = dx.to(tl.float16)
    dy = dy.to(tl.float16)
    tl.store(dx_ptrs + offsets, dx, mask)
    tl.store(dy_ptrs + offsets, dy, mask)


class TritonSwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z = silu(x) * y = x * sigmoid(x) * y
        n_elemnts = x.numel()
        z = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(n_elemnts, META['BLOCK_SIZE']), )
        kernel_swiglu_fwd[grid](
            x,
            y,
            z,
            n_elemnts,
            BLOCK_SIZE=1024,
        )

        return z

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, dz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors

        n_elemnts = x.numel()
        dx = torch.empty_like(x)
        dy = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(n_elemnts, META['BLOCK_SIZE']), )
        kernel_swiglu_bwd[grid](
            x,
            y,
            dz,
            dx,
            dy,
            n_elemnts,
            BLOCK_SIZE=1024,
        )

        return dx, dy


class TorchSwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = F.silu(x)
        z = v * y

        ctx.save_for_backward(x, y)

        return z

    @staticmethod
    def backward(ctx, dz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors

        u = torch.sigmoid(x)
        v = F.silu(x)
        dy = dz * v
        dt = dz * y
        dx = dt * u * (1.0 + x * (1.0 - u))
        return dx, dy


swiglu_tor = TorchSwiGLU.apply
swiglu_tri = TritonSwiGLU.apply


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[11008],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='swiglu-performance',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def benchmark(M, N, dtype, provider, mode='forward'):

    quantiles = [0.5, 0.2, 0.8]
    X = torch.randn(M * N, requires_grad=True, device='xpu', dtype=dtype)
    Y = torch.randn(M * N, requires_grad=True, device='xpu', dtype=dtype)

    # utility functions
    if provider == 'triton':
        fwd = lambda: swiglu_tri(X, Y)
        dZ = torch.ones(M * N, device='xpu', dtype=dtype, requires_grad=True)
    if provider == 'torch':
        fwd = lambda: swiglu_tor(X, Y)
        dZ = torch.ones(M * N, device='xpu', dtype=dtype, requires_grad=True)

    # forward pass
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles)
        gbps = lambda ms: 2 * X.numel() * X.element_size() / ms * 1e-6
    # backward pass
    if mode == 'backward':
        z = fwd()
        bwd = lambda: z.backward(dZ, retain_graph=True)
        ms, min_ms, max_ms = triton.testing.do_bench(bwd, quantiles=quantiles)
        gbps = lambda ms: 3 * X.numel() * X.element_size() / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
