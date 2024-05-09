import math
import torch
import intel_extension_for_pytorch

import triton
import triton.language as tl

kAlpha = tl.constexpr(math.sqrt(2.0 / math.pi))


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


class GeluTritonKernel:

    @staticmethod
    @triton.jit
    def Gelu_Triton_Forward(
        output_ptr,
        stride_output_row,
        input_ptr,
        stride_input_row,
        num_cols,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.arange(0, block_size)

        # setup input ptrs
        input_pointers = input_ptr + (pid * stride_input_row) + offsets

        # move data to SRAM
        row_mask = tl.arange(0, block_size) < num_cols  # only load or write the real data
        x = tl.load(input_pointers, mask=row_mask).to(tl.float32)

        # Gelu formula
        sm_out = 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
        sm_out = sm_out.to(tl.float16)

        # move SRAM to HBM
        output_pointers = output_ptr + (pid * stride_output_row) + offsets
        tl.store(output_pointers, sm_out, mask=row_mask)

    @staticmethod
    @triton.jit
    def Gelu_Triton_Backward(
        output_ptr,
        stride_output_row,
        input_ptr,
        stride_input_row,
        num_cols,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.arange(0, block_size)

        # setup input ptrs
        input_pointers = input_ptr + (pid * stride_input_row) + offsets

        # move data to SRAM
        row_mask = tl.arange(0, block_size) < num_cols  # only load or write the real data
        x = tl.load(input_pointers, mask=row_mask).to(tl.float32)

        # Gelu formula
        tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        sm_out = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        sm_out = sm_out.to(tl.float16)

        # move SRAM to HBM
        output_pointers = output_ptr + (pid * stride_output_row) + offsets
        tl.store(output_pointers, sm_out, mask=row_mask)


class GeluTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        rows, cols = input.shape
        output = GeluTriton.__forward(input, rows, cols)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        (input, ) = ctx.saved_tensors
        rows, cols = input.shape
        output = GeluTriton.__backward(input, rows, cols)
        return output

    @staticmethod
    def __forward(input: torch.Tensor, rows: int, cols: int) -> torch.Tensor:

        assert input.dim() == 2, f"only accepts 2D tensors"
        block_size = triton.next_power_of_2(cols)
        num_warps = 4  # * 32 (32 threads in a warp)
        if block_size > 2047:  # 2048
            num_warps = 8
        if block_size > 4095:  # 4096
            num_warps = 32

        grid = (rows, )

        sm_out = torch.empty_like(input)
        GeluTritonKernel.Gelu_Triton_Forward[grid](sm_out, sm_out.stride(0), input, input.stride(0), cols,
                                                   block_size=block_size, num_warps=num_warps)

        return sm_out

    @staticmethod
    def __backward(input: torch.Tensor, rows: int, cols: int) -> torch.Tensor:

        assert input.dim() == 2, f"only accepts 2D tensors for now"
        block_size = triton.next_power_of_2(cols)
        num_warps = 4  # * 32 (32 threads in a warp)
        if block_size > 2047:  # 2048
            num_warps = 8
        if block_size > 4095:  # 4096
            num_warps = 32

        grid = (rows, )

        # allocate

        sm_out = torch.empty_like(input)  # duplicate size of X but empty
        GeluTritonKernel.Gelu_Triton_Backward[grid](sm_out, sm_out.stride(0), input, input.stride(0), cols,
                                                    block_size=block_size, num_warps=num_warps)

        return sm_out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[11008],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='gelu-performance',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'},
    ))
def benchmark(M, N, dtype, provider, mode='forward'):
    quantiles = [0.5, 0.2, 0.8]
    inputs = torch.rand(M, N, dtype=dtype, device='xpu', requires_grad=True)

    if provider == 'triton':
        fwd = lambda: GeluTriton.apply(inputs)
    if provider == 'torch':
        fwd = lambda: torch.nn.functional.gelu(inputs)

    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles)
        gbps = lambda ms: inputs.numel() * inputs.element_size() / ms * 1e-6
    if mode == 'backward':
        y = fwd()
        gradient_matrix = torch.ones(M, N, dtype=dtype, device='xpu')
        bwd = lambda: y.backward(gradient_matrix, retain_graph=True)
        ms, min_ms, max_ms = triton.testing.do_bench(bwd, quantiles=quantiles)
        gbps = lambda ms: 2 * inputs.numel() * inputs.element_size() / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
