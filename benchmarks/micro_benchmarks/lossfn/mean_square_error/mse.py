import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl


class utils:

    @staticmethod
    @triton.jit
    def GetStartingPointerData(input_ptr, stride_input_row, block_size):
        row_index = tl.program_id(0)
        row_start_ptr = input_ptr + (row_index * stride_input_row)
        col_offsets = tl.arange(0, block_size)
        return row_start_ptr + col_offsets


class MSETritonKernel:

    @staticmethod
    @triton.jit
    def MSE_Triton_Forward(
        output_ptr,
        stride_output_row,
        predicted_ptr,
        predicted_stride_row,
        expected_ptr,
        expected_stride_row,
        N,
        block_size: tl.constexpr,
    ):

        pid = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        predicted_pointers = predicted_ptr + (pid * predicted_stride_row) + offsets
        expected_pointers = expected_ptr + (pid * expected_stride_row) + offsets
        row_mask = tl.arange(0, block_size) < N  # only load or write the real data

        predicted_row_data = tl.load(predicted_pointers, mask=row_mask)
        expected_row_data = tl.load(expected_pointers, mask=row_mask)

        sm_out = tl.sum(tl.extra.intel.libdevice.pow(predicted_row_data - expected_row_data, 2) / N, 0)

        # move SRAM to HBM
        output_pointers = output_ptr + (pid * stride_output_row) + offsets
        tl.store(output_pointers, sm_out, mask=row_mask)

    @staticmethod
    @triton.jit
    def MSE_Triton_Backward(output_ptr, stride_output_row, predicted_ptr, predicted_stride_row, expected_ptr,
                            expected_stride_row, N, block_size: tl.constexpr):

        # setup input ptrs
        pid = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        predicted_pointer = predicted_ptr + (pid * predicted_stride_row) + offsets
        expected_pointer = expected_ptr + (pid * expected_stride_row) + offsets

        # move data to SRAM
        row_mask = tl.arange(0, block_size) < N  # only load or write the real data
        predicted = tl.load(predicted_pointer, mask=row_mask)
        expected = tl.load(expected_pointer, mask=row_mask)
        sm_out = ((predicted - expected) * 2) / N

        # move SRAM to HBM
        tl.store(utils.GetStartingPointerData(output_ptr, stride_output_row, block_size), sm_out, mask=row_mask)


class MSETriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, predicted, expected):
        rows, cols = predicted.shape
        output = MSETriton.__forward(predicted, expected, rows, cols)
        ctx.save_for_backward(predicted, expected)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        predicted, expected = ctx.saved_tensors
        rows, cols = predicted.shape
        output = MSETriton.__backward(predicted, expected, rows, cols)
        return output, None

    @staticmethod
    def __forward(predicted: torch.Tensor, expected: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        # assert input.dim() == 2, f"only accepts 2D tensors"
        block_size = triton.next_power_of_2(cols)
        num_warps = 4  # * 32 (32 threads in a warp)
        if block_size > 2047:  # 2048
            num_warps = 8
        if block_size > 4095:  # 4096
            num_warps = 32

        grid = (rows, )

        sm_out = torch.empty_like(predicted)
        MSETritonKernel.MSE_Triton_Forward[grid](sm_out, sm_out.stride(0), predicted, predicted.stride(0), expected,
                                                 expected.stride(0), cols, block_size=block_size, num_warps=num_warps)

        return sm_out

    @staticmethod
    def __backward(predicted: torch.Tensor, expected: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        assert predicted.dim() == 2, f"only accepts 2D tensors for now"
        block_size = triton.next_power_of_2(cols)
        num_warps = 4  # * 32 (32 threads in a warp)
        if block_size > 2047:  # 2048
            num_warps = 8
        if block_size > 4095:  # 4096
            num_warps = 32

        grid = (rows, )

        # allocate

        sm_out = torch.empty_like(predicted)  # duplicate size of X but empty
        MSETritonKernel.MSE_Triton_Backward[grid](sm_out, sm_out.stride(0), predicted, predicted.stride(0), expected,
                                                  expected.stride(0), cols, block_size=block_size, num_warps=num_warps)

        return sm_out


def triton_MSE(predicted: torch.tensor, expected: torch.tensor) -> torch.tensor:
    predicted.grad = None
    output = MSETriton.apply(predicted, expected)
    output.backward(predicted, retain_graph=True)
    return output.data[0][0]


def torch_MSE(predicted: torch.tensor, expected: torch.tensor) -> torch.tensor:
    predicted.grad = None
    MSELoss = torch.nn.MSELoss()
    output = MSELoss(predicted, expected)
    output.backward()
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[256, 1024, 2048, 4096],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float32, 'mode': 'forward'},
    ))
def benchmark(M, N, dtype, provider, mode='forward'):
    predicted = torch.rand(1, M * N, dtype=dtype, device="xpu", requires_grad=True)

    expected = torch.rand(1, M * N, dtype=dtype, device="xpu")

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_MSE(predicted, expected))
    if provider == "torch":
        predicted_torch = predicted.clone().detach().requires_grad_(True)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_MSE(predicted_torch, expected))

    gbps = lambda ms: 2 * predicted.nelement() * predicted.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
