import math
import torch
import intel_extension_for_pytorch

import triton
import triton.language as tl


class SinusoidKernel:

    @staticmethod
    @triton.jit
    def SinusoidTriton(output_ptr, stride_output_row, input_pos_ptr, stride_input_pos_ptr, num_cols: tl.constexpr,
                       block_size: tl.constexpr, dimensions: tl.constexpr, constant_N: tl.constexpr):
        row_pos_start_pointer = input_pos_ptr
        positions_input_ptr = row_pos_start_pointer + tl.arange(0, block_size)

        row_mask = tl.arange(0, block_size) < num_cols

        pos = tl.load(positions_input_ptr, mask=row_mask).to(tl.float32)

        i = tl.program_id(0) // 2  # (i = 0, 0, 1, 1, 2, 2, ...)
        denominator = tl.extra.intel.libdevice.pow(constant_N, (2 * i) / dimensions)

        if tl.program_id(0) % 2 == 0:
            sm_out = tl.math.sin(pos / denominator)  # Even
        else:
            sm_out = tl.math.cos(pos / denominator)  # Odd

        output_row_index = tl.program_id(0)
        output_row_start_pointer = output_ptr + (output_row_index * stride_output_row)
        col_offsets = tl.arange(0, block_size)
        output_start_ptr = output_row_start_pointer + col_offsets

        tl.store(output_start_ptr, sm_out, mask=row_mask)


class SinusoidTriton:

    @staticmethod
    def encode(input_positions: torch.tensor, embedded_dimensions: tl.constexpr,
               constant_N: tl.constexpr) -> torch.tensor:
        rows, cols = input_positions.shape
        assert input_positions.dim() == 2, "only accepts 2D tensors"
        assert rows == 1, "Input positions should only have 1 row"
        block_size = triton.next_power_of_2(cols)
        num_warps = 4
        if block_size > 2047:
            num_warps = 8
        if block_size > 4095:
            num_warps = 32

        NumberOfTokens = cols

        sm_out = torch.zeros(embedded_dimensions, NumberOfTokens, dtype=input_positions.dtype, device='xpu')

        rows, cols = sm_out.shape
        grid = (rows, )
        SinusoidKernel.SinusoidTriton[grid](sm_out, sm_out.stride(0), input_positions, input_positions.stride(0),
                                            NumberOfTokens, block_size=block_size, num_warps=num_warps,
                                            dimensions=embedded_dimensions, constant_N=constant_N)

        return sm_out


def posenc_sin_pytorch(Tokens: int, dimensions: torch.tensor, dtype, N=10000) -> torch.tensor:
    assert Tokens % 2 == 0, "Number of tokens should be divisible by 2"
    rows, Dimension = dimensions.shape  # cols is dimension

    positions = torch.arange(0, Tokens, dtype=dtype, device='xpu').unsqueeze_(1).repeat(1, Dimension)

    divisors = torch.exp(-math.log(N) * (2 * (dimensions // 2)) / Dimension)
    positions *= divisors

    positions[:, 0::2] = torch.sin(positions[:, 0::2])
    positions[:, 1::2] = torch.cos(positions[:, 1::2])

    return positions


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_tokens'],
        x_vals=[10000 * i for i in range(2, 10, 2)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='sinusoidal-encoding-performance',
        args={'num_dims': 512, 'N': 10000, 'dtype': torch.float16},
    ))
def benchmark(num_tokens, num_dims, N, dtype, provider):
    quantiles = [0.5, 0.2, 0.8]
    input_positions = torch.arange(0, num_tokens, dtype=dtype, device='xpu').unsqueeze(dim=0)
    dimensions = torch.arange(0, num_dims, dtype=dtype, device='xpu').unsqueeze(0).repeat(num_tokens, 1)

    triton_sin_encoding = lambda: SinusoidTriton.encode(input_positions, embedded_dimensions=num_dims, constant_N=N)
    torch_sin_encoding = lambda: posenc_sin_pytorch(Tokens=num_tokens, dimensions=dimensions, dtype=dtype, N=N)

    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(triton_sin_encoding, quantiles=quantiles)
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(torch_sin_encoding, quantiles=quantiles)

    gbps = lambda ms: 2 * num_tokens * num_dims * input_positions.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
