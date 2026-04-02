import triton
import triton.language as tl
import torch


@triton.jit
def minimal_2d_load_kernel(
    ptr,
    out_ptr,
    stride_row: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # tensor<8x16x!tt.ptr<f16>>
    ptrs = ptr + offs_m[:, None] * stride_row + offs_n[None, :]

    data = tl.load(ptrs)
    tl.store(out_ptr + offs_m[:, None] * stride_row + offs_n[None, :], data)


x = torch.randn(8, 64, dtype=torch.float16, device='xpu')
out = torch.empty_like(x)
minimal_2d_load_kernel[(1, )](x, out, x.stride(0), BLOCK_M=8, BLOCK_N=16)
