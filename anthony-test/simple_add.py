# Attempt to write a simple 2d matrix addition with blocks

import torch
import triton
import triton.language as tl


@triton.jit
def add_2d_block(x_ptr, y_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, stride_m: tl.constexpr, stride_n: tl.constexpr,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting indices of the block
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(M, N), strides=(stride_m, stride_n), offsets=(m_start, n_start),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    y_block_ptr = tl.make_block_ptr(base=y_ptr, shape=(M, N), strides=(stride_m, stride_n), offsets=(m_start, n_start),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    x_data = tl.load(x_block_ptr, boundary_check=(0, 1))
    y_data = tl.load(y_block_ptr, boundary_check=(0, 1))

    z_data = x_data + y_data

    z_block_ptr = tl.make_block_ptr(base=z_ptr, shape=(M, N), strides=(stride_m, stride_n), offsets=(m_start, n_start),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    tl.store(z_block_ptr, z_data, boundary_check=(0, 1))


def triton_2dadd(X, Y):
    if (X.shape != Y.shape):
        raise ValueError("Input tensors must have the same shape.")

    M, N = X.shape
    stride0, stride1 = X.stride(0), X.stride(1)

    # Allocates output.
    Z = torch.empty((M, N), device=X.device, dtype=X.dtype)

    # Defines block size.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64

    # Computes grid size.
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    print("stride0:", stride0)
    print("stride1:", stride1)

    # Launches kernel.
    add_2d_block[grid](
        X,
        Y,
        Z,
        M,
        N,
        stride0,
        stride1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return Z


M = 512
N = 512
dtype = torch.float32

X = torch.randn((M, N), device='cpu', dtype=dtype)
Y = torch.randn((N, M), device='cpu', dtype=dtype)

triton_result = triton_2dadd(X, Y)
torch_result = X + Y

# Verify correctness
if not torch.allclose(triton_result, torch_result, atol=1e-1):
    print("Results do not match!")
