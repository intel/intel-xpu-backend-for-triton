import numpy as np
import pytest
import torch

import triton
import triton.language as tl


@pytest.mark.parametrize("BLOCK_M, BLOCK_N", [(128, 128), (64, 128), (32, 128), (32, 64), (32, 32)])
def test_tensor_atomic_rmw_2d_grid(BLOCK_M, BLOCK_N, device):
    """Regression: atomic_add must not be duplicated by remove-layout-conversions
    when the grid has multiple row-tile programs (BLOCK_M < M)."""
    M, N = 128, 128

    @triton.jit
    def kernel(dy_ptr, x_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        num_blocks_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_blocks_m
        pid_n = pid // num_blocks_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        val = tl.sum(x, axis=1)
        tl.atomic_add(dy_ptr + rows, val, sem="relaxed")

    x = torch.ones(M, N, device=device, dtype=torch.float32)
    dy = torch.zeros(M, device=device, dtype=torch.float32)
    nm = triton.cdiv(M, BLOCK_M)
    nn = triton.cdiv(N, BLOCK_N)
    kernel[(nm * nn, )](dy, x, M, N, BLOCK_M, BLOCK_N, num_warps=4)

    expected = torch.full((M, ), float(N), device=device, dtype=torch.float32)
    np.testing.assert_allclose(dy.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4)


@pytest.mark.parametrize("BLOCK_M, BLOCK_N", [(128, 128), (64, 128), (32, 128), (32, 64), (32, 32)])
def test_tensor_atomic_rmw_2d_grid_with_1d_load(BLOCK_M, BLOCK_N, device):
    """Regression: atomic_add must not be duplicated by remove-layout-conversions
    when a 1D row load (y_ptr + rows) shares the `rows` variable with the 2D
    load and the atomic pointer.  This pattern triggers the bug more aggressively
    than the 2D-only case (at lower nm values, with up to 3x over-count)."""
    M, N = 128, 128

    @triton.jit
    def kernel(dy_ptr, x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        num_blocks_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_blocks_m
        pid_n = pid // num_blocks_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :])
        y = tl.load(y_ptr + rows)
        val = (x * y[:, None]).sum(axis=1)
        tl.atomic_add(dy_ptr + rows, val, sem="relaxed")

    x = torch.ones(M, N, device=device, dtype=torch.float32)
    y = torch.ones(M, device=device, dtype=torch.float32)
    dy = torch.zeros(M, device=device, dtype=torch.float32)
    nm = triton.cdiv(M, BLOCK_M)
    nn = triton.cdiv(N, BLOCK_N)
    kernel[(nm * nn, )](dy, x, y, M, N, BLOCK_M, BLOCK_N, num_warps=4)

    expected = torch.full((M, ), float(N), device=device, dtype=torch.float32)
    np.testing.assert_allclose(dy.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4)
