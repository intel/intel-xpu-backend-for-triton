import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import numpy_random, to_triton, unwrap_tensor, tma_dtypes
from typing import Optional


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32)])
def test_tensor_descriptor_load(dtype_str, M_BLOCK, N_BLOCK):

    @triton.jit
    def kernel1(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )

        assert desc.shape[0] == M
        assert desc.shape[1] == N
        assert desc.strides[0] == N
        assert desc.strides[1] == 1
        assert desc.block_shape == [M_BLOCK, N_BLOCK]
        block = desc.load([M_BLOCK, 2 * N_BLOCK])
        idx = tl.arange(0, M_BLOCK)[:, None] * N_BLOCK + tl.arange(0, N_BLOCK)[None, :]
        tl.store(out_ptr + idx, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    M, N = 32, 128
    inp = to_triton(numpy_random((M, N), dtype_str), device="xpu", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M_BLOCK, N_BLOCK))

    kernel1[(1, )](out, inp, M, N, M_BLOCK, N_BLOCK)

    expect = unwrap_tensor(inp)[1 * M_BLOCK:2 * M_BLOCK, 2 * N_BLOCK:3 * N_BLOCK]
    torch.testing.assert_close(expect, unwrap_tensor(out))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32)])
def test_tensor_descriptor_store(dtype_str, M_BLOCK, N_BLOCK):

    @triton.jit
    def kernel2(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        midx = moffset + tl.arange(0, M_BLOCK)[:, None]
        nidx = noffset + tl.arange(0, N_BLOCK)[None, :]
        idx = midx * N + nidx

        val = tl.load(a_ptr + idx)

        desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )

        assert desc.shape[0] == M
        assert desc.shape[1] == N
        assert desc.strides[0] == N
        assert desc.strides[1] == 1
        assert desc.block_shape == [M_BLOCK, N_BLOCK]
        desc.store([moffset, noffset], val)

    M, N = 32, 128
    inp = to_triton(numpy_random((M, N), dtype_str), device="xpu", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    kernel2[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))


# Exercise the functional load/store builtins once to ensure they map through.
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_tensor_descriptor_functional_interface(dtype_str):
    """Copies an entire tensor blockwise using the descriptor builtins."""

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        block = tl.load_tensor_descriptor(in_desc, [moffset, noffset])
        tl.store_tensor_descriptor(out_desc, [moffset, noffset], block)

    M, N = 32, 128
    inp = to_triton(numpy_random((M, N), dtype_str), device="xpu", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 2 * 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))
