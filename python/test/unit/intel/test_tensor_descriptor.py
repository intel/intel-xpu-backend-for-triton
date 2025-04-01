import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_interpreter, numpy_random, to_triton, unwrap_tensor, tma_dtypes
from typing import Optional


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32)])
def test_tensor_descriptor_load(dtype_str, M_BLOCK, N_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
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

    M, N = M_BLOCK * 3, N_BLOCK * 4
    inp = to_triton(numpy_random((M, N), dtype_str), device="xpu", dst_type=dtype_str)
    out = inp.new_empty((M_BLOCK, N_BLOCK))

    kernel[(1, )](out, inp, M, N, M_BLOCK, N_BLOCK)

    expect = unwrap_tensor(inp)[1 * M_BLOCK:2 * M_BLOCK, 2 * N_BLOCK:3 * N_BLOCK]
    torch.testing.assert_close(expect, unwrap_tensor(out))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32)])
def test_tensor_descriptor_store(dtype_str, M_BLOCK, N_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
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
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)

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


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("K_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_load3d(dtype_str, K_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
               K_BLOCK: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N, K],
            strides=[stride_m, stride_n, stride_k],
            block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
        )

        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs = pid_m * M_BLOCK, pid_n * N_BLOCK, pid_k * K_BLOCK

        block = desc.load(offs)

        idx_m = offs[0] + tl.arange(0, M_BLOCK)[:, None, None]
        idx_n = offs[1] + tl.arange(0, N_BLOCK)[None, :, None]
        idx_k = offs[2] + tl.arange(0, K_BLOCK)[None, None, :]
        idx = idx_m * N * K + idx_n * K + idx_k
        mask = (idx_m < M) & (idx_n < N) & (idx_k < K)
        tl.store(out_ptr + idx, block, mask)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    inp = to_triton(numpy_random((10, 64, 128), dtype_str), device="xpu", dst_type=dtype_str)
    inp.data = inp.data[:, :50, :119]

    if K_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    M_BLOCK, N_BLOCK = 8, 8
    out = inp.new_empty(inp.shape)

    grid = tuple(triton.cdiv(size, block) for size, block in zip(inp.shape, (M_BLOCK, N_BLOCK, K_BLOCK)))
    kernel[grid](out, inp, *inp.shape, *inp.stride(), M_BLOCK, N_BLOCK, K_BLOCK)

    actual = unwrap_tensor(out)
    expect = unwrap_tensor(inp)
    torch.testing.assert_close(expect, actual)


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("K_BLOCK", [16, 32, 64, 128])
def test_tensor_descriptor_store3d(dtype_str, K_BLOCK):

    if dtype_str == 'bfloat16':
        return pytest.skip("TODO: bfloat16 test fails verification")

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
               K_BLOCK: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N, K],
            strides=[stride_m, stride_n, stride_k],
            block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
        )

        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs = pid_m * M_BLOCK, pid_n * N_BLOCK, pid_k * K_BLOCK

        idx_m = offs[0] + tl.arange(0, M_BLOCK)[:, None, None]
        idx_n = offs[1] + tl.arange(0, N_BLOCK)[None, :, None]
        idx_k = offs[2] + tl.arange(0, K_BLOCK)[None, None, :]
        idx = idx_m * N * K + idx_n * K + idx_k
        mask = (idx_m < M) & (idx_n < N) & (idx_k < K)
        block = tl.load(a_ptr + idx, mask)

        desc.store(offs, block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    inp = to_triton(numpy_random((10, 50, 119), dtype_str), device="xpu", dst_type=dtype_str)

    if K_BLOCK * inp.element_size() < 32:
        return pytest.skip("Invalid last dim size")

    M_BLOCK, N_BLOCK = 8, 8
    out = inp.new_empty((10, 64, 128))

    grid = tuple(triton.cdiv(size, block) for size, block in zip(inp.shape, (M_BLOCK, N_BLOCK, K_BLOCK)))
    kernel[grid](out, inp, *inp.shape, *out.stride(), M_BLOCK, N_BLOCK, K_BLOCK)

    expect = unwrap_tensor(inp)
    actual = unwrap_tensor(out)[:, :50, :119]
    torch.testing.assert_close(expect, actual)


@triton.jit(noinline=False)
def tensor_descriptor_in_function_helper(out_ptr, in_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
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
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


@pytest.mark.interpreter
def test_tensor_descriptor_in_function():

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        tensor_descriptor_in_function_helper(out_ptr, a_ptr, M, N, M_BLOCK, N_BLOCK)

    M, N = 32, 128
    inp = torch.randn((M, N), device="xpu")

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

    expect = inp.abs()
    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(expect, out)


@triton.jit(noinline=False)
def tensor_descriptor_return_helper(ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    return tl.make_tensor_descriptor(
        ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )


@pytest.mark.interpreter
def test_tensor_descriptor_return_value():

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tensor_descriptor_return_helper(a_ptr, M, N, M_BLOCK, N_BLOCK)
        out_desc = tensor_descriptor_return_helper(out_ptr, M, N, M_BLOCK, N_BLOCK)
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        value = in_desc.load([moffset, noffset])
        out_desc.store([moffset, noffset], value.abs())

    M, N = 32, 128
    inp = torch.randn((M, N), device="xpu")

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_zeros((M, N))

    def alloc_fn(size: int, align: int, stream: Optional[int]) -> torch.Tensor:
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    expect = inp.abs()
    kernel[(M // M_BLOCK, N // N_BLOCK)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(expect, out)


@triton.jit(noinline=False)
def tensor_descriptor_arg_helper(in_desc, out_desc, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


@pytest.mark.interpreter
def test_tensor_descriptor_argument():

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        out_desc = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M_BLOCK, N_BLOCK])
        in_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1], block_shape=[M_BLOCK, N_BLOCK])
        tensor_descriptor_arg_helper(in_desc, out_desc, M_BLOCK, N_BLOCK)

    M, N = 32, 128
    inp = torch.randn((M, N), device="xpu")

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_zeros((M, N))

    def alloc_fn(size: int, align: int, stream: Optional[int]) -> torch.Tensor:
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    expect = inp.abs()
    kernel[(M // M_BLOCK, N // N_BLOCK)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(expect, out)


@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(a_desc.dtype)
    c_desc.store([offs_am, offs_bn], accumulator)


@pytest.mark.interpreter
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, num_stages", [
    (128, 128, 16, 1),
    (256, 64, 32, 1),
    (64, 512, 32, 1),
    (128, 128, 16, 1),
    (64, 128, 32, 1),
    (32, 32, 32, 1),
    (256, 128, 32, 1),
])
def test_make_tensor_descriptor_matmul(num_stages, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "xpu"
    if is_interpreter():
        M, N, K = BLOCK_M, BLOCK_N, BLOCK_K
    else:
        M, N, K = 1024, 512, 256
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 3 * 128 * grid[0] * grid[1]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    matmul_kernel_make_tensor_desciptor[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=8,
        num_stages=num_stages,
    )
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)


@triton.jit
def batched_gemm_3d_tma_kernel(a_ptr, b_ptr, c_ptr,  #
                               B, M, N, K,  #
                               dtype: tl.constexpr,  #
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                               NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles_per_batch = num_tiles_m * num_tiles_n
    num_tiles = B * num_tiles_per_batch

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    tile_m = 0
    tile_n = 0
    tile_b = 0

    offs_m = 0
    offs_n = 0
    offs_b = 0

    a_desc = tl.make_tensor_descriptor(a_ptr, [B, M, K], [K * M, K, 1], [1, BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [B, N, K], [N * K, K, 1], [1, BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, [B, M, N], [M * N, N, 1], [1, BLOCK_M, BLOCK_N])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            tile_b = tile_id // num_tiles_per_batch
            tile_m = (tile_id // num_tiles_n) % num_tiles_m
            tile_n = tile_id % num_tiles_n

            offs_b = tile_b
            offs_m = tile_m * BLOCK_M
            offs_n = tile_n * BLOCK_N

        offs_k = ki * BLOCK_K

        a = a_desc.load([offs_b, offs_m, offs_k]).reshape([BLOCK_M, BLOCK_K])
        b = b_desc.load([offs_b, offs_n, offs_k]).reshape([BLOCK_N, BLOCK_K])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            c_desc.store([offs_b, offs_m, offs_n], c.reshape((1, BLOCK_M, BLOCK_N)))
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


@pytest.mark.interpreter
def test_tensor_descriptor_batched_gemm_3d_tma():
    device = "xpu"
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    if is_interpreter():
        B, M, N, K = 2, BLOCK_M, BLOCK_N, BLOCK_K
    else:
        B, M, N, K = 2, 1024, 1024, 128
    NUM_SMS = 96
    num_stages = 3

    grid = (min(NUM_SMS, B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )

    a = torch.randn((B, M, K), device=device, dtype=torch.float16)
    b = torch.randn((B, N, K), device=device, dtype=torch.float16)
    c = torch.empty((B, M, N), device=device, dtype=torch.float16)

    expect = torch.bmm(a, b.mT)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        # TODO: should only need num_stages * 3 descriptors per SM
        assert size == 128 * 3 * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)

    batched_gemm_3d_tma_kernel[grid](
        a, b, c,  #
        B, M, N, K,  #
        tl.float16,  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        NUM_SMS,  #
        num_stages=num_stages, num_warps=8)
    torch.xpu.synchronize()

    torch.testing.assert_close(c, expect, rtol=1e-3, atol=1e-3)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit()
def matmul_kernel_rank_reducing(a_ptr, b_ptr, c_ptr,  #
                                M, N, K,  #
                                BLOCK_SIZE_M: tl.constexpr,  #
                                BLOCK_SIZE_N: tl.constexpr,  #
                                BLOCK_SIZE_K: tl.constexpr,  #
                                NUM_SMS: tl.constexpr):  #
    # Matmul using TMA and device-side descriptor creation
    GROUP_SIZE_M: tl.constexpr = 8
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[1, M, K],
        strides=[M * K, K, 1],
        block_shape=[1, BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[1, N, K],
        strides=[N * K, K, 1],
        block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[1, M, N],
        strides=[M * N, N, 1],
        block_shape=[1, BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([0, offs_am, offs_k]).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = b_desc.load([0, offs_bn, offs_k]).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        c = accumulator.to(dtype).reshape(1, BLOCK_SIZE_M, BLOCK_SIZE_N)
        c_desc.store([0, offs_cm, offs_cn], c)


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16", "float32"])
def test_tensor_descriptor_rank_reducing_matmul(dtype_str):
    NUM_SMS = 4
    M, N, K = 256, 256, 64
    A = to_triton(numpy_random((1, M, K), dtype_str), device="xpu", dst_type=dtype_str)
    B = to_triton(numpy_random((1, N, K), dtype_str), device="xpu", dst_type=dtype_str)
    C = A.new_empty(1, M, N)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        return torch.empty(size, dtype=torch.int8, device="xpu")

    triton.set_allocator(alloc_fn)
    matmul_kernel_rank_reducing[(NUM_SMS, )](
        A,
        B,
        C,
        M,
        N,
        K,
        NUM_SMS=4,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )

    actual = unwrap_tensor(C)
    expect = torch.matmul(A, B.mT)
    torch.testing.assert_close(expect, actual, atol=1e-1, rtol=1e-4)
