import pytest
import torch

import triton
import triton.language as tl
from triton.tools.experimental_descriptor import (create_1d_tma_descriptor, create_2d_tma_descriptor)
from triton._internal_testing import dtypes_with_bfloat16, numpy_random, to_triton, requires_tma, supports_tma, tma_skip_msg

from typing import Optional


def create_tma_desc_gmem_ptr(ptr, dims, block_dims, element_size):
    cpu_desc = torch.empty(128, device="cpu")
    if len(dims) == 1:
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dims[0], block_dims[0], element_size,
                                                                  cpu_desc.data_ptr())
    else:
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dims[0], dims[1], block_dims[0], block_dims[1],
                                                                  element_size, cpu_desc.data_ptr())
    return cpu_desc.cuda()


def unwrap_tensor(t: torch.Tensor | triton.runtime.jit.TensorWrapper):
    if isinstance(t, triton.runtime.jit.TensorWrapper):
        return t.base
    return t


tma_dtypes = sorted(set(dtypes_with_bfloat16) - {"int64", "uint64", "float64"})


@pytest.mark.parametrize("byval_tma", [True, False])
def test_experimetal_descriptor_load(byval_tma):
    if not supports_tma(byval_tma):
        pytest.skip(tma_skip_msg(byval_tma))

    device = "cuda"
    SIZE = 128

    @triton.jit
    def kernel(Z, desc, SIZE: tl.constexpr, BYVAL_TMA: tl.constexpr):
        if not BYVAL_TMA:
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc)
        off_desc = 0
        off = tl.arange(0, SIZE)
        x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype.element_ty)
        tl.store(Z + off, x)

    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    if byval_tma:
        desc = create_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size())
    else:
        desc = create_tma_desc_gmem_ptr(x.data_ptr(), [SIZE], [SIZE], x.element_size())
    z_tri = torch.empty_like(x)
    compiled_kernel = kernel[(1, )](z_tri, desc, SIZE=SIZE, BYVAL_TMA=byval_tma, num_warps=4)
    assert torch.equal(x, z_tri)
    if byval_tma:
        assert ".param .align 64 .b8" in compiled_kernel.asm["ptx"]


@triton.jit
def matmul_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                      M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      BYVAL_TMA: tl.constexpr, dtype: tl.constexpr):
    if not BYVAL_TMA:
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], dtype)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(dtype)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@pytest.mark.parametrize("num_stages", [1, 4])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 32), (128, 64, 64), (128, 128, 64), (128, 256, 64)])
@pytest.mark.parametrize("byval_tma", [True, False])
def test_experimental_tma_matmul(num_stages, BLOCK_M, BLOCK_N, BLOCK_K, byval_tma):
    if not supports_tma(byval_tma):
        pytest.skip(tma_skip_msg(byval_tma))

    device = "cuda"
    M, N, K = 8192, 8192, 1024
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    if byval_tma:
        desc_a = create_2d_tma_descriptor(A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size())
        desc_b = create_2d_tma_descriptor(B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size())
        desc_c = create_2d_tma_descriptor(C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size())
    else:
        desc_a = create_tma_desc_gmem_ptr(A.data_ptr(), [M, K], [BLOCK_M, BLOCK_K], A.element_size())
        desc_b = create_tma_desc_gmem_ptr(B.data_ptr(), [K, N], [BLOCK_K, BLOCK_N], B.element_size())
        desc_c = create_tma_desc_gmem_ptr(C.data_ptr(), [M, N], [BLOCK_M, BLOCK_N], C.element_size())
    kernel = matmul_kernel_tma[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1,
                                1)](desc_a, desc_b, desc_c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, BYVAL_TMA=byval_tma,
                                    num_warps=8, num_stages=num_stages, dtype=tl.float16)
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    if BLOCK_M >= 64 and BLOCK_N >= 64:
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]
    if byval_tma:
        assert ".param .align 64 .b8" in kernel.asm["ptx"]


@triton.jit
def device_tensormap_kernel2d(in_ptr, out_ptr, in_desc, out_desc, ready_flag, M, N, M_BLOCK: tl.constexpr,
                              N_BLOCK: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    if pid_m == 0 and pid_n == 0:
        # Write out descriptor
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=in_desc,
            global_address=in_ptr,
            load_size=[M_BLOCK, N_BLOCK],
            global_size=[M, N],
            element_ty=in_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=out_desc,
            global_address=out_ptr,
            load_size=[M_BLOCK, N_BLOCK],
            global_size=[M, N],
            element_ty=out_ptr.dtype.element_ty,
        )
        tl.atomic_xchg(ready_flag, 1, sem="release")
    else:
        # Spin until descriptor is ready
        flag = tl.full([], 0, tl.int32)
        while flag == 0:
            flag = tl.atomic_add(ready_flag, 0, sem="acquire")
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(in_desc)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(out_desc)

    moffset = pid_m * M_BLOCK
    noffset = pid_n * N_BLOCK

    x = tl._experimental_descriptor_load(in_desc, [moffset, noffset], [M_BLOCK, N_BLOCK], in_ptr.dtype.element_ty)
    tl._experimental_descriptor_store(out_desc, x, [moffset, noffset])


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_device_tensormap2d(dtype_str):
    M_BLOCK, N_BLOCK = 32, 64
    M_GRID, N_GRID = 2, 4

    shape = (M_BLOCK * M_GRID, M_BLOCK * N_GRID)
    device = "cuda"
    inp = to_triton(numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str)
    inp_copy = inp.clone()
    out = to_triton(numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str)

    in_desc = torch.randint(0, 256, size=(128, ), dtype=torch.uint8, device="cuda")
    out_desc = torch.randint(0, 256, size=(128, ), dtype=torch.uint8, device="cuda")
    ready_flag = torch.zeros((), dtype=torch.int32, device="cuda")

    device_tensormap_kernel2d[M_GRID, N_GRID](inp, out, in_desc, out_desc, ready_flag, *shape, M_BLOCK=M_BLOCK,
                                              N_BLOCK=N_BLOCK)

    # Check results are correct
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(inp_copy))


@triton.jit
def device_tensormap_kernel1d(in_ptr, out_ptr, in_desc, out_desc, ready_flag, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)

    if pid == 0:
        # Write out descriptor
        tl.extra.cuda.experimental_device_tensormap_create1d(
            desc_ptr=in_desc,
            global_address=in_ptr,
            load_size=BLOCK,
            global_size=numel,
            element_ty=in_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_device_tensormap_create1d(
            desc_ptr=out_desc,
            global_address=out_ptr,
            load_size=BLOCK,
            global_size=numel,
            element_ty=out_ptr.dtype.element_ty,
        )
        tl.atomic_xchg(ready_flag, 1, sem="release")
    else:
        # Spin until descriptor is ready
        flag = tl.full([], 0, tl.int32)
        while flag == 0:
            flag = tl.atomic_add(ready_flag, 0, sem="acquire")
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(in_desc)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(out_desc)

    offset = pid * BLOCK

    x = tl._experimental_descriptor_load(in_desc, [offset], [BLOCK], in_ptr.dtype.element_ty)
    tl._experimental_descriptor_store(out_desc, x, [offset])


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_device_tensormap1d(dtype_str):
    BLOCK = 256
    GRID = 8

    shape = (BLOCK * GRID, )
    device = "cuda"
    inp = to_triton(numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str)
    inp_copy = inp.clone()
    out = to_triton(numpy_random(shape, dtype_str=dtype_str), device=device, dst_type=dtype_str)

    in_desc = torch.randint(0, 256, size=(128, ), dtype=torch.uint8, device="cuda")
    out_desc = torch.randint(0, 256, size=(128, ), dtype=torch.uint8, device="cuda")
    ready_flag = torch.zeros((), dtype=torch.int32, device="cuda")

    device_tensormap_kernel1d[
        1,
    ](inp, out, in_desc, out_desc, ready_flag, *shape, BLOCK=BLOCK)

    # Check results are correct
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))
    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(inp_copy))


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_tensor_descriptor_load(dtype_str):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        desc = tl._experimental_make_tensor_descriptor(
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
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    M, N = 32, 128
    inp = to_triton(numpy_random((M, N), dtype_str), device="cuda", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M_BLOCK, N_BLOCK))

    kernel[(1, )](out, inp, M, N, M_BLOCK, N_BLOCK)

    expect = unwrap_tensor(inp)[1 * M_BLOCK:2 * M_BLOCK, 2 * N_BLOCK:3 * N_BLOCK]
    torch.testing.assert_close(expect, unwrap_tensor(out))


@requires_tma
@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_tensor_descriptor_store(dtype_str):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        midx = moffset + tl.arange(0, M_BLOCK)[:, None]
        nidx = noffset + tl.arange(0, N_BLOCK)[None, :]
        idx = midx * N + nidx

        val = tl.load(a_ptr + idx)

        desc = tl._experimental_make_tensor_descriptor(
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
    inp = to_triton(numpy_random((M, N), dtype_str), device="cuda", dst_type=dtype_str)

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)

    torch.testing.assert_close(unwrap_tensor(inp), unwrap_tensor(out))


@triton.jit(noinline=True)
def tensor_descriptor_in_function_helper(out_ptr, in_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    in_desc = tl._experimental_make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    out_desc = tl._experimental_make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


@requires_tma
def test_tensor_descriptor_in_function():

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        tensor_descriptor_in_function_helper(out_ptr, a_ptr, M, N, M_BLOCK, N_BLOCK)

    M, N = 32, 128
    inp = torch.randn((M, N), device="cuda")

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 2 * 128 * (grid_m * grid_n)
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    expect = inp.abs()
    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(expect, out)


@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
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


@requires_tma
@pytest.mark.parametrize("num_stages", [1, 4])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(32, 32, 32), (128, 64, 64), (128, 128, 64), (128, 256, 64)])
def test_experimental_make_tensor_descriptor_matmul(num_stages, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "cuda"
    M, N, K = 8192, 8192, 1024
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 3 * 128 * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    kernel = matmul_kernel_make_tensor_desciptor[grid](
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
    assert "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned" in kernel.asm["ptx"]
    if BLOCK_M >= 64 and BLOCK_N >= 64:
        assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in kernel.asm["ptx"]


@triton.jit
def kernel_make_tensor_desciptor_loop_carried(a_ptr, M, N, MBLOCK: tl.constexpr, NBLOCK: tl.constexpr):
    # Test that descriptors work with
    pid = tl.program_id(0)
    moffset = MBLOCK * pid

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[MBLOCK, NBLOCK],
    )

    for i in range(0, N, NBLOCK):
        assert isinstance(a_desc, tl._experimental_tensor_descriptor)
        if i % (3 * NBLOCK) == 0:
            a_desc = tl._experimental_make_tensor_descriptor(
                a_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[MBLOCK, NBLOCK],
            )
            assert isinstance(a_desc, tl._experimental_tensor_descriptor)
        assert isinstance(a_desc, tl._experimental_tensor_descriptor)
        a = a_desc.load([moffset, i])
        a_desc.store([moffset, i], a + 10)

    n = 0
    while n < N:
        assert isinstance(a_desc, tl._experimental_tensor_descriptor)
        if n % (3 * NBLOCK) == 0:
            assert isinstance(a_desc, tl._experimental_tensor_descriptor)
            a_desc = tl._experimental_make_tensor_descriptor(
                a_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[MBLOCK, NBLOCK],
            )
        assert isinstance(a_desc, tl._experimental_tensor_descriptor)
        a = a_desc.load([moffset, n])
        a_desc.store([moffset, n], a + 5)

        n += NBLOCK


@requires_tma
def test_experimental_make_tensor_descriptor_loop_carried():
    device = "cuda"
    M, N = 8192, 8192
    torch.manual_seed(42)
    A = torch.randn((M, N), dtype=torch.float32, device=device)
    MBLOCK, NBLOCK = 8, 128
    grid = (triton.cdiv(M, MBLOCK), )

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert size == 128 * grid[0]
        assert align == 128
        assert stream == 0
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    ref_out = A + 15
    kernel = kernel_make_tensor_desciptor_loop_carried[grid](
        A,
        M,
        N,
        MBLOCK,
        NBLOCK,
    )
    torch.testing.assert_close(ref_out, A)
    assert "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned" in kernel.asm["ptx"]
