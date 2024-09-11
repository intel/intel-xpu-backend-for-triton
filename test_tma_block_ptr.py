import torch
import intel_extension_for_pytorch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

device = "xpu"
SIZE = 128
BYVAL_TMA = True


def create_tma_desc_gmem_ptr(ptr, dims, block_dims, element_size):
    cpu_desc = torch.empty(128, device="cpu", dtype=torch.int8)
    if len(dims) == 1:
        triton.runtime.driver.active.utils.fill_1d_tma_descriptor(ptr, dims[0], block_dims[0], element_size,
                                                                  cpu_desc.data_ptr())
    else:
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(ptr, dims[0], dims[1], block_dims[0], block_dims[1],
                                                                  element_size, cpu_desc.data_ptr())
    return cpu_desc.xpu()


@triton.jit
def tma_copy_kernel(Z, desc, SIZE: tl.constexpr):
    off_desc = 0
    off = tl.arange(0, SIZE)
    x = tl._experimental_descriptor_load(desc, [off_desc], [SIZE], Z.dtype.element_ty)
    tl.store(Z + off, x)


def test_1d_tma():
    x = torch.randn(SIZE, dtype=torch.float32, device=device)
    z_tri = torch.empty_like(x)

    if BYVAL_TMA:
        desc = triton.tools.experimental_descriptor.create_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size())
    else:
        desc = create_tma_desc_gmem_ptr(x.data_ptr(), [SIZE], [SIZE], x.element_size())

    tma_copy_kernel[(1, )](z_tri, desc, SIZE=SIZE)
    assert torch.equal(x.cpu(), z_tri.cpu())


@triton.jit
def matmul_kernel_tma_persistent(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                                 # Matrix dimensions
                                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr  #
                                 ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for offs_k in range(0, K, BLOCK_SIZE_K):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16)
        accumulator = tl.dot(a, b, accumulator)

    c = accumulator.to(tl.float16)

    tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])


def matmul_tma_persistent(a, b):
    # Autotuner does not work with TMA. Use manual config.
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 32
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 16, "GROUP_SIZE_M": 4, "num_stages": 3,
            "num_warps": 32
        }
    }

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.zeros((M, N), device=a.device, dtype=dtype)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), M, K,
                                                                           configs[dtype]["BLOCK_SIZE_M"],
                                                                           configs[dtype]["BLOCK_SIZE_K"],
                                                                           a.element_size())
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), K, N,
                                                                           configs[dtype]["BLOCK_SIZE_K"],
                                                                           configs[dtype]["BLOCK_SIZE_N"],
                                                                           b.element_size())
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(c.data_ptr(), M, N,
                                                                           configs[dtype]["BLOCK_SIZE_M"],
                                                                           configs[dtype]["BLOCK_SIZE_N"],
                                                                           c.element_size())

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_tma_persistent[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
        grf_mode='large')
    return c


def test_2d_tma():
    M, N, K = 1024, 1024, 1024

    a = torch.rand((M, K), device='xpu', dtype=torch.float16)
    b = torch.rand((K, N), device='xpu', dtype=torch.float16)
    # a = torch.arange(0, (M*K), device='xpu').to(dtype=torch.float16) * 0.001
    # a = a.view(M, K)
    # b = torch.arange(0, (K*N), device='xpu').to(dtype=torch.float16) * 0.001
    # b = b.view(K, N)

    triton_fn = lambda: matmul_tma_persistent(a, b)
    rtol = 1e-2 if a.dtype == torch.float16 else 1e-3
    tri_c = triton_fn()
    torch_c = torch.matmul(a, b)

    triton.testing.assert_close(tri_c.cpu(), torch_c.cpu(), atol=0.02, rtol=rtol)


# @triton.jit
# def block_copy_kernel(Z, x_ptr, N, SIZE: tl.constexpr):
#     off_desc = 0
#     off = tl.arange(0, SIZE)
#     x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(N, ), strides=(1, ), offsets=(off_desc, ),
#                                     block_shape=(SIZE, ), order=(0, ))
#     x = tl.load(x_block_ptr, boundary_check=(0, ))
#     tl.store(Z + off, x)
#
# x = torch.randn(SIZE, dtype=torch.float32, device=device)
# z_tri = torch.empty_like(x)
# block_copy_kernel[(1, )](z_tri, x, N=SIZE, SIZE=SIZE)

if __name__ == "__main__":
    test_1d_tma()
    test_2d_tma()
