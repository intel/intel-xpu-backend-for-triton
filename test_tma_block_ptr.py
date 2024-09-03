import torch
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


x = torch.randn(SIZE, dtype=torch.float32, device=device)
z_tri = torch.empty_like(x)

if BYVAL_TMA:
    desc = triton.tools.experimental_descriptor.create_1d_tma_descriptor(x.data_ptr(), SIZE, SIZE, x.element_size())
else:
    desc = create_tma_desc_gmem_ptr(x.data_ptr(), [SIZE], [SIZE], x.element_size())

tma_copy_kernel[(1, )](z_tri, desc, SIZE=SIZE)
assert torch.equal(x.cpu(), z_tri.cpu())

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
    pass
