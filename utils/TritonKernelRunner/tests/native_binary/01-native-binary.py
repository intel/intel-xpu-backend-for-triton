"""
Native Binary (AOT/zebin) Kernel
=================================

A plain vector-add kernel compiled with `generate_native_code=True`, which
routes through the AOT `make_zebin`/`ocloc` path (`metadata["binary_ext"] =
"zebin"`) instead of JIT SPIR-V. Used to check that the TritonKernelRunner repro
dump correctly records the binary as native (`is_native_binary=True`,
`spv_name` ending in `.zebin`) and that TritonKernelRunner can load and run it via
`ZE_MODULE_FORMAT_NATIVE`.
"""

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def native_binary_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


size = 1024
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output = torch.empty_like(x)
native_binary_kernel[(triton.cdiv(size, 1024), )](x, y, output, size, BLOCK_SIZE=1024, generate_native_code=True)
