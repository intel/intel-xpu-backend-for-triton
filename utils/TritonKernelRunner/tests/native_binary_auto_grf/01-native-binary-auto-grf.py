"""
Native Binary (AOT/zebin) Kernel That Spills
==============================================

Same spill-triggering kernel shape as `tests/auto_grf/01-auto-grf.py`, but
compiled with `generate_native_code=True`. This routes through
`make_zebin`'s own auto-large-GRF retry (measuring real spill via
`extract_spill_size_from_zebin`/`accepts_default_grf`, see compiler.py)
rather than driver.c's JIT retry. Used to check that the dump/TritonKernelRunner
replay also stay aligned with the GRF mode `make_zebin` settles on for a
native binary, not just for JIT SPIR-V.
"""

import torch

import triton
import triton.language as tl
from triton.language.target_info import is_xpu_cri

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def native_binary_auto_grf_kernel(z_ptr, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    a = tl.load(z_ptr + off)
    result = tl.sum(a, axis=0, keep_dims=True)
    tl.store(z_ptr + off, a + result)


# CRI's larger (512-GRF) register file needs a bigger tile to spill; other
# targets already spill at 8K (see test_auto_grf in test_driver.py).
BLOCK = 1024 * 32 if is_xpu_cri() else 1024 * 8
z = torch.empty(BLOCK, dtype=torch.int32, device=DEVICE)
native_binary_auto_grf_kernel[(1, )](z, BLOCK=BLOCK, num_warps=2, generate_native_code=True)
