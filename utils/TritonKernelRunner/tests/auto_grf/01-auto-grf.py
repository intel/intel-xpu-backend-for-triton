"""
Auto Large-GRF Kernel
======================

A kernel sized to spill past `REBUILD_SPILL_BYTES_PER_THREAD`, which makes
`load_binary` retry at large GRF mode and rewrite `build_flags` with the
`-cl-intel-<256|512>-GRF-per-thread` flag it settled on. Used to check that
the TritonKernelRunner repro dump (`args_data.json`) reflects that retried value
rather than the pre-retry one.
"""

import torch

import triton
import triton.language as tl
from triton.language.target_info import is_xpu_cri

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def auto_grf_kernel(z_ptr, BLOCK: tl.constexpr):
    # make it hard to re-schedule.
    off = tl.arange(0, BLOCK)
    a = tl.load(z_ptr + off)
    result = tl.sum(a, axis=0, keep_dims=True)
    tl.store(z_ptr + off, a + result)


# CRI's larger (512-GRF) register file needs a bigger tile to spill; other
# targets already spill at 8K (see test_auto_grf in test_driver.py).
BLOCK = 1024 * 32 if is_xpu_cri() else 1024 * 8
z = torch.empty(BLOCK, dtype=torch.int32, device=DEVICE)
auto_grf_kernel[(1, )](z, BLOCK=BLOCK, num_warps=2)
