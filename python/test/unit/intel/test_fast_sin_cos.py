import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def fast_sin_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.extra.intel.libdevice.fast_sinf(x)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def fast_cos_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.extra.intel.libdevice.fast_cosf(x)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def sin_ep_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.extra.intel.libdevice.sinf_ep(x)
    tl.store(out_ptr + offsets, result, mask=mask)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("N", [256, 1024])
def test_fast_sin(N, device):
    """Test fast_sinf with graphics-grade precision."""
    BLOCK_SIZE = 128
    # Input range [-pi, pi]
    x = torch.linspace(-np.pi, np.pi, N, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    # Launch kernel
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    fast_sin_kernel[grid](x, out, N, BLOCK_SIZE)

    # Reference (via float64 for better precision)
    ref = torch.sin(x.to(torch.float64)).to(torch.float32)

    # Graphics-grade tolerance: max abs error <= 1e-3
    max_err = torch.max(torch.abs(out - ref)).item()
    assert max_err <= 1e-3, f"Max absolute error {max_err} exceeds tolerance 1e-3"


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("N", [256, 1024])
def test_sinf_ep(N, device):
    """Test __imf_sinf_ep (IMF enhanced-performance sin tier)."""
    BLOCK_SIZE = 128
    x = torch.linspace(-np.pi, np.pi, N, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    compiled = sin_ep_kernel[grid](x, out, N, BLOCK_SIZE)

    # IGC inlines IMF entry points, so __imf_sinf_ep itself does not survive.
    # The inlined internal implementation `__imf_impl_sin_s_ep` does, and the
    # `_ep` suffix proves we hit the enhanced-performance tier (not _ha / _la).
    llir = compiled.asm["llir"]
    assert "__imf_impl_sin_s_ep" in llir, \
        "Expected inlined __imf_sinf_ep (look for __imf_impl_sin_s_ep) in LLVM IR"

    ref = torch.sin(x.to(torch.float64)).to(torch.float32)
    max_err = torch.max(torch.abs(out - ref)).item()
    # _ep tier is the loosest IMF accuracy tier; 1e-3 is comfortable.
    assert max_err <= 1e-3, f"Max absolute error {max_err} exceeds tolerance 1e-3"


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("N", [256, 1024])
def test_fast_cos(N, device):
    """Test fast_cosf with graphics-grade precision."""
    BLOCK_SIZE = 128
    # Input range [-pi, pi]
    x = torch.linspace(-np.pi, np.pi, N, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    # Launch kernel
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    fast_cos_kernel[grid](x, out, N, BLOCK_SIZE)

    # Reference (via float64 for better precision)
    ref = torch.cos(x.to(torch.float64)).to(torch.float32)

    # Graphics-grade tolerance: max abs error <= 1e-3
    max_err = torch.max(torch.abs(out - ref)).item()
    assert max_err <= 1e-3, f"Max absolute error {max_err} exceeds tolerance 1e-3"
