import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.language.extra import libdevice


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

    # Reference (via float64 for better precision). Cast on CPU: XPU devices without
    # fp64 hardware support (e.g. A770) would otherwise raise "Required aspect fp64
    # is not supported on the device".
    ref = torch.sin(x.cpu().to(torch.float64)).to(torch.float32).to(device)

    # Graphics-grade tolerance: max abs error <= 1e-3
    max_err = torch.max(torch.abs(out - ref)).item()
    assert max_err <= 1e-3, f"Max absolute error {max_err} exceeds tolerance 1e-3"


@triton.jit
def exp2_kernel(x_ptr, y_ptr, SIZE: tl.constexpr):
    off = tl.arange(0, SIZE)
    x = tl.load(x_ptr + off)
    y = libdevice.exp2(x)
    tl.store(y_ptr + off, y)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("dtype_str", ["float16", "float32"])
def test_exp2_native_lowering(dtype_str, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)

    x = torch.randn((SIZE, ), dtype=dtype, device=device)
    y = torch.empty((SIZE, ), dtype=dtype, device=device)
    # Reference via float64 for better precision. Cast on CPU: XPU devices without fp64
    # hardware support (e.g. A770) would otherwise raise "Required aspect fp64 is not
    # supported on the device".
    ref = torch.exp2(x.cpu().to(torch.float64)).to(dtype).to(device)

    h = exp2_kernel[(1, )](x, y, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(ref, y, equal_nan=True)

    llir = h.asm["llir"]
    llvm_suffix = {"float16": "f16", "float32": "f32"}[dtype_str]
    assert f"llvm.exp2.{llvm_suffix}" in llir
    assert "__imf_exp2" not in llir


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

    # Reference (via float64 for better precision). Cast on CPU: XPU devices without
    # fp64 hardware support (e.g. A770) would otherwise raise "Required aspect fp64
    # is not supported on the device".
    ref = torch.cos(x.cpu().to(torch.float64)).to(torch.float32).to(device)

    # Graphics-grade tolerance: max abs error <= 1e-3
    max_err = torch.max(torch.abs(out - ref)).item()
    assert max_err <= 1e-3, f"Max absolute error {max_err} exceeds tolerance 1e-3"
