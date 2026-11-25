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


@triton.jit
def fp4_dot_scaled_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K // 2)
    rs = tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * (K // 2) + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    a_scale = tl.load(a_scale_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b_scale = tl.load(b_scale_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
def test_fp4_dot_scaled_fast_math(device, monkeypatch):
    # Without fp4 DPAS, each e2m1 value is decoded with a multiply by 2^126 and then multiplied by its scale. Under
    # fast math the two multiplies could be reassociated into one by 2^126 times the scale, which overflows for scales
    # above 2^1 and turned zeros into NaN. Every product and sum here is exact in f32.
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    M, N, K = 64, 64, 128
    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8)
    b = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8)
    a_scale = torch.randint(129, 134, (M, K // 32), dtype=torch.uint8)
    b_scale = torch.randint(129, 134, (N, K // 32), dtype=torch.uint8)
    # The low nibble holds the first element along K.
    a_f = torch.stack([e2m1[(a & 0xF).long()], e2m1[(a >> 4).long()]], dim=2).reshape(M, K)
    b_f = torch.stack([e2m1[(b & 0xF).long()], e2m1[(b >> 4).long()]], dim=1).reshape(K, N)
    a_f = a_f * torch.exp2(a_scale.double() - 127).repeat_interleave(32, dim=1)
    b_f = b_f * torch.exp2(b_scale.double() - 127).repeat_interleave(32, dim=1).T
    c = torch.empty(M, N, dtype=torch.float32, device=device)
    fp4_dot_scaled_kernel[(1, )](a.to(device), a_scale.to(device), b.to(device), b_scale.to(device), c, M, N, K)
    torch.testing.assert_close(c.cpu(), (a_f @ b_f).float(), rtol=0, atol=0)
