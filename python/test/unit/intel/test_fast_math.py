import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
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
def fp8_upcast_times_scale_kernel(x_ptr, out_ptr, scale, N, DST: tl.constexpr, CLEAR_SIGN: tl.constexpr,
                                  BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    if CLEAR_SIGN:
        # A sign bit known to be zero lets LLVM fold away integer operations on it.
        x = (x.to(tl.uint8, bitcast=True) & 0x7F).to(x.dtype, bitcast=True)
    if DST == "fp16":
        y = x.to(tl.float16) * scale.to(tl.float16)
    elif DST == "bf16":
        y = x.to(tl.bfloat16) * scale.to(tl.bfloat16)
    else:
        y = x.to(tl.float32) * scale
    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("clear_sign", [False, True])
@pytest.mark.parametrize("scale_exp", [0, 7, 8, 12, 15])
@pytest.mark.parametrize("dst", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("src", ["e4m3", "e5m2"])
def test_fp8_upcast_times_scale_fast_math(src, dst, scale_exp, clear_sign, device, monkeypatch):
    # Under fast math every FP instruction gets `fast`, so the rebias multiply of an fp8 upcast could be reassociated
    # with a multiply of its result: `x.to(f16) * s` as `x * (256 * s)`, which overflows fp16 for s >= 256 and turned
    # every value, zeros included, into inf or NaN. The products themselves stay finite here.
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    src_dtype = torch.float8_e4m3fn if src == "e4m3" else torch.float8_e5m2
    dst_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dst]
    every_byte = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(src_dtype).float()
    values = every_byte[torch.isfinite(every_byte) & (every_byte.abs() <= 1)]
    scale = 2.0**scale_exp
    out = torch.empty(values.numel(), dtype=torch.float32, device=device)
    fp8_upcast_times_scale_kernel[(1, )](values.to(src_dtype).to(device), out, scale, values.numel(), dst, clear_sign,
                                         BLOCK=256)
    if clear_sign:
        values = values.abs()
    ref = (values.to(dst_dtype) * torch.tensor(scale, dtype=dst_dtype)).float()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@gluon.jit
def gluon_fp8_upcast_times_scale_kernel(x_ptr, out_ptr, scale, N, DST: ttgl.constexpr):
    layout: ttgl.constexpr = ttgl.BlockedLayout([8], [32], [1], [0])
    offsets = ttgl.arange(0, 256, layout=layout)
    mask = offsets < N
    x = ttgl.load(x_ptr + offsets, mask=mask)
    y = x.to(DST) * scale.to(DST)
    ttgl.store(out_ptr + offsets, y.to(ttgl.float32), mask=mask)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("scale_exp", [8, 15])
def test_gluon_fp8_upcast_times_scale_fast_math(scale_exp, device, monkeypatch):
    # Gluon kernels skip `make_ttgir`, which annotates the module with fast math, but get fast math all the same.
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    every_byte = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    values = every_byte[torch.isfinite(every_byte) & (every_byte.abs() <= 1)]
    # Every lane active (N == 256): with this Gluon kernel, a block of 114 active lanes is not rewritten into
    # `x * (256 * s)`.
    values = values.repeat(3)[:256]
    scale = 2.0**scale_exp
    out = torch.empty(values.numel(), dtype=torch.float32, device=device)
    gluon_fp8_upcast_times_scale_kernel[(1, )](values.to(torch.float8_e4m3fn).to(device), out, scale, values.numel(),
                                               ttgl.float16, num_warps=1)
    ref = (values.to(torch.float16) * torch.tensor(scale, dtype=torch.float16)).float()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@triton.jit
def fp8_range_limited_upcast_times_scale_kernel(raw_ptr, out_ptr, scale, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    # Exponent and mantissa limited to values of magnitude <= 1: LLVM can tell that no byte is a reserved one.
    raw = tl.load(raw_ptr + offsets) & 0x38
    x = raw.to(tl.float8e4nv, bitcast=True)
    tl.store(out_ptr + offsets, (x.to(tl.float16) * scale.to(tl.float16)).to(tl.float32))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
def test_fp8_range_limited_upcast_times_scale_fast_math(device, monkeypatch):
    # The separation of the rebias multiply from the multiply by the scale must not depend on what LLVM can prove
    # about the input.
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    out = torch.empty(256, dtype=torch.float32, device=device)
    fp8_range_limited_upcast_times_scale_kernel[(1, )](raw.to(device), out, 256.0, BLOCK=256)
    values = (raw & 0x38).view(torch.float8_e4m3fn).to(torch.float16)
    ref = (values * torch.tensor(256.0, dtype=torch.float16)).float()
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@triton.jit
def fp8_uniform_upcast_times_scale_kernel(x_ptr, out_ptr, scale, BLOCK: tl.constexpr, SCALAR: tl.constexpr):
    pid = tl.program_id(0)
    if SCALAR:
        # A scalar upcast.
        x = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, (x.to(tl.float16) * scale.to(tl.float16)).to(tl.float32))
    else:
        # A tensor loaded from a single address: every lane holds the same value.
        offsets = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + pid + offsets * 0)
        y = x.to(tl.float16) * scale.to(tl.float16)
        tl.store(out_ptr + pid * BLOCK + offsets, y.to(tl.float32))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("scale_exp", [8, 15])
def test_fp8_uniform_upcast_times_scale_fast_math(scale_exp, scalar, device, monkeypatch):
    # Values that IGC can keep uniform (one per sub-group) go through the separation as well.
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    every_byte = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    values = every_byte[torch.isfinite(every_byte) & (every_byte.abs() <= 1)]
    scale = 2.0**scale_exp
    block = 16
    out = torch.empty(values.numel() * (1 if scalar else block), dtype=torch.float32, device=device)
    fp8_uniform_upcast_times_scale_kernel[(values.numel(), )](values.to(torch.float8_e4m3fn).to(device), out, scale,
                                                              BLOCK=block, SCALAR=scalar)
    ref = (values.to(torch.float16) * torch.tensor(scale, dtype=torch.float16)).float()
    if not scalar:
        ref = ref.repeat_interleave(block)
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@triton.jit
def fp16_e4m3_dot_scaled_kernel(a_ptr, b_ptr, b_scale_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    b_scale = tl.load(b_scale_ptr + rn[:, None] * (K // 32) + tl.arange(0, K // 32)[None, :])
    c = tl.dot_scaled(a, None, "fp16", b, b_scale, "e4m3")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("scale_exp", [8, 15])
def test_fp16_e4m3_dot_scaled_fast_math(scale_exp, device, monkeypatch):
    # tl.dot_scaled of fp16 x e4m3 upcasts the e4m3 operand to fp16 and multiplies it by its scale in fp16. Every
    # product and sum here is exact in f32.
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    M, N, K = 32, 32, 64
    torch.manual_seed(0)
    a = torch.randint(-2, 3, (M, K)).to(torch.float16)
    every_byte = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
    values = every_byte[torch.isfinite(every_byte) & (every_byte.abs() <= 1) & (every_byte.abs() >= 2.0**-3)]
    b = values[torch.randint(0, values.numel(), (K, N))].to(torch.float8_e4m3fn)
    b_scale = torch.full((N, K // 32), 127 + scale_exp, dtype=torch.uint8)
    c = torch.empty(M, N, dtype=torch.float32, device=device)
    fp16_e4m3_dot_scaled_kernel[(1, )](a.to(device), b.to(device), b_scale.to(device), c, M, N, K)
    ref = a.double() @ (b.double() * 2.0**scale_exp)
    torch.testing.assert_close(c.cpu(), ref.float(), rtol=0, atol=0)
