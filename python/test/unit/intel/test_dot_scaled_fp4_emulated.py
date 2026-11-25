"""mxfp4 x mxfp4 tt.dot_scaled on targets without block-scaled DPAS, which emulate it with bf16 DPAS that decode the
packed operands (see isEmulatedFp4ScaledDot in AccelerateMatmul)."""
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu, is_xpu_cri

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)

pytestmark = pytest.mark.skipif(not is_xpu() or is_xpu_cri(),
                                reason="emulation on XPU targets without block-scaled DPAS")


@triton.jit
def fp4_dot_scaled_kernel(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BLOCK_K: tl.constexpr, DESC: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rs = tl.arange(0, BLOCK_K // 32)
    acc = tl.zeros((M, N), dtype=tl.float32)
    if DESC:
        a_desc = tl.make_tensor_descriptor(a_ptr, [M, K // 2], [K // 2, 1], [M, BLOCK_K // 2])
        b_desc = tl.make_tensor_descriptor(b_ptr, [K // 2, N], [N, 1], [BLOCK_K // 2, N])
    for k in range(0, K, BLOCK_K):
        if DESC:
            a = a_desc.load([0, k // 2])
            b = b_desc.load([k // 2, 0])
        else:
            rk = tl.arange(0, BLOCK_K // 2)
            a = tl.load(a_ptr + rm[:, None] * (K // 2) + k // 2 + rk[None, :])
            b = tl.load(b_ptr + (k // 2 + rk[:, None]) * N + rn[None, :])
        a_scale = tl.load(a_scale_ptr + rm[:, None] * (K // 32) + k // 32 + rs[None, :])
        b_scale = tl.load(b_scale_ptr + rn[:, None] * (K // 32) + k // 32 + rs[None, :])
        acc = tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def _run(a, a_scale, b, b_scale, desc, M, N, K, device, kernel=fp4_dot_scaled_kernel):
    c = torch.empty(M, N, dtype=torch.float32, device=device)
    h = kernel[(1, )](a.to(device), a_scale.to(device), b.to(device), b_scale.to(device), c, M, N, K, BLOCK_K=128,
                      DESC=desc, num_warps=8)
    return c.cpu(), h.asm["ttgir"]


def _decode(packed, dim):
    lo, hi = E2M1[(packed & 0xF).long()], E2M1[(packed >> 4).long()]
    stacked = torch.stack([lo, hi], dim=dim + 1)
    shape = list(packed.shape)
    shape[dim] *= 2
    return stacked.reshape(shape)


@pytest.mark.parametrize("desc", [False, True])
@pytest.mark.parametrize("fast_math", [False, True])
def test_emulated_fp4_dot_scaled(desc, fast_math, device, monkeypatch):
    # Under fast math the two multiplies of a value (by 2^126 to decode it, and by its scale) could be reassociated
    # into one by 2^126 times the scale, which overflows for scales above 2^1 and turned zeros into NaN. Every
    # product and sum here is exact in f32.
    if fast_math:
        monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
    M, N, K = 64, 64, 256
    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8)
    b = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8)
    a_scale = torch.randint(126, 134, (M, K // 32), dtype=torch.uint8)
    b_scale = torch.randint(126, 134, (N, K // 32), dtype=torch.uint8)
    a_f = _decode(a, 1) * torch.exp2(a_scale.double() - 127).repeat_interleave(32, dim=1)
    b_f = _decode(b, 0) * torch.exp2(b_scale.double() - 127).repeat_interleave(32, dim=1).T
    c, ttgir = _run(a, a_scale, b, b_scale, desc, M, N, K, device)
    # The dot is emulated, not decomposed.
    assert "tt.dot_scaled" in ttgir and "ttg.fp4_to_fp" not in ttgir
    torch.testing.assert_close(c, (a_f @ b_f).float(), rtol=0, atol=0)


@pytest.mark.parametrize("scale_byte", [0, 1, 254, 255])
def test_emulated_fp4_dot_scaled_extreme_scales(scale_byte, device, monkeypatch):
    # Scale byte 0 is 2^-127, a denormal, and byte 255 is NaN. The emulation gives the results of the decomposition.
    M, N, K = 64, 64, 256
    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8)
    b = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8)
    a_scale = torch.full((M, K // 32), scale_byte, dtype=torch.uint8)
    # The other operand brings the products back into range, except with byte 255.
    b_scale = torch.full((N, K // 32), 254 - scale_byte if scale_byte < 255 else 127, dtype=torch.uint8)
    emulated, ttgir = _run(a, a_scale, b, b_scale, False, M, N, K, device)
    assert "tt.dot_scaled" in ttgir and "ttg.fp4_to_fp" not in ttgir
    monkeypatch.setenv("TRITON_INTEL_DISABLE_EMULATED_FP4_DPAS", "1")
    # A new JITFunction: the in-memory cache of the first one does not look at the knob.
    decomposed, ttgir = _run(a, a_scale, b, b_scale, False, M, N, K, device, triton.jit(fp4_dot_scaled_kernel.fn))
    assert "ttg.fp4_to_fp" in ttgir
    if scale_byte == 255:
        assert torch.isnan(emulated).all()
    torch.testing.assert_close(emulated, decomposed, rtol=0, atol=0, equal_nan=True)
