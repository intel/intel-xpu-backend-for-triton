"""f32 -> bf16 RTNE conversions use the 8-wide `__spirv_ConvertFToBF16INTEL` when the device has `cl_intel_bfloat16_conversions`.

https://github.com/intel/intel-xpu-backend-for-triton/issues/8594

IGC keeps scalar conversion calls scalar, which raises register pressure in bf16 epilogues.
"""
import numpy as np
import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                                reason="Intel XPU device not available")

VECTOR_CALL = "_Z27__spirv_ConvertFToBF16INTELDv8_f("


@triton.jit
def _to_bf16(X, Y, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + offs, tl.load(X + offs).to(tl.bfloat16))


def _rtne_bf16_bits(x):
    """Independent RTNE f32 -> bf16 on the integer bits (non-NaN inputs)."""
    u = x.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def test_vector_conversion(device):
    edges = np.array([
        0x00000000, 0x80000000, 0x3F808000, 0x3F818000, 0xBF808000, 0x3F808001, 0x3F807FFF, 0x00000001, 0x007FFFFF,
        0x00800000, 0x7F7FFFFF, 0x7F7F7FFF, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFA00000
    ], dtype=np.uint32)
    # Distinct expected results, so a lane permutation changes the output: round-down and ties give the even
    # upper half, round-up gives the odd one.
    i = np.arange(4096 - edges.size, dtype=np.uint32)
    body = ((0x3C00 + 2 * i) << 16) | np.array([0x0000, 0x7FFF, 0x8000, 0x8001], dtype=np.uint32)[i % 4]
    x = torch.from_numpy(np.concatenate([edges, body]).view(np.float32)).to(device)
    assert x.numel() % 2048 == 0  # the kernel is unmasked
    y = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
    # 2048 elements over 4 warps: 16 per thread, so full groups of 8 exist.
    k = _to_bf16[(x.numel() // 2048, )](x, y, BLOCK=2048, num_warps=4)
    if '"ttig.support_bfloat16_conversion"' not in k.asm["ttgir"]:
        pytest.skip("the device lacks cl_intel_bfloat16_conversions")
    assert VECTOR_CALL in k.asm["llir"]
    xs, got = x.cpu().numpy(), y.cpu().view(torch.int16).numpy().view(np.uint16)
    nan = np.isnan(xs)
    np.testing.assert_array_equal(got[~nan], _rtne_bf16_bits(xs[~nan]))
    g = got[nan]
    assert ((g & 0x7F80) == 0x7F80).all() and ((g & 0x007F) != 0).all()  # still NaN
    assert ((g >> 15) == (xs[nan].view(np.uint32) >> 31)).all()  # sign kept
