"""Bit-exact check of ttg.fp4_to_fp for all 256 packed e2m1 bytes.

On targets without the e2m1 conversion builtins, the lowering decodes each nibble with integer operations and a
multiplication by 2^126, which takes 0.5 (nibbles 0x1 and 0x9) through an f32 denormal and must keep the sign of
-0.0 (nibble 0x8). Numerical GEMM comparisons do not tell these apart, so the decoded bits are compared with an
independent table of the bf16 and fp16 encodings of the 16 e2m1 values, for both packing axes.
"""
import pytest
import torch

from triton._internal_testing import is_xpu
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

pytestmark = pytest.mark.skipif(not is_xpu(), reason="Intel XPU test")

# Encodings of the e2m1 values 0, 0.5, 1, 1.5, 2, 3, 4, 6; nibbles 0x8-0xf are the same with the sign bit set.
BF16_BITS = [0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0]
FP16_BITS = [0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600]


def expected_bits(magnitudes):
    return [bits | (0x8000 if nibble >= 8 else 0) for nibble in range(16) for bits in [magnitudes[nibble % 8]]]


@gluon.jit
def fp4_to_fp_kernel(src, dst, OUT_DTYPE: ttgl.constexpr, AXIS: ttgl.constexpr):
    layout: ttgl.constexpr = ttgl.BlockedLayout([2, 2], [4, 8], [1, 1], [1, 0])
    rows = ttgl.arange(0, 16, layout=ttgl.SliceLayout(1, layout))
    cols = ttgl.arange(0, 16, layout=ttgl.SliceLayout(0, layout))
    x = ttgl.load(src + rows[:, None] * 16 + cols[None, :])
    y = ttgl.fp4_to_fp(x, OUT_DTYPE, AXIS)
    out_layout: ttgl.constexpr = y.type.layout
    OUT_ROWS: ttgl.constexpr = 32 if AXIS == 0 else 16
    OUT_COLS: ttgl.constexpr = 32 if AXIS == 1 else 16
    out_rows = ttgl.arange(0, OUT_ROWS, layout=ttgl.SliceLayout(1, out_layout))
    out_cols = ttgl.arange(0, OUT_COLS, layout=ttgl.SliceLayout(0, out_layout))
    ttgl.store(dst + out_rows[:, None] * OUT_COLS + out_cols[None, :], y)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float16"])
def test_fp4_to_fp_all_bytes(out_dtype, axis, device):
    torch_dtype = getattr(torch, out_dtype)
    table = torch.tensor(expected_bits(BF16_BITS if out_dtype == "bfloat16" else FP16_BITS), dtype=torch.int32)

    src = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(16, 16)
    # The low nibble holds the first element along the packing axis, the high nibble the second.
    lo, hi = table[(src & 0xF).long()], table[(src >> 4).long()]
    expected = torch.stack([lo, hi], dim=axis + 1).reshape((32, 16) if axis == 0 else (16, 32))

    out = torch.empty(expected.shape, dtype=torch_dtype, device=device)
    fp4_to_fp_kernel[(1, )](src.to(device), out, getattr(ttgl, out_dtype), axis, num_warps=1)

    actual = out.cpu().view(torch.int16).to(torch.int32) & 0xFFFF
    mismatches = (actual != expected).nonzero().tolist()
    assert not mismatches, f"{len(mismatches)} mismatched values, first at {mismatches[0]}: " \
                           f"{actual[tuple(mismatches[0])]:#06x} != {expected[tuple(mismatches[0])]:#06x}"
