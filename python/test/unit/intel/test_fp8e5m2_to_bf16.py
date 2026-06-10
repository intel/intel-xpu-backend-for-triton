"""FP8E5M2 -> BF16 cast correctness test.

Tests the FP8E5M2 -> BF16 converter across all 256 possible fp8 byte values
and asserts bit-exact equality with the ground truth derived from the current
converter implementation (Fp8E5M2_to_Bf16 in ElementwiseOpToLLVM.cpp).

Note: The current converter does NOT preserve IEEE NaN or infinity encodings
for bytes 0x7C-0x7F and 0xFC-0xFF — it returns finite values. This is the
existing behavior and is preserved bit-exactly.
"""
import torch
import pytest
import triton
import triton.language as tl
from triton._internal_testing import is_xpu

# Ground truth: 256 BF16 bit patterns from the current converter (Fp8E5M2_to_Bf16).
# Generated via transliteration of the C++ implementation.
EXPECTED_BF16_BITS = [
    0x0000, 0x3780, 0x3800, 0x3840, 0x3880, 0x38a0, 0x38c0, 0x38e0, 0x3900, 0x3920, 0x3940, 0x3960, 0x3980, 0x39a0,
    0x39c0, 0x39e0,  # 0x00-0x0f
    0x3a00, 0x3a20, 0x3a40, 0x3a60, 0x3a80, 0x3aa0, 0x3ac0, 0x3ae0, 0x3b00, 0x3b20, 0x3b40, 0x3b60, 0x3b80, 0x3ba0,
    0x3bc0, 0x3be0,  # 0x10-0x1f
    0x3c00, 0x3c20, 0x3c40, 0x3c60, 0x3c80, 0x3ca0, 0x3cc0, 0x3ce0, 0x3d00, 0x3d20, 0x3d40, 0x3d60, 0x3d80, 0x3da0,
    0x3dc0, 0x3de0,  # 0x20-0x2f
    0x3e00, 0x3e20, 0x3e40, 0x3e60, 0x3e80, 0x3ea0, 0x3ec0, 0x3ee0, 0x3f00, 0x3f20, 0x3f40, 0x3f60, 0x3f80, 0x3fa0,
    0x3fc0, 0x3fe0,  # 0x30-0x3f
    0x4000, 0x4020, 0x4040, 0x4060, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4120, 0x4140, 0x4160, 0x4180, 0x41a0,
    0x41c0, 0x41e0,  # 0x40-0x4f
    0x4200, 0x4220, 0x4240, 0x4260, 0x4280, 0x42a0, 0x42c0, 0x42e0, 0x4300, 0x4320, 0x4340, 0x4360, 0x4380, 0x43a0,
    0x43c0, 0x43e0,  # 0x50-0x5f
    0x4400, 0x4420, 0x4440, 0x4460, 0x4480, 0x44a0, 0x44c0, 0x44e0, 0x4500, 0x4520, 0x4540, 0x4560, 0x4580, 0x45a0,
    0x45c0, 0x45e0,  # 0x60-0x6f
    0x4600, 0x4620, 0x4640, 0x4660, 0x4680, 0x46a0, 0x46c0, 0x46e0, 0x4700, 0x4720, 0x4740, 0x4760, 0x4780, 0x47a0,
    0x47c0, 0x47e0,  # 0x70-0x7f
    0x8000, 0xb780, 0xb800, 0xb840, 0xb880, 0xb8a0, 0xb8c0, 0xb8e0, 0xb900, 0xb920, 0xb940, 0xb960, 0xb980, 0xb9a0,
    0xb9c0, 0xb9e0,  # 0x80-0x8f
    0xba00, 0xba20, 0xba40, 0xba60, 0xba80, 0xbaa0, 0xbac0, 0xbae0, 0xbb00, 0xbb20, 0xbb40, 0xbb60, 0xbb80, 0xbba0,
    0xbbc0, 0xbbe0,  # 0x90-0x9f
    0xbc00, 0xbc20, 0xbc40, 0xbc60, 0xbc80, 0xbca0, 0xbcc0, 0xbce0, 0xbd00, 0xbd20, 0xbd40, 0xbd60, 0xbd80, 0xbda0,
    0xbdc0, 0xbde0,  # 0xa0-0xaf
    0xbe00, 0xbe20, 0xbe40, 0xbe60, 0xbe80, 0xbea0, 0xbec0, 0xbee0, 0xbf00, 0xbf20, 0xbf40, 0xbf60, 0xbf80, 0xbfa0,
    0xbfc0, 0xbfe0,  # 0xb0-0xbf
    0xc000, 0xc020, 0xc040, 0xc060, 0xc080, 0xc0a0, 0xc0c0, 0xc0e0, 0xc100, 0xc120, 0xc140, 0xc160, 0xc180, 0xc1a0,
    0xc1c0, 0xc1e0,  # 0xc0-0xcf
    0xc200, 0xc220, 0xc240, 0xc260, 0xc280, 0xc2a0, 0xc2c0, 0xc2e0, 0xc300, 0xc320, 0xc340, 0xc360, 0xc380, 0xc3a0,
    0xc3c0, 0xc3e0,  # 0xd0-0xdf
    0xc400, 0xc420, 0xc440, 0xc460, 0xc480, 0xc4a0, 0xc4c0, 0xc4e0, 0xc500, 0xc520, 0xc540, 0xc560, 0xc580, 0xc5a0,
    0xc5c0, 0xc5e0,  # 0xe0-0xef
    0xc600, 0xc620, 0xc640, 0xc660, 0xc680, 0xc6a0, 0xc6c0, 0xc6e0, 0xc700, 0xc720, 0xc740, 0xc760, 0xc780, 0xc7a0,
    0xc7c0, 0xc7e0,  # 0xf0-0xff
]


@triton.jit
def type_convert(src, dst, BLOCK_SIZE: tl.constexpr):
    """Load FP8E5M2, cast to BF16, store."""
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + idxs)
    y = x.to(dst.dtype.element_ty)
    tl.store(dst + idxs, y)


@pytest.mark.skipif(not is_xpu(), reason="FP8E5M2 tests are XPU-specific")
def test_fp8e5m2_to_bf16_bit_exact(device):
    """Test FP8E5M2 -> BF16 cast for all 256 byte values.

    Validates bit-exact equality with the ground truth from the current
    converter implementation (Fp8E5M2_to_Bf16 in ElementwiseOpToLLVM.cpp).

    This includes the subnormal table (bytes 0x01-0x03, 0x81-0x83), infinity
    encodings (0x7C, 0xFC), and NaN encodings (0x7D-0x7F, 0xFD-0xFF) which the
    current converter collapses to finite values.
    """
    SIZE = 256
    BLOCK_SIZE = 256

    # Construct all 256 fp8 byte values via triton.reinterpret (preserves bits)
    src = torch.arange(0, SIZE, dtype=torch.uint8, device=device)
    src_fp8 = triton.reinterpret(src, torch.float8_e5m2)
    dst = torch.empty(SIZE, dtype=torch.bfloat16, device=device)

    # Launch kernel
    type_convert[(SIZE // BLOCK_SIZE, )](src_fp8, dst, BLOCK_SIZE)

    # Primary assertion: bit-exact equality for all 256 bytes
    expected_uint16 = torch.tensor(EXPECTED_BF16_BITS, dtype=torch.uint16, device=device)
    dst_uint16 = dst.view(torch.uint16)

    if not torch.equal(dst_uint16, expected_uint16):
        # Detailed failure report
        mismatch_mask = dst_uint16 != expected_uint16
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).squeeze(1)
        failures = []
        for idx in mismatch_indices[:20]:  # Report up to 20 failures
            byte = idx.item()
            expected = expected_uint16[byte].item()
            actual = dst_uint16[byte].item()
            failures.append(f"byte=0x{byte:02x} expected=0x{expected:04x} actual=0x{actual:04x}")
        msg = f"{len(mismatch_indices)} byte(s) mismatch:\n  " + "\n  ".join(failures)
        if len(mismatch_indices) > 20:
            msg += f"\n  ... and {len(mismatch_indices) - 20} more"
        pytest.fail(msg)

    # Subnormal-table sub-asserts: validate each subnormal byte individually
    # for easier debugging if a single subnormal value fails.
    # E5M2 has 3 non-zero positive subnormals (0x01-0x03) and 3 negative (0x81-0x83).
    subnormals = list(range(0x01, 0x04)) + list(range(0x81, 0x84))
    for byte in subnormals:
        expected = expected_uint16[byte].item()
        actual = dst_uint16[byte].item()
        assert actual == expected, \
            f"Subnormal byte 0x{byte:02x}: expected=0x{expected:04x} actual=0x{actual:04x}"

    # NaN sub-asserts: E5M2 has multiple NaN encodings (0x7D-0x7F, 0xFD-0xFF).
    # The current converter does NOT preserve IEEE NaN — it produces finite values.
    # We assert against the actual converter output, not the ideal IEEE NaN.
    nan_bytes = [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]
    for byte in nan_bytes:
        expected = expected_uint16[byte].item()
        actual = dst_uint16[byte].item()
        assert actual == expected, \
            f"NaN byte 0x{byte:02x}: expected=0x{expected:04x} actual=0x{actual:04x}"

    # Infinity sub-asserts: E5M2 has inf encodings (0x7C, 0xFC).
    # The current converter does NOT preserve IEEE infinity — it produces finite values.
    inf_bytes = [0x7C, 0xFC]
    for byte in inf_bytes:
        expected = expected_uint16[byte].item()
        actual = dst_uint16[byte].item()
        assert actual == expected, \
            f"Inf byte 0x{byte:02x}: expected=0x{expected:04x} actual=0x{actual:04x}"

    # Sanity check: validate a few normal values explicitly
    normal_samples = [0x04, 0x40, 0x60, 0x84, 0xC0, 0xE0]
    for byte in normal_samples:
        expected = expected_uint16[byte].item()
        actual = dst_uint16[byte].item()
        assert actual == expected, \
            f"Normal byte 0x{byte:02x}: expected=0x{expected:04x} actual=0x{actual:04x}"
