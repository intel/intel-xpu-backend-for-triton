"""
Regression test for FP8E4M3FN -> FP16 conversion.

Guards the fmul-based converter optimization that replaces the 22-op lookup
table with a single fmul. Validates bit-exact output across all 256 possible
fp8 byte values.

NOTE: This test validates the SOFTWARE fallback converter (used on PVC/Max).
On Xe3P+ with ttig.support_f8_conversion, the SPIR-V builtin path is used
instead and this test does not exercise that path.
"""
import struct
import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_xpu


def expected_fp16_for_byte(byte: int) -> int:
    """
    Reference implementation of the fmul-based converter.

    Returns the fp16 bit pattern (uint16) that the Triton converter produces
    for the given fp8 byte value.

    Algorithm:
    1. Strip sign: u = byte & 0x7F
    2. Shift to fp16 layout: shifted = u << 7
       - fp8: 1 sign + 4 exp + 3 mantissa
       - fp16: 1 sign + 5 exp + 10 mantissa
       - After <<7, bits land at fp16 positions with exp bias still 7
    3. Multiply by 256.0 (2^8) to rebias from 7 to 15
       - For normals: (1 + m/8) * 2^(e-7) * 256 = (1 + m/8) * 2^(e+1)
       - For subnormals: same multiplier works due to mantissa position
    4. Restore sign bit from byte bit 7 to fp16 bit 15

    NOTE: FP8 NaN bytes (0x7f, 0xff) are NOT preserved as fp16 NaN.
    The converter produces ±480.0 (fp16 bits 0x5F80 / 0xDF80).
    """
    # Strip sign, take 7 bits of exp+mantissa
    u = byte & 0x7F
    # Shift to align with fp16 layout
    shifted = (u << 7) & 0xFFFF
    # Bitcast to fp16
    h_in = struct.unpack('<e', struct.pack('<H', shifted))[0]
    # Multiply by 256.0 to rebias exponent from 7 to 15
    mul = np.float16(256.0)
    h_out = float(np.float16(h_in) * mul)
    out_bits = struct.unpack('<H', struct.pack('<e', h_out))[0]
    # Restore sign bit: byte bit 7 -> fp16 bit 15
    sign_bit = (byte & 0x80) << 8
    return (out_bits | sign_bit) & 0xFFFF


@triton.jit
def fp8_to_fp16_kernel(src_ptr, dst_ptr, BLOCK: tl.constexpr):
    """Convert FP8E4M3FN to FP16 via tl.cast."""
    offs = tl.arange(0, BLOCK)
    x = tl.load(src_ptr + offs)
    y = x.to(tl.float16)
    tl.store(dst_ptr + offs, y)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
def test_fp8e4m3fn_to_fp16_all_bytes(device):
    """
    Exhaustive test of all 256 FP8E4M3FN byte values.

    Validates bit-exact conversion to FP16 against the reference
    implementation. The converter uses a single fmul (multiply by 256.0)
    to rebias the exponent, replacing the previous 22-op lookup table.

    This test ensures no regressions in the conversion path.
    """
    # Allocate 256 input bytes covering all possible fp8 values
    src_uint8 = torch.arange(256, dtype=torch.uint8, device=device)
    src = triton.reinterpret(src_uint8, torch.float8_e4m3fn)
    dst = torch.empty(256, dtype=torch.float16, device=device)

    # Launch kernel with BLOCK=256 (single workgroup)
    fp8_to_fp16_kernel[(1, )](src, dst, BLOCK=256)

    # Convert output to uint16 for bit-exact comparison
    dst_bits = dst.view(torch.uint16).cpu().tolist()
    expected_bits = [expected_fp16_for_byte(b) for b in range(256)]

    # Check for mismatches
    mismatches = [(b, e, g) for b, (e, g) in enumerate(zip(expected_bits, dst_bits)) if e != g]

    if mismatches:
        # Format up to 20 mismatches for diagnostics
        msg = '\n'.join(f"  byte=0x{b:02x} expected=0x{e:04x} got=0x{g:04x}" for b, e, g in mismatches[:20])
        total = len(mismatches)
        if total > 20:
            msg += f"\n  ... and {total - 20} more mismatches"
        pytest.fail(f"{total} bytes mismatched (out of 256):\n{msg}")
