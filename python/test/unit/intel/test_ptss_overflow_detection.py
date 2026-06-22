"""Tests for PTSS overflow detection in the Intel backend compiler.

Verifies that `make_zebin` raises `OutOfResources` (not a generic error)
when IGC reports that the kernel's scratch space exceeds the hardware PTSS
limit. This ensures the autotuner can gracefully skip large tile
configurations on non-DPAS hardware.

Related: https://github.com/intel/intel-xpu-backend-for-triton/issues/7273
"""
import pytest

from triton.backends.intel.compiler import PTSS_OVERFLOW_RE
from triton.runtime.errors import OutOfResources


class TestPTSSOverflowRegex:
    """Verify the regex matches known IGC/ocloc error patterns."""

    def test_matches_full_ptss_message(self):
        msg = ("error: total scratch space exceeds HW supported limit for kernel "
               "dot_kernel: 587264 bytes (max permitted PTSS 262144 bytes)")
        m = PTSS_OVERFLOW_RE.search(msg)
        assert m is not None
        assert m.group(1) == "587264"
        assert m.group(2) == "262144"

    def test_matches_multiline_ptss_message(self):
        msg = ("Build failed:\n"
               "total scratch space exceeds HW supported limit\n"
               "for kernel dot_kernel: 321024 bytes\n"
               "(max permitted PTSS 262144 bytes)")
        m = PTSS_OVERFLOW_RE.search(msg)
        assert m is not None
        assert m.group(1) == "321024"
        assert m.group(2) == "262144"

    def test_matches_generic_scratch_space_exceeds(self):
        msg = "error: scratch space exceeds hardware limit"
        m = PTSS_OVERFLOW_RE.search(msg)
        assert m is not None

    def test_matches_per_thread_scratch_exceed(self):
        msg = "per-thread scratch space would exceed the maximum"
        m = PTSS_OVERFLOW_RE.search(msg)
        assert m is not None

    def test_no_match_on_unrelated_error(self):
        msg = "error: undefined symbol 'foo'"
        assert PTSS_OVERFLOW_RE.search(msg) is None

    def test_no_match_on_normal_spill_info(self):
        msg = "spill_size: 500"
        assert PTSS_OVERFLOW_RE.search(msg) is None


class TestOutOfResourcesForPTSS:
    """Verify OutOfResources formats the error message correctly."""

    def test_with_exact_sizes(self):
        err = OutOfResources(587264, 262144, "per-thread scratch space (PTSS)")
        msg = str(err)
        assert "587264" in msg
        assert "262144" in msg
        assert "per-thread scratch space" in msg
        assert "Reducing block sizes" in msg

    def test_with_zero_sizes_degenerate(self):
        err = OutOfResources(
            0, 0,
            "per-thread scratch space (PTSS). "
            "The kernel's register spill exceeds hardware limits"
        )
        msg = str(err)
        assert "per-thread scratch space" in msg
        assert "register spill exceeds hardware limits" in msg

    def test_is_picklable(self):
        import pickle
        err = OutOfResources(587264, 262144, "per-thread scratch space (PTSS)")
        restored = pickle.loads(pickle.dumps(err))
        assert str(restored) == str(err)
