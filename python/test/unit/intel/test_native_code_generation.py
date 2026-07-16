import os
import tempfile

import pytest
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri
from triton.backends.intel.compiler import MAX_REG_SPILL, extract_spill_size_from_zebin


def test_empty_kernel(device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    x = to_triton(numpy_random(SIZE, dtype_str="bfloat16"), device=device, dst_type="bfloat16")
    kernel[(1, )](x, SIZE=SIZE, num_warps=4, generate_native_code=True)


@pytest.mark.xfail(is_xpu_cri(), reason="unable to get spill_size")
def test_auto_large_grf(device):
    SIZE = 2048

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        x = tl.arange(0, SIZE)
        y = tl.sort(x, descending=True)
        tl.store(X + x, y)

    x = to_triton(numpy_random(SIZE, dtype_str="float32"), device=device, dst_type="float32")
    # Triton XPU will choose large GRF mode when spill_size > MAX_REG_SPILL.
    k = kernel[(1, )](x, SIZE=SIZE, num_warps=1, generate_native_code=True, grf_mode='default')
    # delete=False + explicit close: on Windows, extract_spill_size_from_zebin's own
    # open() of the same path would otherwise hit a PermissionError while this
    # handle is still open (no such restriction on POSIX).
    f = tempfile.NamedTemporaryFile(mode='wb', suffix='.zebin', delete=False)
    try:
        f.write(k.kernel)
        f.close()
        spill_size = extract_spill_size_from_zebin(f.name)
    finally:
        os.unlink(f.name)
    if spill_size <= MAX_REG_SPILL:
        pytest.skip(f"Kernel did not spill above MAX_REG_SPILL ({spill_size} <= {MAX_REG_SPILL}); "
                    "auto-large-GRF path was not exercised. Consider increasing SIZE.")
    assert "-cl-intel-256-GRF-per-thread" in k.metadata.build_flags
