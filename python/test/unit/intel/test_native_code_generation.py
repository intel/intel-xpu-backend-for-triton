import pytest
import triton
import triton.language as tl

from triton._internal_testing import numpy_random, to_triton, is_xpu_cri


def test_empty_kernel(device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass

    x = to_triton(numpy_random(SIZE, dtype_str="bfloat16"), device=device, dst_type="bfloat16")
    kernel[(1, )](x, SIZE=SIZE, num_warps=4, generate_native_code=True)


@pytest.mark.xfail(is_xpu_cri(), reason="unable to get spill_size")
def test_auto_large_grf(device):
    if is_xpu_cri():
        pytest.xfail("unable to get spill_size")

    SIZE = 1024

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        x = tl.arange(0, SIZE)
        y = tl.sort(x, descending=True)
        tl.store(X + x, y)

    x = to_triton(numpy_random(SIZE, dtype_str="float32"), device=device, dst_type="float32")
    # Triton XPU will auto choose large GRF mode for grf_mode='default'
    k = kernel[(1, )](x, SIZE=SIZE, num_warps=1, generate_native_code=True, grf_mode='default')
    assert "-cl-intel-256-GRF-per-thread" in k.metadata.build_flags
