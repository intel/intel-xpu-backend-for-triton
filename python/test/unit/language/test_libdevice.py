import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl
from triton.language.extra.intel import libdevice


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize(
    "libdevice_fn, torch_special_fn",
    [
        ("j0", "bessel_j0"),
        ("j1", "bessel_j1"),
        ("y0", "bessel_y0"),
        ("y1", "bessel_y1"),
    ],
)
def test_bessel(dtype_str, libdevice_fn, torch_special_fn, device):
    SIZE = 128
    dtype = getattr(torch, dtype_str)

    x = torch.randn((SIZE, ), dtype=dtype, device=device)
    y_exp = torch.empty((SIZE, ), dtype=dtype, device=device)
    y_ref = getattr(torch.special, torch_special_fn)(x)

    @triton.jit
    def kernel(in_p, out_p, fn: tl.constexpr, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(in_p + off)
        res = getattr(libdevice, fn)(x)
        tl.store(out_p + off, res)

    kernel[(1, )](x, y_exp, fn=libdevice_fn, SIZE=SIZE, num_warps=4, num_ctas=1)

    torch.testing.assert_close(y_ref, y_exp, equal_nan=True)
