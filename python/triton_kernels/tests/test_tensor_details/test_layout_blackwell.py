import pytest
from triton._internal_testing import is_xpu
import torch
from triton_kernels.tensor_details.layout import BlackwellMXScaleLayout

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 254, 60),
        (1, 320, 160),
        (2, 16, 512),
        (3, 2, 36),
    ],
)
@pytest.mark.xfail(condition=is_xpu(), reason="Test expected to fail on XPU devices")
def test_mxfp4_scale_roundtrip(shape):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = BlackwellMXScaleLayout(x.shape)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res == x).all()
