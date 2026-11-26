"""Unit test for `get_grf_register_size()`'s per-target GRF register width.

https://github.com/intel/intel-xpu-backend-for-triton/issues/8417

`get_grf_register_size()` is what `parse_target` calls to populate
`dev_prop['grf_register_size']`, which reaches `RegisterPressureAnalysis` as the
`ttig.grf_register_size` module attribute. The width is 4 bytes times the
smallest sub-group size, whatever the `arch` string is.

`test_get_grf_register_size` needs no device; `test_module_attribute` compiles a
kernel and checks that the width reaches the TTGIR.
"""
import pytest
import torch
import triton
import triton.language as tl

from triton.backends.intel.compiler import get_grf_register_size


@pytest.mark.parametrize("arch, sub_group_sizes, expected", [
    ("dg2", [8, 16, 32], 32),
    ("mtl", [8, 16, 32], 32),
    ("", [8, 16, 32], 32),  # a DG2/MTL/ARL SKU `arch_parser.c` does not list
    ("pvc", [16, 32], 64),
    ("bmg", [16, 32], 64),
    ("cri", [16, 32], 64),
    ("unknown", [16, 32], 64),
])
def test_get_grf_register_size(arch, sub_group_sizes, expected):
    assert get_grf_register_size({"arch": arch, "sub_group_sizes": sub_group_sizes}) == expected


@triton.jit
def _copy(X, Y, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(Y + offs, tl.load(X + offs))


@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="Intel XPU device not available")
def test_module_attribute(device):
    x = torch.arange(16, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    k = _copy[(1, )](x, y, BLOCK=16)
    expected = 4 * min(triton.runtime.driver.active.get_current_target().arch["sub_group_sizes"])
    assert f"ttig.grf_register_size = {expected} : i32" in k.asm["ttgir"]
