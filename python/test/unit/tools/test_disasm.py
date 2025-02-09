import torch

import triton
import pytest
import triton.language as tl


def test_disam_cubin():
    if not triton.runtime.driver.active.get_current_target().backend == "cuda":
        pytest.skip("Test requires CUDA.")

    @triton.jit
    def kernel(X, i: tl.constexpr):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device='cuda')
    h = kernel[(1, )](x, i=12)
    assert x[0] == 12
    sass = h.asm["sass"]
    # check that the sass has a store instruction.
    assert "STG.E" in sass


def test_disam_spvbin():
    if not triton.runtime.driver.active.get_current_target().backend == "xpu":
        pytest.skip("Test requires XPU.")

    @triton.jit
    def kernel(X, i: tl.constexpr):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device='xpu')
    h = kernel[(1, )](x, i=12)
    assert x[0] == 12
    dis = h.asm["spvdis"]
    # check that the spvdis has a store instruction.
    assert "OpStore" in dis
