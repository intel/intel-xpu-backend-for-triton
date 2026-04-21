import torch

import triton
import pytest
import triton.language as tl
import triton.tools.disasm as disasm


def test_disam_cubin():
    if not triton.runtime.driver.active.get_current_target().backend == "cuda":
        pytest.xfail("Test requires CUDA.")

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
    assert "PredicatedStore" in dis or "OpStore" in dis


def test_extract_handles_large_instruction_offsets(monkeypatch):
    # cuobjdump widens instruction offsets to 5+ hex digits past 64 KiB.
    # Make sure the parser keeps consuming instructions instead of stopping at
    # /*10000*/.
    def fake_check_output(cmd):
        assert cmd == ["/fake/cuobjdump", "-sass", "fake.cubin"]
        return b"""Function : test_kernel
.headerflags    @"EF_CUDA_SM103 EF_CUDA_PTX_SM(EF_CUDA_SM103)"
        /*fff0*/                   NOP;                                   /* 0x0000000000007918 */
                                                                              /* 0x0000000000000000 */
        /*10000*/                  EXIT;                                  /* 0x000000000000794d */
                                                                              /* 0x0000000000000000 */
"""

    monkeypatch.setattr(disasm, "path_to_cuobjdump", lambda: "/fake/cuobjdump")
    monkeypatch.setattr(disasm.subprocess, "check_output", fake_check_output)

    sass = disasm.extract("fake.cubin", None)

    assert sass.startswith("Function:test_kernel\n")
    assert "\tNOP;\n" in sass
    assert "\tEXIT;\n" in sass
