import re

import torch
import triton
import triton.language as tl


def test_auto_grf(device, monkeypatch, capfd):
    monkeypatch.setenv("TRITON_DEBUG", "1")
    BLOCK = 1024 * 8
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr):
        # make it hard to re-schedule.
        off = tl.arange(0, BLOCK)
        a = tl.load(z + off)
        result = tl.sum(a, axis=0, keep_dims=True)
        tl.store(z + off, a + result)

    _kernel[(1, )](z_tri, BLOCK=BLOCK, num_warps=2)
    _ = torch.arange(0, BLOCK, dtype=torch.int32, device=device)

    outs = [line for line in capfd.readouterr().out.splitlines() if line]
    # The output should contain the recompiling information for large GRF mode.
    assert re.search(r"recompiling the kernel using large GRF mode", outs[0])
    # The spill size of returned kernel should be same kernel as the one compiled with large GRF mode.
    assert re.findall(r"\d+\.?\d*", outs[1])[0] == re.findall(r"\d+\.?\d*", outs[2])[0]
