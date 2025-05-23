import re
import tempfile
import subprocess
import sys
import os


def test_auto_grf():

    test_code = """
import numpy as np
import torch
import triton
import triton.language as tl

from triton._internal_testing import to_numpy


def test_auto_grf(device):
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
    z_ref = torch.arange(0, BLOCK, dtype=torch.int32, device=device)

test_auto_grf("xpu")
    """

    with (tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f):
        f.write(test_code)
        f.flush()
        env = os.environ.copy()
        env["TRITON_DEBUG"] = "1"
        proc = subprocess.run(
            [sys.executable, f.name],
            capture_output=True,
            env=env,
        )
        assert proc.returncode == 0
        outs = [line for line in proc.stdout.decode("UTF-8").splitlines() if line]
        # The output should contain the recompiling information for large GRF mode.
        assert re.search(r"recompiling the kernel using large GRF mode", outs[0])
        # The spill size of returned kernel should be same kernel as the one compiled with large GRF mode.
        assert re.findall(r"\d+\.?\d*", outs[1])[0] == re.findall(r"\d+\.?\d*", outs[2])[0]
