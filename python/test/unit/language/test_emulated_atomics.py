import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401
import os

import triton
import triton.language as tl

os.environ["TRITON_INTEL_EMULATE_FP16_ATOMICS"] = "1"

BLOCK = 16
NUMELEM = 15

assert NUMELEM < BLOCK


def gen_indices(device):
    return torch.tensor([2, 0, 1, 4, 5, 0, 1, 4, 3, 5, 15, 15, 2, 3, 2, 3], dtype=torch.int64, device=device)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_atomic_add(dtype, device):

    @triton.jit
    def kernel(in_p, offs_p, out_p, numelem, BLOCK: tl.constexpr):
        index = tl.arange(0, BLOCK)
        mask = index < numelem
        x = tl.load(in_p + index, mask)
        offs = tl.load(offs_p + index, mask)
        tl.atomic_add(out_p + offs, x, mask)

    x = torch.randn((BLOCK, ), dtype=dtype, device=device)
    offs = gen_indices(device)
    y = torch.randn((BLOCK, ), dtype=dtype, device=device)

    ref = y.clone()
    for i in range(NUMELEM):
        ref[offs[i]] = ref[offs[i]] + x[i]

    kernel[(1, )](x, offs, y, NUMELEM, BLOCK=BLOCK)

    torch.testing.assert_close(ref, y)
