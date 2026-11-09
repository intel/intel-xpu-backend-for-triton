"""Runtime check for the sub-group reduce of an `i1` add (#8212).

`i1` add is mod 2. The lit checks pin the logical-op mapping; this test
checks parity on hardware.
"""
import re

import pytest
import torch

import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _kernel(out_ptr, in_ptr, R: tl.constexpr, M: tl.constexpr):
    rows = tl.arange(0, R)[:, None]
    x = tl.load(in_ptr + rows * M + tl.arange(0, M)[None, :]) != 0
    tl.store(out_ptr + tl.arange(0, R), tl.reduce(x, 1, _add).to(tl.int32))


# Cluster size of each `LogicalXor` call on the default 32-lane sub-group, 0 for
# a whole-sub-group Reduce. Cross-warp combines the four warp partials in a
# 4-lane cluster.
@pytest.mark.parametrize("M, num_warps, clusters", [(8, 1, {8}), (128, 1, {0}), (512, 4, {0, 4})],
                         ids=["clustered", "sub-group", "cross-warp"])
def test_i1_add_reduce(M, num_warps, clusters, device):
    rows = [
        [1] + [0] * (M - 1),
        [1] + [0] * (M - 2) + [1],
        [1, 1] + [0] * (M - 3) + [1],
        [1, 1, 0, 1, 0, 0, 1, 0] * (M // 8),
    ]
    x = torch.tensor(rows, dtype=torch.int32, device=device)
    out = torch.empty(len(rows), dtype=torch.int32, device=device)
    kernel = _kernel[(1, )](out, x, len(rows), M, num_warps=num_warps)

    calls = re.findall(r"LogicalXoriibj?\(i32 3, i32 \d, i1 %\w+(?:, i32 (\d+))?\)", kernel.asm["llir"])
    assert {int(c or 0) for c in calls} == clusters
    assert out.tolist() == [sum(row) % 2 for row in rows]
