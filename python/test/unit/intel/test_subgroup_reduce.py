"""Runtime check for the sub-group reduce of an `i1` add (#8212).

`i1` add is mod 2. The lit checks pin the logical-op mapping; this test
checks parity on hardware.
"""
import re

import pytest
import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

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


@gluon.jit
def _add_partials(a, b, c, d):
    return a + c, b + d


@gluon.jit
def _single_barrier_kernel(x_ptr, y_ptr, out_x_ptr, out_y_ptr, M: gl.constexpr, N: gl.constexpr, LAYOUT: gl.constexpr):
    rows = gl.arange(0, M, layout=gl.SliceLayout(1, LAYOUT))[:, None]
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, LAYOUT))
    x = gl.load(x_ptr + rows * N + cols[None, :])
    y = gl.load(y_ptr + rows * N + cols[None, :])
    a, b = gl.reduce((x, y), axis=0, combine_fn=_add_partials)
    gl.store(out_x_ptr + cols, a)
    gl.store(out_y_ptr + cols, b)


@pytest.mark.parametrize("M, N, threads, warps", [
    (4, 256, [1, 32], [4, 1]),
    (128, 16, [8, 4], [4, 1]),
    (1024, 1, [32, 1], [32, 1]),
    (1024, 16, [8, 4], [32, 1]),
    (64, 64, [8, 4], [2, 2]),
])
@pytest.mark.parametrize("dtype_y", [torch.int8, torch.int64])
def test_reduce_single_barrier_mixed_operands(M, N, threads, warps, dtype_y, device):
    layout = gl.BlockedLayout([1, 4], threads, warps, [1, 0])
    generator = torch.Generator().manual_seed(17)
    x = torch.randint(-2, 3, (M, N), generator=generator).to(device=device, dtype=torch.float32)
    y = torch.randint(-2, 3, (M, N), generator=generator).to(device=device, dtype=dtype_y)
    out_x = torch.empty(N, dtype=x.dtype, device=device)
    out_y = torch.empty(N, dtype=y.dtype, device=device)
    kernel = _single_barrier_kernel[(1, )](x, y, out_x, out_y, M, N, layout, num_warps=warps[0] * warps[1])

    torch.testing.assert_close(out_x, x.sum(dim=0), rtol=0, atol=0)
    torch.testing.assert_close(out_y, y.sum(dim=0).to(dtype_y), rtol=0, atol=0)
    assert len(re.findall(r"call .*@_Z7barrierj\(", kernel.asm["llir"])) == 1
