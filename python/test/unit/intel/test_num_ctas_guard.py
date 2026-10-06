"""The Intel backend must reject `num_ctas > 1` instead of miscompiling it.

https://github.com/intel/intel-xpu-backend-for-triton/issues/6719

Nothing in the backend implements CTA clusters: `getClusterCTAId` returns 0,
`clusterBarrier` lowers to a workgroup barrier, and `loadDShared`/`storeDShared`
ignore the `ctaId` they are handed. `num_ctas` was nevertheless settable — it is
only a dataclass default, and `parse_options` copies through whatever the user
passes — so a kernel that communicates across CTAs compiled to code that reduced
within one CTA and returned a wrong answer.

This surfaced while removing the Intel-specific reduce lowering: that lowering
happened to *reject* a non-CTA-local `tt.reduce`, so deleting it in favour of the
common one would have turned a loud compile-time failure into a silent wrong
answer. The guard makes the rejection independent of which reduce lowering runs.
"""
import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)


@triton.jit
def _sum_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(y_ptr, tl.sum(tl.load(x_ptr + offs), axis=0))


@pytest.mark.parametrize("num_ctas", [2, 4])
def test_num_ctas_greater_than_one_is_rejected(num_ctas):
    x = torch.arange(128, device='xpu', dtype=torch.float32)
    y = torch.empty(1, device='xpu', dtype=torch.float32)
    with pytest.raises(ValueError, match=f"num_ctas={num_ctas} is unsupported"):
        _sum_kernel[(1, )](x, y, BLOCK=128, num_ctas=num_ctas)


def test_num_ctas_one_still_compiles():
    """Positive control: the guard must not reject the only supported value."""
    x = torch.arange(128, device='xpu', dtype=torch.float32)
    y = torch.empty(1, device='xpu', dtype=torch.float32)
    _sum_kernel[(1, )](x, y, BLOCK=128, num_ctas=1)
    torch.xpu.synchronize()
    torch.testing.assert_close(y[0], x.sum())
