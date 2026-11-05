"""num_warps * warp_size above the device's maximum work-group size.

`validate_options` bounds num_warps by `max_num_sub_groups`, which is the limit
at the smallest sub-group size. A kernel with 32 threads per warp could pass it
and still need more work-items than a work-group holds, e.g. a DPAS kernel with
num_warps=64 under TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32. It then failed at
launch with ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION (an `IntelGPUError`),
which the autotuner does not catch, so a single such config aborted the whole
autotune. The compiler now raises `OutOfResources`, which the autotuner skips.
"""
import pickle

import pytest
import torch
import triton
import triton.language as tl

from triton.backends.intel.errors import WorkGroupSizeOutOfResources
from triton.runtime.errors import OutOfResources

pytestmark = pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()),
                                reason="Intel XPU device not available")


def _target_arch():
    return triton.runtime.driver.active.get_current_target().arch


def _num_warps_filling_work_group():
    """The largest num_warps that fits a work-group at the smallest sub-group size.

    At 32 threads per warp the same num_warps needs twice the work-items.
    """
    arch = _target_arch()
    num_warps = arch["max_work_group_size"] // min(arch["sub_group_sizes"])
    if 32 not in arch["sub_group_sizes"] or num_warps > arch["max_num_sub_groups"]:
        pytest.skip("target cannot express num_warps * 32 > max_work_group_size with a valid num_warps")
    return num_warps


def _has_dpas():
    # Evaluated at collection time: must not raise without a usable device.
    try:
        return bool(_target_arch().get("has_subgroup_matrix_multiply_accumulate", False))
    except Exception:
        return False


@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs[:, None] * BLOCK + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * BLOCK + offs[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * BLOCK + offs[None, :], c)


@triton.jit
def _add_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + offs, tl.load(x_ptr + offs) + 1.0)


def _dot_inputs(device, block=128):
    a = torch.randn((block, block), device=device, dtype=torch.float16)
    b = torch.randn((block, block), device=device, dtype=torch.float16)
    c = torch.empty((block, block), device=device, dtype=torch.float32)
    return a, b, c


def test_work_group_size_error_message_and_pickling():
    err = WorkGroupSizeOutOfResources(64, 32, 1024, smaller_warp_size=16, sub_32_dpas=True)
    assert isinstance(err, OutOfResources)
    assert str(err) == ("out of resource: work-group size, Required: 2048 work-items (num_warps=64 * warp_size=32), "
                        "Hardware limit: 1024. To fit, use num_warps <= 32 or warp_size=16. "
                        "sub_32_dpas (TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32) is enabled, so DPAS kernels use "
                        "warp_size=32 by default.")
    no_smaller = WorkGroupSizeOutOfResources(128, 16, 1024)
    assert str(no_smaller) == ("out of resource: work-group size, Required: 2048 work-items "
                               "(num_warps=128 * warp_size=16), Hardware limit: 1024. To fit, use num_warps <= 64.")
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is WorkGroupSizeOutOfResources and str(clone) == str(err)


@pytest.mark.skipif(not _has_dpas(), reason="needs DPAS")
def test_sub_32_dpas_oversized_work_group_raises(device, fresh_triton_cache):
    num_warps = _num_warps_filling_work_group()
    a, b, c = _dot_inputs(device)
    with pytest.raises(OutOfResources, match="work-group size") as excinfo:
        _dot_kernel[(1, )](a, b, c, BLOCK=128, num_warps=num_warps, sub_32_dpas=True)
    msg = str(excinfo.value)
    max_work_group_size = _target_arch()["max_work_group_size"]
    assert f"num_warps={num_warps} * warp_size=32" in msg
    assert f"num_warps <= {max_work_group_size // 32} or warp_size=16" in msg
    assert "TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32" in msg
    assert "num_stages" not in msg


def test_explicit_warp_size_oversized_work_group_raises(device, fresh_triton_cache):
    num_warps = _num_warps_filling_work_group()
    x = torch.zeros(4096, device=device)
    y = torch.empty_like(x)
    with pytest.raises(OutOfResources, match="work-group size") as excinfo:
        _add_kernel[(1, )](x, y, BLOCK=4096, num_warps=num_warps, warp_size=32)
    msg = str(excinfo.value)
    assert f"num_warps={num_warps} * warp_size=32" in msg
    assert "or warp_size=16" in msg
    assert "TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32" not in msg


@pytest.mark.skipif(not _has_dpas(), reason="needs DPAS")
def test_autotuner_skips_oversized_work_group(device, fresh_triton_cache):
    num_warps = _num_warps_filling_work_group()
    kernel = triton.autotune(
        configs=[
            triton.Config({"sub_32_dpas": True}, num_warps=num_warps),
            triton.Config({"sub_32_dpas": True}, num_warps=4),
        ],
        key=[],
    )(_dot_kernel)
    a, b, c = _dot_inputs(device)
    kernel[(1, )](a, b, c, BLOCK=128)
    assert kernel.best_config.num_warps == 4
    torch.testing.assert_close(c, torch.matmul(a.float(), b.float()), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _has_dpas(), reason="needs DPAS")
def test_default_dpas_kernel_with_many_warps_still_runs(device, fresh_triton_cache):
    # Without the knob, DPAS kernels use 16 threads per warp, which fits.
    num_warps = _num_warps_filling_work_group()
    a, b, c = _dot_inputs(device)
    k = _dot_kernel[(1, )](a, b, c, BLOCK=128, num_warps=num_warps)
    assert k.metadata.threads_per_warp == 16
    torch.testing.assert_close(c, torch.matmul(a.float(), b.float()), atol=1e-2, rtol=1e-2)
