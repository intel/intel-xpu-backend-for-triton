import re

import pytest
import torch
import triton
import triton.language as tl

import pathlib

from triton.runtime.driver import driver
from triton._internal_testing import is_xpu_cri


@pytest.mark.xfail(is_xpu_cri(), reason="unable to get spill_size")
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


def test_get_properties_error(device):
    device_count, = driver.active.utils.device_count

    with pytest.raises(RuntimeError, match="Device is not found"):
        # Expected an exception when querying an invalid device index
        driver.active.utils.get_device_properties(device_count)


def test_load_binary_error_device_error(device, tmp_path: pathlib.Path):
    ir = """
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"} {
      tt.func public @empty_func() {
        tt.return
      }
    }
    """

    temp_file = tmp_path / "test_regression_load_binary_error.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    device_count, = driver.active.utils.device_count

    with pytest.raises(RuntimeError, match="Device is not found"):
        # Expected an exception when loading binary on an invalid device index
        _ = driver.active.utils.load_binary(kernel.name, kernel.kernel, kernel.metadata.shared,
                                            kernel.metadata.build_flags, not kernel.metadata.generate_native_code,
                                            device_count)


def test_load_binary_error_kernel_error(device, tmp_path: pathlib.Path):
    ir = """
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"} {
      tt.func public @empty_func() {
        tt.return
      }
    }
    """

    temp_file = tmp_path / "test_regression_load_binary_error.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    device = driver.active.get_current_device()

    with pytest.raises(RuntimeError, match=r".*ZE_RESULT_ERROR_INVALID_KERNEL_NAME.*"):
        _ = driver.active.utils.load_binary("invalid name", kernel.kernel, kernel.metadata.shared,
                                            kernel.metadata.build_flags, not kernel.metadata.generate_native_code,
                                            device)


def test_wait_on_sycl_queue_error(device):
    # Pass an invalid (non-pointer) value to trigger conversion error
    with pytest.raises(RuntimeError, match=r"Failed to convert PyObject to void\* for queue.*"):
        driver.active.utils.wait_on_sycl_queue("invalid_queue_pointer")


def test_has_opencl_extension_error(device):
    device_count, = driver.active.utils.device_count

    # Pass an invalid device_id (out of range) to trigger error
    with pytest.raises(RuntimeError, match="Device is not found"):
        driver.active.utils.has_opencl_extension(device_count, b"cl_khr_fp16")


@pytest.mark.parametrize("grf_mode, expect_retry, expect_fail",
                         [("default", True, False),  # Should auto-retry with large GRF and succeed
                          ("256", False, False),  # Explicit large GRF — compiles on first attempt
                          ("128", False, True),  # Explicit small GRF — should fail, no retry
                          ])
@pytest.mark.parametrize("generate_native_code", [False, True], ids=["load_binary", "make_zebin"])
def test_auto_grf_on_build_failure(device, monkeypatch, capfd, grf_mode, expect_retry, expect_fail,
                                   generate_native_code):
    """Test GRF mode behavior for register-heavy kernels on both compilation paths:
    - load_binary (generate_native_code=False): L0 runtime compilation via zeModuleCreate
    - make_zebin (generate_native_code=True): offline compilation via ocloc
    """
    monkeypatch.setenv("TRITON_DEBUG", "1")

    @triton.jit
    def _register_heavy_kernel(
        output_ptr,
        input_ptr,
        q_ptr,
        size,
        BLOCK: tl.constexpr,
    ):
        off = tl.arange(0, BLOCK)
        mask = off < size
        x = tl.load(input_ptr + off, mask=mask, other=0.0)
        q = tl.load(q_ptr + off, mask=mask, other=float("-inf"))
        result = tl.argmax(x / q, axis=-1)
        tl.store(output_ptr, result)

    BLOCK = 131072  # Large enough to exceed PTSS with default/small GRF
    size = 128000

    x = torch.randn(size, dtype=torch.float32, device=device)
    q = torch.rand(size, dtype=torch.float32, device=device)
    out = torch.empty(1, dtype=torch.int32, device=device)

    if expect_fail:
        with pytest.raises(RuntimeError):
            _register_heavy_kernel[(1, )](out, x, q, size, BLOCK=BLOCK, grf_mode=grf_mode,
                                          generate_native_code=generate_native_code)
    else:
        _register_heavy_kernel[(1, )](out, x, q, size, BLOCK=BLOCK, grf_mode=grf_mode,
                                      generate_native_code=generate_native_code)

        outs = capfd.readouterr().out
        if expect_retry and not generate_native_code:
            # load_binary path prints a retry message to stdout.
            assert "retrying with large GRF mode" in outs or "recompiling the kernel using large GRF mode" in outs
        elif expect_retry and generate_native_code:
            # make_zebin path retries silently via ocloc — no stdout message.
            # Success without exception is sufficient verification.
            pass
        else:
            assert "retrying with large GRF mode" not in outs
            assert "Build failed" not in outs
