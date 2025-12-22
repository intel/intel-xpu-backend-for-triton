import re

import torch
import triton
import triton.language as tl

import pathlib


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
    from triton.runtime.driver import driver
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

    from triton.runtime.driver import driver

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

    from triton.runtime.driver import driver
    device = driver.active.get_current_device()

    with pytest.raises(RuntimeError, match=r".*ZE_RESULT_ERROR_INVALID_KERNEL_NAME.*"):
        _ = driver.active.utils.load_binary("invalid name", kernel.kernel, kernel.metadata.shared,
                                            kernel.metadata.build_flags, not kernel.metadata.generate_native_code,
                                            device)


def test_wait_on_sycl_queue_error(device):
    from triton.runtime.driver import driver

    # Pass an invalid (non-pointer) value to trigger conversion error
    try:
        driver.active.utils.wait_on_sycl_queue("invalid_queue_pointer")
        assert False, "Expected an exception when passing invalid queue pointer"
    except RuntimeError as e:
        assert "Failed to convert PyObject to void* for queue" in str(e)


def test_has_opencl_extension_error(device):
    from triton.runtime.driver import driver
    device_count, = driver.active.utils.device_count

    # Pass an invalid device_id (out of range) to trigger error
    try:
        driver.active.utils.has_opencl_extension(device_count, b"cl_khr_fp16")
        assert False, "Expected an exception when querying an invalid device index"
    except RuntimeError as e:
        assert "Device is not found" in str(e)
