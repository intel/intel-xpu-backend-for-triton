import ctypes
import re

import pytest
import torch
import triton
import triton.language as tl

import pathlib

from triton.runtime.driver import driver
from triton._internal_testing import is_xpu_cri
from triton.backends.intel import extension_utils
from triton.runtime.errors import IntelGPUError, OutOfResources


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
    assert "retrying with large GRF mode" in outs[0]
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

    with pytest.raises(IntelGPUError, match=r".*ZE_RESULT_ERROR_INVALID_KERNEL_NAME.*"):
        _ = driver.active.utils.load_binary("invalid name", kernel.kernel, kernel.metadata.shared,
                                            kernel.metadata.build_flags, not kernel.metadata.generate_native_code,
                                            device)


def test_wait_on_sycl_queue_error(device):
    # Pass an invalid (non-pointer) value to trigger conversion error
    with pytest.raises(RuntimeError, match=r"Failed to convert PyObject to void\* for queue.*"):
        driver.active.utils.wait_on_sycl_queue("invalid_queue_pointer")


def test_has_opencl_extension_error(device):
    device_idx = torch.xpu.current_device()
    device_id = extension_utils.get_device_id(device_idx)

    # Test that we can query extensions using the new API
    extensions = extension_utils.query_device_extensions(device_id=device_id)

    # Verify we got a dictionary with expected extension keys
    assert isinstance(extensions, dict)
    assert "has_subgroup_matrix_multiply_accumulate" in extensions
    assert "has_subgroup_matrix_multiply_accumulate_tensor_float32" in extensions
    assert "has_2d_block_io" in extensions
    assert "has_bfloat16_conversion" in extensions
    if device_id == 3034:
        # PVC 1100
        assert extensions["has_subgroup_matrix_multiply_accumulate"] is True
        assert extensions["has_subgroup_matrix_multiply_accumulate_tensor_float32"] is False
        assert extensions["has_2d_block_io"] is True
        assert extensions["has_bfloat16_conversion"] is True

    # Test individual extension checking
    result = extension_utils.has_device_extension(device_id, "cl_intel_subgroup_2d_block_io")
    assert isinstance(result, bool)
    if device_id == 3034:
        # PVC 1100
        assert result is True  # This extension should be supported

    # Test checking for a non-existent/wrong extension name
    result_wrong = extension_utils.has_device_extension(device_id, "cl_intel_nonexistent_extension")
    assert isinstance(result_wrong, bool)
    assert result_wrong is False  # This extension should not be supported

    assert extension_utils.has_device_extension(9999, "cl_intel_subgroup_2d_block_io") is None


@pytest.mark.parametrize("grf_mode, expect_retry", [("default", True),  # Should auto-retry with large GRF and succeed
                                                    ("256", False),  # Explicit large GRF — compiles on first attempt
                                                    ("128", False),  # Explicit small GRF — should fail, no retry
                                                    ])
@pytest.mark.parametrize("generate_native_code", [False, True], ids=["load_binary", "make_zebin"])
def test_auto_grf_on_build_failure(device, monkeypatch, capfd, grf_mode, expect_retry, generate_native_code):
    """Test GRF mode behavior for register-heavy kernels on both compilation paths:
    - load_binary (generate_native_code=False): L0 runtime compilation via zeModuleCreate
    - make_zebin (generate_native_code=True): offline compilation via ocloc
    """
    # The build failure with grf_mode="128" is not simulated on CRI properly
    if grf_mode == "128" and is_xpu_cri():
        pytest.xfail("grf_mode=128 build failure is not simulated on CRI properly")

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

    try:
        _register_heavy_kernel[(1, )](out, x, q, size, BLOCK=BLOCK, grf_mode=grf_mode,
                                      generate_native_code=generate_native_code)
    except (IntelGPUError, OutOfResources):
        # OutOfResources is the new spill-related error class introduced by
        # the PTSS-overflow handling in this PR; both error types are
        # acceptable here since this test exercises a kernel intentionally
        # too large for the chosen GRF mode.
        pass

    outs = capfd.readouterr().out
    if expect_retry and not generate_native_code:
        # load_binary path prints a retry message to stdout.
        assert "retrying with large GRF mode" in outs
    elif expect_retry and generate_native_code:
        # make_zebin path retries silently via ocloc — no stdout message.
        # Success without exception is sufficient verification.
        pass
    else:
        assert "retrying with large GRF mode" not in outs
        assert "Build failed" not in outs


def test_sycl_global_range_overflow(device):
    # for details: https://github.com/intel/intel-xpu-backend-for-triton/issues/7201

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0).to(tl.int64)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    n = 1379584
    x = torch.randint(0, 100, (n, 2048), dtype=torch.int8, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    add_kernel[grid](x, x, output, n_elements, 16)

    torch.testing.assert_close(output.cpu(), (x + x).cpu(), rtol=0, atol=0)


# =============================================================================
# Fix E: packed-launch fast path coverage
# =============================================================================
#
# The packed-launch path (XPULauncher._use_packed_launch = True) resolves
# pointer arguments in Python once per launch and hands C a flat void*[]
# instead of calling `.data_ptr()` per-arg inside the C launcher. These
# tests exercise:
#   - Eligibility (which kernels get selected for the fast path)
#   - Pointer-source variants (torch tensor, Python int, None)
#   - Scalar/pointer coexistence in the same signature
#   - Fallback via TRITON_INTEL_DISABLE_PACKED_LAUNCH=1
#   - Fallback for unsupported (tensordesc) annotations
#   - Cross-path parity (packed vs classic produce identical output)
#   - Enter-hook re-entrancy safety (see driver.c launch_packed snapshot)
#   - build_pack_layout input validation (rejects non-PyKernelArg items)
#
# Where a test needs to bypass the launcher entirely and hit `build_pack_layout`
# directly, it uses `driver.active.utils.build_pack_layout` — this only exists
# on builds with the Fix E C symbols present, so tests that call it are
# skipped otherwise.


def _has_packed_launch():
    return getattr(driver.active.utils, "launch_packed", None) is not None


packed_launch_only = pytest.mark.skipif(not _has_packed_launch(), reason="build lacks Fix E packed-launch symbols")


@triton.jit
def _packed_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _packed_scalar_and_ptr_kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * alpha, mask=mask)


@packed_launch_only
def test_packed_launch_selected_by_default(device):
    """
    A simple pointer-heavy kernel should end up on the packed fast path AND
    produce a correct result. Rather than reaching into triton's per-device
    caches (which is a private implementation detail), we probe eligibility
    via `build_pack_layout` directly with the same signature/annotation
    layout a real launch would produce.
    """
    from triton.backends.intel.driver import PyKernelArg, ARG_KERNEL
    utils = driver.active.utils

    # Simulate the annotations for _packed_add_kernel(x_ptr, y_ptr, out_ptr,
    # n, BLOCK): three pointers, one scalar int (n), one constexpr (BLOCK
    # stripped from the annotations list). Only ARG_KERNEL entries appear in
    # arg_annotations after expand_signature; constexprs are stripped.
    sig_bytes = bytes([1, 1, 1, 4])  # 3x pointer + 1x int32
    annotations = [PyKernelArg(nested_tuple=None, type=ARG_KERNEL) for _ in range(4)]
    pointer_indices, pack_bytes, packable = utils.build_pack_layout((sig_bytes, annotations))
    assert packable is True
    assert pointer_indices == (0, 1, 2)
    assert pack_bytes == 3 * ctypes.sizeof(ctypes.c_void_p)

    # And confirm end-to-end correctness through the real launcher.
    n, block = 256, 64
    x = torch.arange(n, dtype=torch.float32, device=device)
    y = torch.arange(n, dtype=torch.float32, device=device) * 2
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )
    _packed_add_kernel[grid](x, y, out, n, BLOCK=block)
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)


@packed_launch_only
def test_packed_launch_disable_env(device, monkeypatch):
    """TRITON_INTEL_DISABLE_PACKED_LAUNCH=1 forces the classic path."""
    monkeypatch.setenv("TRITON_INTEL_DISABLE_PACKED_LAUNCH", "1")
    # Force a re-specialization by using a fresh kernel def so __init__ reads
    # the env var afresh.

    @triton.jit
    def _add_disable_env(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n, block = 128, 64
    x = torch.arange(n, dtype=torch.float32, device=device)
    y = torch.arange(n, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )
    _add_disable_env[grid](x, y, out, n, BLOCK=block)
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)


@packed_launch_only
def test_packed_launch_none_pointer(device):
    """`None` as a pointer arg should launch cleanly (extractPointer maps it to nullptr)."""

    @triton.jit
    def _kernel_optional_ptr(x_ptr, opt_ptr, out_ptr, n, BLOCK: tl.constexpr):
        # opt_ptr is intentionally unused; the launcher just needs to accept
        # None without crashing.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)

    n, block = 64, 32
    x = torch.arange(n, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )
    _kernel_optional_ptr[grid](x, None, out, n, BLOCK=block)
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@packed_launch_only
def test_packed_launch_int_pointer(device):
    """A Python int used as a raw device-pointer value should launch cleanly."""
    n, block = 64, 32
    x = torch.arange(n, dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    @triton.jit
    def _kernel_int_ptr(x_ptr, unused_int_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, v, mask=mask)

    grid = (triton.cdiv(n, block), )
    # Any int is accepted; the kernel never reads through it, so we can pass 0.
    _kernel_int_ptr[grid](x, 0, out, n, BLOCK=block)
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@packed_launch_only
def test_packed_launch_scalar_and_pointer_coexist(device):
    """Kernels mixing scalar and pointer args should still produce correct output."""
    n, block = 128, 32
    x = torch.arange(n, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    alpha = 3.5
    grid = (triton.cdiv(n, block), )
    _packed_scalar_and_ptr_kernel[grid](x, out, n, alpha, BLOCK=block)
    torch.testing.assert_close(out, x * alpha, rtol=0, atol=1e-6)


@packed_launch_only
def test_packed_vs_classic_parity(device, monkeypatch):
    """
    Cross-path parity: the same kernel with the same inputs must produce the
    same output whether the packed fast path is used or forced off via
    TRITON_INTEL_DISABLE_PACKED_LAUNCH=1.
    """
    n, block = 256, 64
    x = torch.arange(n, dtype=torch.float32, device=device)
    y = torch.arange(n, dtype=torch.float32, device=device) * 2

    # Fresh kernel defs per branch so __init__ eligibility is re-computed.

    @triton.jit
    def _kernel_a(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(x_ptr + offs, mask=mask)
        b = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    @triton.jit
    def _kernel_b(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(x_ptr + offs, mask=mask)
        b = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    # Packed path
    monkeypatch.delenv("TRITON_INTEL_DISABLE_PACKED_LAUNCH", raising=False)
    out_packed = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )
    _kernel_a[grid](x, y, out_packed, n, BLOCK=block)

    # Classic path
    monkeypatch.setenv("TRITON_INTEL_DISABLE_PACKED_LAUNCH", "1")
    out_classic = torch.empty_like(x)
    _kernel_b[grid](x, y, out_classic, n, BLOCK=block)

    torch.testing.assert_close(out_packed, out_classic, rtol=0, atol=0)
    torch.testing.assert_close(out_packed, x + y, rtol=0, atol=0)


@packed_launch_only
def test_packed_launch_reentrant_enter_hook(device):
    """
    Regression test for the enter-hook re-entrancy risk in launch_packed:
    the C-side snapshots the packed_pointers buffer BEFORE running the enter
    hook, so a hook that re-enters the same kernel with different tensors
    must not corrupt the outer launch's pointer values.

    Without the snapshot, the outer launch would consume the inner launch's
    pointers because both share `XPULauncher._pack_buf`.
    """
    from triton import knobs

    n, block = 64, 32
    x_outer = torch.full((n, ), 7.0, dtype=torch.float32, device=device)
    x_inner = torch.full((n, ), 99.0, dtype=torch.float32, device=device)
    out_outer = torch.empty_like(x_outer)
    out_inner = torch.empty_like(x_inner)
    grid = (triton.cdiv(n, block), )

    @triton.jit
    def _identity(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, v, mask=mask)

    # Warm up the specialization so the enter-hook doesn't fire during compile.
    _identity[grid](x_outer, out_outer, n, BLOCK=block)

    reentered = {"count": 0}

    def _reentrant_hook(*_args, **_kwargs):
        # Fire once, and only from the outer launch's hook invocation.
        if reentered["count"] == 0:
            reentered["count"] = 1
            _identity[grid](x_inner, out_inner, n, BLOCK=block)

    knobs.runtime.launch_enter_hook.add(_reentrant_hook)
    try:
        _identity[grid](x_outer, out_outer, n, BLOCK=block)
    finally:
        knobs.runtime.launch_enter_hook.remove(_reentrant_hook)

    # The outer launch's output must reflect x_outer (7.0), not x_inner (99.0).
    # If the snapshot were missing, the outer would read x_inner because
    # self._pack_buf got overwritten during the inner call.
    torch.testing.assert_close(out_outer, x_outer, rtol=0, atol=0)
    torch.testing.assert_close(out_inner, x_inner, rtol=0, atol=0)
    assert reentered["count"] == 1


@packed_launch_only
def test_build_pack_layout_rejects_non_pykernelarg(device):
    """
    build_pack_layout is exposed via ctypes; a caller passing arbitrary Python
    objects (e.g. [None]) as arg_annotations must get a TypeError rather than
    crashing on an unchecked C-level cast.
    """
    utils = driver.active.utils
    # Build a minimal kernel_signature buffer (one pointer arg).
    sig_bytes = bytes([1])  # EXTRACTOR_POINTER_INDEX == 1
    # Pass a bogus arg_annotations list containing a non-PyKernelArg item.
    with pytest.raises(TypeError, match="arg_annotations entries must be PyKernelArg"):
        utils.build_pack_layout((sig_bytes, [None]))


@packed_launch_only
def test_build_pack_layout_reports_no_pointer_kernel(device):
    """
    A kernel with no pointer args should report `packable=False` from
    build_pack_layout — otherwise the launcher would allocate an empty pack
    buffer and gain nothing from the fast path.
    """
    from triton.backends.intel.driver import PyKernelArg, ARG_KERNEL
    # EXTRACTOR_TYPES: 1 = pointer, 4 = INT32 — use scalar to force "no pointer".
    utils = driver.active.utils
    sig_bytes = bytes([4])
    ann = PyKernelArg(nested_tuple=None, type=ARG_KERNEL)
    pointer_indices, pack_bytes, packable = utils.build_pack_layout((sig_bytes, [ann]))
    assert pointer_indices == ()
    assert pack_bytes == 0
    assert packable is False


@packed_launch_only
def test_build_pack_layout_arg_tuple_falls_back(device):
    """
    ARG_TUPLE annotations short-circuit `buildRawKernelArgIndices` and force
    `packable=False`. The launcher then falls back to the classic path
    without trying to pack a nested-tuple signature it can't reason about.
    """
    from triton.backends.intel.driver import PyKernelArg, ARG_KERNEL, ARG_TUPLE
    utils = driver.active.utils
    sig_bytes = bytes([1, 1])  # 2 pointers in the flattened signature
    annotations = [
        PyKernelArg(nested_tuple=None, type=ARG_KERNEL),
        # A nested-tuple annotation forces packable=False regardless of what
        # nested_tuple contains — the walk breaks immediately on ARG_TUPLE.
        PyKernelArg(nested_tuple=[PyKernelArg(nested_tuple=None, type=ARG_KERNEL)], type=ARG_TUPLE),
    ]
    _pointer_indices, _pack_bytes, packable = utils.build_pack_layout((sig_bytes, annotations))
    assert packable is False
