import pytest
import torch
import triton.language as tl
import triton
import sys
import subprocess
import os
from triton._internal_testing import run_in_process


def _run_expect_zero_device_assert(device):
    triton.knobs.refresh_knobs()
    x = torch.ones([16], dtype=torch.float32, device=device)
    out = torch.empty_like(x)

    @triton.jit
    def _kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        y = tl.load(x_ptr + offsets)
        y = tl.expect_zero(y, offsets == 0)
        tl.store(out_ptr + offsets, y)

    _kernel[(1, )](x, out, BLOCK_SIZE=16)
    getattr(torch, device).synchronize()


def test_expect_zero_device_assert(device):
    result = run_in_process(_run_expect_zero_device_assert, (device, ), env={"TRITON_DEBUG": "1"})
    assert isinstance(result.exc, RuntimeError)


@pytest.mark.parametrize('cond', [True, False])
@pytest.mark.parametrize('mask', [True, False, None])
@pytest.mark.parametrize('opt_flag', [True, False, None])
@pytest.mark.parametrize('env_var', [True, False])
@pytest.mark.parametrize('jit_flag', [True, False])
@pytest.mark.forked
def test_device_assert(cond, mask, opt_flag, env_var, jit_flag, device):
    """Temporary subprocess solution due to:
    https://github.com/pytorch/pytorch/issues/142135"""

    is_debug = env_var or (opt_flag if opt_flag is not None else jit_flag)

    should_fail = not cond and is_debug and mask is not False
    kernel_file = os.path.join(os.path.dirname(__file__), "test_debug_kernels.py")
    mask_str = "None" if mask is None else str(mask)
    opt_flag_str = "None" if opt_flag is None else str(opt_flag)

    env = os.environ.copy()
    env["TRITON_DEBUG"] = str(int(env_var))

    result = subprocess.run(
        [sys.executable, kernel_file, "device_assert",
         str(cond), mask_str, opt_flag_str,
         str(jit_flag), device], capture_output=True, text=True, env=env)

    if should_fail:
        if device == 'xpu':
            assert result.returncode == -6, (f"Expected SIGABRT but got exit code {result.returncode}. "
                                             f"stdout: {result.stdout}, stderr: {result.stderr}")
        else:
            assert result.returncode == 1, (f"Expected runtime error but got unexpected exit code {result.returncode}. "
                                            f"stdout: {result.stdout}, stderr: {result.stderr}")
    else:
        assert result.returncode == 0, (f"Expected success but got unexpected exit code {result.returncode}. "
                                        f"stdout: {result.stdout}, stderr: {result.stderr}")


@pytest.mark.forked
def test_device_assert_barrier(device):
    """Subprocess solution to handle XPU SIGABRT from device_assert infrastructure.
    See: https://github.com/pytorch/pytorch/issues/142135"""
    kernel_file = os.path.join(os.path.dirname(__file__), "test_debug_kernels.py")

    env = os.environ.copy()
    env["TRITON_DEBUG"] = "1"

    result = subprocess.run([sys.executable, kernel_file, "device_assert_barrier", device], capture_output=True,
                            text=True, env=env)

    assert result.returncode == 0, (f"Expected success but got exit code {result.returncode}. "
                                    f"stdout: {result.stdout}, stderr: {result.stderr}")


@pytest.mark.parametrize("cond", [False, True])
def test_static_assert(cond):

    @triton.jit
    def _kernel(COND: tl.constexpr):
        tl.static_assert(COND)

    if not cond:
        with pytest.raises(triton.compiler.errors.CompileTimeAssertionFailure):
            _kernel[(1, )](cond)
        return

    _kernel[(1, )](cond)


def _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, tri_func, ref_func, device):
    # On XPU, __devicelib_assert_fail fires SIGABRT instead of RuntimeError.
    # Use subprocess to catch the signal. See https://github.com/intel/intel-xpu-backend-for-triton/issues/2755
    if should_overflow and debug and device == 'xpu':
        kernel_file = os.path.join(os.path.dirname(__file__), "test_debug_kernels.py")
        op = tri_func.fn.__name__.removeprefix("_kernel_")
        result = subprocess.run(
            [sys.executable, kernel_file, "overflow", op,
             str(x), str(y), x_dtype, y_dtype,
             str(debug), device], capture_output=True, text=True)
        assert result.returncode == -6, (f"Expected SIGABRT but got exit code {result.returncode}. "
                                         f"stdout: {result.stdout}, stderr: {result.stderr}")
        return
    x = torch.tensor([x], dtype=getattr(torch, x_dtype), device=device)
    y = torch.tensor([y], dtype=getattr(torch, y_dtype), device=device)
    z = torch.empty_like(x)
    if should_overflow and debug:
        with pytest.raises(RuntimeError) as exc_info:
            tri_func[(1, )](x, y, z, debug=debug)
            getattr(torch, device).synchronize()
        assert "device-side assert" in str(exc_info.value)
    else:
        tri_func[(1, )](x, y, z, debug=debug)
        getattr(torch, device).synchronize()
        assert int(z) == int(ref_func(x, y))


# integer overflow sanitization


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (-2**31, -1, 'int32', 'int32', False, False),
    (-2**31, -1, 'int32', 'int32', True, True),
    (2**31 - 1, 1, 'int32', 'int32', True, True),
    (2**31 - 1, 100, 'int32', 'int32', True, True),
    (-2**31, 0, 'int32', 'int32', True, False),
    (-2**31, 2, 'int32', 'int32', True, False),
    (0, -1, 'int32', 'int32', True, False),
    (-2**15, -1, 'int16', 'int16', True, True),
    (2**15 - 1, 1, 'int16', 'int16', True, True),
])
@pytest.mark.forked
def test_sanitize_int_add_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, device):

    @triton.jit
    def _kernel_add(X, Y, Z):
        tl.store(Z, tl.load(X) + tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, _kernel_add, lambda x, y: x + y, device)


# mul overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (2**30, 4, 'int32', 'int32', False, False),
    (2**30, 4, 'int32', 'int32', True, True),
    (2**30, 2, 'int32', 'int32', True, True),
    (-2**30, -4, 'int32', 'int32', True, True),
    (-2**31, 1, 'int32', 'int32', True, False),
    (-2**30, 2, 'int32', 'int32', True, False),
])
@pytest.mark.forked
def test_sanitize_int_mul_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, device):

    @triton.jit
    def _kernel_mul(X, Y, Z):
        tl.store(Z, tl.load(X) * tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, _kernel_mul, lambda x, y: x * y, device)


# sub overflow


@pytest.mark.parametrize("x, y, x_dtype, y_dtype, debug, should_overflow", [
    (-2**31, 1, 'int32', 'int32', False, False),
    (-2**31, 1, 'int32', 'int32', True, True),
    (2**31 - 1, -1, 'int32', 'int32', True, True),
    (2**31 - 1, 1, 'int32', 'int32', True, False),
    (-2**31, -1, 'int32', 'int32', True, False),
])
@pytest.mark.forked
def test_sanitize_int_sub_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, device):

    @triton.jit
    def _kernel_sub(X, Y, Z):
        tl.store(Z, tl.load(X) - tl.load(Y))

    _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, _kernel_sub, lambda x, y: x - y, device)
