"""Regression test: `spirv_utils` must never be unloaded from the process.

https://github.com/intel/intel-xpu-backend-for-triton/issues/7682

`driver.c` defines `PyKernelArgType` as a *statically allocated* `PyTypeObject`
(`Py_TPFLAGS_DEFAULT`, `PyType_Ready`), so the type object's storage is a data-segment
address inside the JIT-compiled `spirv_utils` shared object. Instances of that type are
created per compiled kernel by `annotate_arguments()` and kept alive for the lifetime of
the kernel in `XPULauncher.arg_annotations`, and they travel inside the launch-args tuple.

`SpirvUtils` used to define a `__del__` that `dlclose()`d (POSIX) / `FreeLibrary()`d
(Windows) that shared object. Unloading it frees the type object's storage while
instances remain reachable, so the next cyclic-GC pass over any tuple holding one runs
`tupletraverse` -> `visit_decref` -> `_PyObject_IS_GC` -> `Py_TYPE(op)` and dereferences a
dangling `ob_type`, killing the interpreter with
`Fatal Python error: Segmentation fault`.

Note the instances are deliberately *not* GC-tracked, and that does not help: the fault
is the `Py_TYPE()` load that `visit_decref` performs before it can know whether the
object is tracked.

`ArchParser` and `ExtensionUtils` keep their unload on purpose. Their modules export no
Python type, so unloading is safe there and still lets Windows delete the cached files
(issue #3090, the reason the unload was introduced in #3230/#3251/#3455).
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from triton.backends.intel import driver as intel_driver

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU device not available",
)


def test_spirv_utils_defines_no_destructor():
    """The canary: catches anyone "restoring consistency" with the sibling classes.

    The fix is a deletion, so this is the only check that runs on every platform and
    fails fast if a `__del__` comes back. It also pins the deliberate asymmetry: the two
    classes whose modules export no Python type still unload.
    """
    assert not hasattr(intel_driver.SpirvUtils,
                       "__del__"), ("SpirvUtils must not define __del__: unloading spirv_utils frees PyKernelArgType "
                                    "while its instances are still reachable. See issue #7682.")
    assert hasattr(intel_driver.ArchParser, "__del__")
    assert hasattr(intel_driver.ExtensionUtils, "__del__")


# Runs in a subprocess because a regression here is a SIGSEGV, not an exception.
_CHILD = textwrap.dedent("""
    import gc, os, threading, sys
    import torch, triton, triton.language as tl
    from triton.backends.intel import driver as drv

    SO = "spirv_utils"

    def mapped():
        with open("/proc/self/maps") as fh:
            return SO in fh.read()

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

    def workload():
        x = torch.randn(8192, device="xpu")
        y = torch.randn(8192, device="xpu")
        o = torch.empty_like(x)
        _add[(8, )](x, y, o, x.numel(), BLOCK=1024)
        torch.xpu.synchronize()
        assert torch.allclose(o, x + y)

    # Compile and launch on a thread that then exits. spirv_utils owns a non-POD
    # thread_local, so glibc pins the module against the thread that first touched it and
    # defers any unload until that thread is gone -- doing this on the main thread would
    # mask a reintroduced unload and make this test silently green.
    worker = threading.Thread(target=workload)
    worker.start()
    worker.join()

    launchers = [o for o in gc.get_objects()
                 if type(o) is drv.XPULauncher and getattr(o, "arg_annotations", None)]
    assert launchers, "no XPULauncher with annotations -- test would prove nothing"

    # A GC-tracked tuple standing in for the Triton/inductor caches that keep
    # PyKernelArg instances alive in a long-running process. The list co-resident in the
    # tuple is what keeps it tracked; a tuple of only untracked items gets untracked.
    HOLDER = (0, [ln.arg_annotations for ln in launchers], drv.PyKernelArg)
    assert gc.is_tracked(HOLDER)

    assert mapped(), "spirv_utils should be mapped after running a kernel"

    # Do what interpreter finalization does, then invoke any destructor that exists --
    # CPython would run it at a moment of its own choosing, which is precisely why its
    # existence is the bug.
    spirv = triton.runtime.driver.active.utils.load_binary.__self__
    destructor = getattr(type(spirv), "__del__", None)
    drv.XPUUtils._instance = None
    if destructor is not None:
        destructor(spirv)

    gc.collect()
    gc.collect()

    assert mapped(), "spirv_utils was unloaded while PyKernelArg instances were alive"
    print("OK")
    """)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/maps")
def test_spirv_utils_survives_gc_after_teardown(tmp_path):
    """A GC pass after driver teardown must not fault: the module stays mapped.

    Reintroducing `SpirvUtils.__del__` makes this child die with SIGSEGV (returncode
    -11) instead of exiting cleanly -- verified on PVC against both revisions.
    """
    script = tmp_path / "child.py"
    script.write_text(_CHILD)

    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONFAULTHANDLER": "1"})

    assert result.returncode == 0, (f"child exited with {result.returncode} "
                                    f"(negative means a fatal signal; -11 is the #7682 SIGSEGV)\n"
                                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "OK" in result.stdout, f"child did not reach the end:\n{result.stdout}\n{result.stderr}"
