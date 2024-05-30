import sys
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    mod = sys.modules[triton.runtime.driver.__module__]
    assert isinstance(triton.runtime.driver.active, getattr(mod, "LazyProxy"))
    assert triton.runtime.driver.active._obj is None
    utils = triton.runtime.driver.active.utils  # noqa: F841
    assert issubclass(triton.runtime.driver.active._obj.__class__, getattr(triton.backends.driver, "DriverBase"))
