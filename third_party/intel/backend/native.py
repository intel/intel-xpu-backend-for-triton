"""The backend's C++ module: `libtriton.intel` in tree, `_C_intel` next to this file when the backend is a package of
its own. If that one cannot be loaded, `intel` is None and `error` says why, so `import triton` keeps working."""
from triton._C import libtriton

intel = getattr(libtriton, "intel", None)
error = None
if intel is None:
    try:
        from . import _C_intel as intel
    except ImportError as e:
        error = f"{type(e).__name__}: {e}"
