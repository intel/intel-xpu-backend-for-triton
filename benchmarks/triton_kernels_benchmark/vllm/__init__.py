import importlib

from triton_kernels_benchmark.benchmark_testing import DEVICE


def import_xpu_only(qualified_name: str):
    """Import `<module>.<attribute>` from a package that only exists on XPU (e.g.
    `vllm_xpu_kernels`), or return None on CUDA."""
    if DEVICE != "xpu":
        return None
    module_name, _, attribute = qualified_name.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)
