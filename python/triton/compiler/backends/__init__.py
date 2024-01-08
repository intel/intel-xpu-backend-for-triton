from .cuda import CUDABackend
from .xpu import XPUBackend


def make_backend(target):
    return {"cuda": CUDABackend, "xpu": XPUBackend}[target[0]](target)
