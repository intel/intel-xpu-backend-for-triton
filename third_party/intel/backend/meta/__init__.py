import os

if (arch := os.getenv("TRITON_INTEL_DEVICE_ARCH", "").strip().lower()):
    from importlib import import_module

    XPUBackendMeta = import_module(f"{__name__}.{arch}").XPUBackendMeta
