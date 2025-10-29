import os
from triton import knobs

os.environ["TRITON_INTEL_DEVICE_ARCH"] = "cri"  # Hardcoded in the main-cri branch

if knobs.intel.device_arch:
    from importlib import import_module

    try:
        XPUBackendMeta = import_module(f"{__name__}.{knobs.intel.device_arch}").XPUBackendMeta
    except (ImportError, AttributeError):
        ...  # Ignore
