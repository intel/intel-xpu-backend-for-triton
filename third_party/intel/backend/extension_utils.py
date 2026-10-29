"""
Lightweight utility for querying device extensions without initializing the full driver.
This allows checking device capabilities during compilation without requiring
a full PyTorch/SYCL runtime initialization.
"""

from triton.backends.intel.driver import compile_module_from_src

import functools
import multiprocessing
import os
import queue
from pathlib import Path
from triton import knobs

_PROBE_TIMEOUT_SECONDS = 60


def _has_spirv_capability_worker(device_id: int, probe_spirv: bytes, result_queue: multiprocessing.Queue) -> None:
    try:
        extension_checker = _get_extension_checker()
        result_queue.put(extension_checker.probe_spirv_module(device_id, probe_spirv))
    except Exception:
        result_queue.put(None)


@functools.lru_cache(maxsize=1)
def _get_extension_checker():
    """
    Lazily compile and load the extension checker module.
    This is a lightweight C module that can query device extensions
    without requiring full driver initialization.
    """
    dirname = os.path.dirname(os.path.realpath(__file__))
    src = Path(os.path.join(dirname, "extension_utils.c")).read_text()
    return compile_module_from_src(src=src, name="extension_utils_impl")


@functools.lru_cache(maxsize=1)
def get_device_extensions_from_env():
    """
    Get device extensions from environment variable.
    Returns None if not set.
    """
    if knobs.intel.device_extensions:
        return set(knobs.intel.device_extensions.split(" "))
    return None


def has_device_extension(device_id: int, extension_name: str) -> bool:
    """
    Check if a device supports a specific OpenCL extension.

    First checks the TRITON_INTEL_DEVICE_EXTENSIONS environment variable.
    If not set, queries the device directly using a lightweight C utility.

    Args:
        device_id: Device ID to query (from sycl::ext::intel::info::device::device_id)
        extension_name: Name of the extension (e.g., 'cl_intel_subgroup_2d_block_io')

    Returns:
        True/False if the extension is supported/not supported, or
        None if the specific GPU instance is not found.
    """
    # Try environment variable first (fastest path)
    env_extensions = get_device_extensions_from_env()
    if env_extensions is not None:
        return extension_name in env_extensions

    # Fall back to querying the device
    # This requires minimal SYCL initialization (no PyTorch dependency)
    extension_checker = _get_extension_checker()
    return extension_checker.check_extension(device_id, extension_name.encode())


@functools.lru_cache(maxsize=None)
def has_spirv_capability(device_id: int, probe_spirv: bytes) -> bool | None:
    """
    Ground-truth probe for a SPIR-V capability/extension, via the driver's own JIT.

    Unlike `has_device_extension`, which only checks what the driver
    *advertises*, this feeds `probe_spirv` -- a minimal SPIR-V module
    declaring only the capability/extension under test -- to `zeModuleCreate`,
    the same call `load_binary` makes for a real kernel, so a rolling driver
    that hasn't advertised a newly-landed capability yet is still measured
    correctly.

    Args:
        device_id: Device ID to query (from sycl::ext::intel::info::device::device_id)
        probe_spirv: a minimal SPIR-V module declaring only the capability/extension to test.

    Returns:
        True/False if the driver accepted/rejected the probe module, or None
        if the device isn't found. A crashed or hung probe is reported as
        False, not None: the crash itself proves the capability unsafe to use.
    """
    # An unsupported extension can abort the whole process instead of
    # returning a clean error
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_has_spirv_capability_worker, args=(device_id, probe_spirv, result_queue))
    proc.start()
    proc.join(_PROBE_TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    result = False
    if not proc.exitcode:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            pass
    return result


def query_device_extensions(device_id: int):
    """
    Query all relevant device extensions.

    All GPUs with the same device_id have the same extensions,
    so we query the first available GPU device.

    Args:
        device_id: Device ID from sycl::ext::intel::info::device::device_id

    Returns:
        Dictionary with extension capabilities
    """
    extensions = {
        "has_subgroup_matrix_multiply_accumulate":
        has_device_extension(device_id, "cl_intel_subgroup_matrix_multiply_accumulate"),
        "has_subgroup_matrix_multiply_accumulate_tensor_float32":
        has_device_extension(device_id, "cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32"),
        "has_2d_block_io":
        has_device_extension(device_id, "cl_intel_subgroup_2d_block_io"),
        "has_bfloat16_conversion":
        has_device_extension(device_id, "cl_intel_bfloat16_conversions"),
    }
    return {extension: supported for extension, supported in extensions.items() if supported is not None}


def get_device_id(device_idx: int) -> int:
    """
    Get the Intel device ID for a device at the given index.

    Args:
        device_idx: Index of the device in the device list (0-based)

    Returns:
        Device ID from sycl::ext::intel::info::device::device_id

    Raises:
        RuntimeError: If device_idx is out of range
    """
    extension_checker = _get_extension_checker()
    return extension_checker.get_device_id(device_idx)
