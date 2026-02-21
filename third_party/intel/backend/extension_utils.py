"""
Lightweight utility for querying device extensions without initializing the full driver.
This allows checking device capabilities during compilation without requiring
a full PyTorch/SYCL runtime initialization.
"""

import functools
import os
from pathlib import Path
from triton import knobs


@functools.lru_cache(maxsize=1)
def _get_extension_checker():
    """
    Lazily compile and load the extension checker module.
    This is a lightweight C module that can query device extensions
    without requiring full driver initialization.
    """
    try:
        from . import driver
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, "extension_utils.c")).read_text()
        return driver.compile_module_from_src(src=src, name="extension_utils_impl")
    except Exception:
        # Silently return None if extension checker can't be loaded
        return None


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
        device_id: Device ID to query
        extension_name: Name of the extension (e.g., 'cl_intel_subgroup_2d_block_io')

    Returns:
        True if the extension is supported, False otherwise
    """
    # Try environment variable first (fastest path)
    env_extensions = get_device_extensions_from_env()
    if env_extensions is not None:
        return extension_name in env_extensions

    # Fall back to querying the device
    # This requires minimal SYCL initialization (no PyTorch dependency)
    extension_checker = _get_extension_checker()
    if extension_checker is None:
        return False

    try:
        return extension_checker.check_extension(device_id, extension_name.encode())
    except (AttributeError, RuntimeError):
        # If we can't check, assume not supported
        return False


def query_device_extensions(device_id: int = 0):
    """
    Query all relevant device extensions for a given device.

    Args:
        device_id: Device ID to query (default: 0)

    Returns:
        Dictionary with extension capabilities
    """
    extensions = {
        "has_subgroup_matrix_multiply_accumulate":
        has_device_extension(device_id, "cl_intel_subgroup_matrix_multiply_accumulate"),
        "has_subgroup_matrix_multiply_accumulate_tensor_float32":
        has_device_extension(device_id, "cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32"),
        "has_subgroup_2d_block_io":
        has_device_extension(device_id, "cl_intel_subgroup_2d_block_io"),
        "has_bfloat16_conversion":
        has_device_extension(device_id, "cl_intel_bfloat16_conversions"),
    }
    return extensions
