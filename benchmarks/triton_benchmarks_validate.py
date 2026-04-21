"""Lightweight validation of triton_kernels_benchmark C++ extensions.

This module lives OUTSIDE the triton_kernels_benchmark package so that it can be
imported and executed without triggering the package's ``__init__.py``, which
pulls in triton and torch.  This is critical for CI environments (e.g. the
build-benchmarks-wheel workflow) where the benchmarks wheel is installed but
triton is not.
"""

import importlib.util
import os
import sys
import sysconfig

REQUIRED_CPP_EXTENSIONS = ("xetla_kernel", "sycl_tla_kernel", "onednn_kernel")


def _find_package_dir():
    """Locate the triton_kernels_benchmark package directory without importing it."""
    spec = importlib.util.find_spec("triton_kernels_benchmark")
    if spec is not None and spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    # Fallback: assume the package is a sibling of this file.
    return os.path.join(os.path.dirname(__file__), "triton_kernels_benchmark")


def validate_cpp_extensions(pkg_dir=None):
    """Validate that required C++ extension .so files are present in the package.

    Args:
        pkg_dir: Override for the package directory to check.  Defaults to the
            installed ``triton_kernels_benchmark`` directory next to this file.
    """
    if pkg_dir is None:
        pkg_dir = _find_package_dir()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Cannot determine platform extension suffix (EXT_SUFFIX)")
    missing = [name for name in REQUIRED_CPP_EXTENSIONS if not os.path.exists(os.path.join(pkg_dir, name + ext_suffix))]
    if missing:
        raise RuntimeError(f"Missing C++ extensions: {missing}")
    print("Benchmarks validation passed: all C++ extensions present")


def main():
    """Entry point for ``triton-benchmarks-validate`` console script."""
    try:
        validate_cpp_extensions()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
