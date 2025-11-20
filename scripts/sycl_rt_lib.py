"""Prints a lib directory for intel-sycl-rt."""

import importlib.metadata
import pathlib


def get_sycl_rt_lib_path() -> pathlib.Path:
    """Returns library path for intel-sycl-rt.

    Raises:
        importlib.metadata.PackageNotFoundError: if intel-sycl-rt not installed.
        AssertionError: if libsycl.so not found.
    """
    files = importlib.metadata.files('intel-sycl-rt') or []
    for f in files:
        if 'libsycl.so' in f.name:
            return pathlib.Path(f.locate()).parent.resolve()
    raise AssertionError('libsycl.so not found')


if __name__ == '__main__':
    print(get_sycl_rt_lib_path())
