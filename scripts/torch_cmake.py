"""Prints cmake directory for PyTorch."""

import importlib.metadata
import pathlib


def get_torch_cmake_path() -> pathlib.Path:
    """Returns directory that contains TorchConfig.cmake.

    Raises:
        importlib.metadata.PackageNotFoundError: if torch not installed.
        AssertionError: if TorchConfig.cmake not found.
    """
    files = importlib.metadata.files('torch') or []
    for f in files:
        if 'TorchConfig.cmake' in f.name:
            return pathlib.Path(f.locate()).parent.resolve()
    raise AssertionError('TorchConfig.cmake not found')


if __name__ == '__main__':
    print(get_torch_cmake_path())
