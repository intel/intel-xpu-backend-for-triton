"""Prints a lib directory for pti."""

import importlib.metadata
import pathlib


def get_pti_lib_path() -> pathlib.Path:
    """Returns library path for pti.

    Raises:
        importlib.metadata.PackageNotFoundError: if 'intel-pti' not installed.
        AssertionError: if libpti_view.so not found.
    """
    files = importlib.metadata.files('intel-pti') or []
    for f in files:
        if any(map(lambda el: el in f.name, ('libpti_view.so', 'pti_view.lib'))):  # pylint: disable=W0640
            return pathlib.Path(f.locate()).parent.resolve()
    raise AssertionError('libpti_view.so not found')


if __name__ == '__main__':
    print(get_pti_lib_path())
