# Triton XPU Compatibility Shim

This package exists solely to satisfy package managers expecting
a distribution named `triton`.

Installing:

    pip install --extra-index-url https://download.pytorch.org/whl/xpu triton==3.7.2+xpu
automatically installs:

    triton-xpu==3.7.2

from the PyTorch XPU wheel index.

This package contains no Triton implementation and no Python
module named `triton`.

It is a temporary compatibility layer until upstream packaging
converges on a unified Triton package naming scheme.
