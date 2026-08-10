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

## Verifying the wheel signature

CI builds this package in the `wheels-triton-shim.yml` workflow and signs each
wheel with keyless [Sigstore](https://www.sigstore.dev/) (OIDC, no stored keys).
Every `*.whl` is published alongside a `*.whl.sigstore.json` bundle.

Verify a downloaded wheel with [cosign](https://github.com/sigstore/cosign):

    cosign verify-blob triton-*.whl \
      --bundle triton-*.whl.sigstore.json \
      --certificate-oidc-issuer https://token.actions.githubusercontent.com \
      --certificate-identity-regexp \
        'https://github.com/intel/intel-xpu-backend-for-triton/.github/workflows/wheels-triton-shim.yml@.*'

Or with the [sigstore](https://pypi.org/project/sigstore/) Python CLI:

    sigstore verify identity triton-*.whl \
      --bundle triton-*.whl.sigstore.json \
      --cert-oidc-issuer https://token.actions.githubusercontent.com \
      --cert-identity \
        'https://github.com/intel/intel-xpu-backend-for-triton/.github/workflows/wheels-triton-shim.yml@refs/heads/main'

A successful check prints `Verified OK`.
