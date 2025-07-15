#!/usr/bin/env bash

# Script to get and set Triton version.
# Currently Triton does not have a single file to set a version. There are two files:
# 1. In `setup.py`, the version is set to `{version}+git{sha}` and this version is used for the
#    Python package/wheel.
# 2. In `python/triton/__init__.py`, the version is set to `{version}` and used in runtime for
#    `triton.__version__`.
# When building a wheel, the both files need to be updated to use the same version.

set -euo pipefail

PROJECT_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )"

get_version() {
    # Use setup.py, as the most reliable source
    sed -n -E "s/^TRITON_VERSION = \"([^\"]+)\".*/\1/p" "$PROJECT_ROOT/setup.py"
}

set_version() {
    local version="$1"
    sed -i -E "s/^TRITON_VERSION = \"([^\"]+)\".*/TRITON_VERSION = \"$version\"/" "$PROJECT_ROOT/setup.py"
    sed -i -E "s/^__version__ = '[^']+'/__version__ = '$version'/" "$PROJECT_ROOT/python/triton/__init__.py"
}

if (( $# == 1 )); then
    set_version "$1"
else
    get_version
fi
