#!/usr/bin/env bash

# This script applies required patches on top of PyTorch cloned repository.
# Needs to be executed in the PyTorch cloned repository.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! $REPO_ROOT ]]; then
    echo "Failed to identify root of the repository."
    exit 1
fi

# Fetches a file to SCRIPTS_DIR with retries to avoid GitHub secondary rate limit.
# https://github.com/intel/intel-xpu-backend-for-triton/issues/4074.
fetch_patch() {(
    cd "$SCRIPTS_DIR"
    curl --retry 10 -sSLO "$1"
)}

# Apply a remote patch from URL or local patch from SCRIPTS_DIR
apply_patch() {
    echo "Applying patch $1"
    cd "$REPO_ROOT"
    if [[ -f $SCRIPTS_DIR/$1 ]]; then
        git apply "$SCRIPTS_DIR/$1"
    else
        fetch_patch "$1"
        git apply "$SCRIPTS_DIR/$(basename "$1")"
    fi
}

echo "Applying PyTorch patches in $REPO_ROOT"

# put your patch applies here
apply_patch ./patch/flex_attn_143553.patch
apply_patch ./patch/flex_decoding.patch
