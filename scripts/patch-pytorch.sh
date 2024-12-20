#!/usr/bin/env bash

# This script applies required patches on top of PyTorch cloned repository.
# Needs to be executed in the PyTorch cloned repository.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! $REPO_ROOT ]]; then
    echo "Failed to identify root of the repository."
    exit 1
fi

echo "Applying PyTorch patches in $REPO_ROOT"
cd "$REPO_ROOT"

curl -sSL https://github.com/pytorch/pytorch/pull/126516.diff | git apply -
# REVERT ME: it's just a trigger for pytorch rebuild
git apply "${SCRIPT_DIR}/pytorch.patch"
