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

# put your patch applies here
curl -sSL https://github.com/pytorch/pytorch/pull/143553.diff | git apply -
