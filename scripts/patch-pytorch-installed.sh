#!/usr/bin/env bash

# This script applies required patches on top of an installed PyTorch wheel.
# Unlike patch-pytorch.sh which operates on a cloned git repo, this script
# patches Python files in-place within site-packages.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python3}"

SITE_ROOT=$("$PYTHON" -c "
import pathlib, importlib.util
spec = importlib.util.find_spec('torch')
if spec is None or spec.origin is None:
    raise SystemExit('ERROR: torch not found in current Python environment')
print(pathlib.Path(spec.origin).parent.parent)
") || {
  echo "ERROR: Could not find PyTorch installation."
  exit 1
}

apply_patch() {
  echo "Applying patch $1"
  if [[ -f "$SCRIPTS_DIR/$1" ]]; then
    patch -p1 -d "$SITE_ROOT" < "$SCRIPTS_DIR/$1"
  else
    echo "ERROR: Patch file not found: $SCRIPTS_DIR/$1"
    return 1
  fi
}

echo "Applying PyTorch patches in $SITE_ROOT"

# put your patch applies here
apply_patch ./patch/inductor_autotune_cache_dir.patch
