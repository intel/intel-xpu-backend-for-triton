#!/usr/bin/env bash

# Install vllm-omni and its dependency closure. Run this only in the vllm-omni test suite as it may modify
# installed packages in the current Python environment in case of differences with pinned vllm dependencies.

set -euo pipefail

readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly SCRIPTS_DIR="$ROOT/scripts"
readonly VLLM_OMNI_PROJ="$ROOT/vllm-omni"

source "$SCRIPTS_DIR/pip-utils.sh"

# Installed from source Packages for XPU.
readonly PROTECTED_PACKAGES=(torch triton pytorch-triton-xpu torchvision torchaudio vllm)

# Print "<name>==<version>" for each installed protected package, skipping the ones that are absent.
protected_versions() {
  local package version

  for package in "${PROTECTED_PACKAGES[@]}"; do
    if version="$(pip show "$package" 2>/dev/null | awk '/^Version:/ { print $2 }')" \
      && [[ -n "$version" ]]; then
      echo "$package==$version"
    fi
  done
}

# vllm-omni imports vllm at module scope without declaring it. It also needs torchvision for its
# cosmos-guardrail requirement, torchvision on PyPI pins torch==<exact release> which will pull
# a PyPI torch and override the installed from source one.
for package in vllm torchvision; do
  if ! pip show "$package" >/dev/null; then
    echo "ERROR: $package must be installed from source before installing vllm-omni." >&2
    exit 1
  fi
done

rm -rf "$VLLM_OMNI_PROJ"
git clone --single-branch -b main https://github.com/vllm-project/vllm-omni.git "$VLLM_OMNI_PROJ"
git -C "$VLLM_OMNI_PROJ" checkout "$(<"$SCRIPTS_DIR/vllm/vllm-omni-pin.txt")"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
constraints="$work_dir/constraints.txt"

protected_versions | tee "$constraints"

VLLM_OMNI_TARGET_DEVICE=xpu pip install -c "$constraints" "$VLLM_OMNI_PROJ"

if ! protected_versions | diff -u "$constraints" -; then
  echo "ERROR: installing vllm-omni changed the XPU stack (see the diff above)." >&2
  exit 1
fi

# vllm-omni picks its platform at runtime, so a replaced torch would silently run
# on CPU instead of failing the install -- assert the platform is XPU.
cd "$work_dir"
python - <<'PY'
import torch

from vllm_omni.platforms import resolve_current_omni_platform_cls_qualname

if not torch.xpu.is_available():
    raise SystemExit(f"ERROR: torch.xpu.is_available() is False (torch {torch.__version__}).")

platform = resolve_current_omni_platform_cls_qualname()
if "xpu" not in platform.lower():
    raise SystemExit(f"ERROR: vllm-omni resolved to {platform}, not an XPU platform.")

print(f"vllm-omni on {torch.xpu.get_device_name(0)} via {platform}")
PY
