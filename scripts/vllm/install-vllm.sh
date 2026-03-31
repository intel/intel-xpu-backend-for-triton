#!/usr/bin/env bash

set -euo pipefail

FORCE_REINSTALL=false
SKIP_INSTALL=false
SMOKE_TEST=false
VENV=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-reinstall)
      FORCE_REINSTALL=true
      shift
      ;;
    --skip-install)
      # Clone and patch only, do not python -m pip install
      SKIP_INSTALL=true
      shift
      ;;
    --smoke-test)
      # Run smoke tests after installation (requires GPU)
      SMOKE_TEST=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --help)
      cat <<EOF
Usage: ./install-vllm.sh [options]

Options:
  --force-reinstall  Force reinstallation even if vLLM is already installed.
  --skip-install     Clone and patch only, skip python -m pip install.
  --smoke-test       Run smoke tests after install (requires XPU device).
  --venv             Activate .venv/ before installation (uv/venv compatible).
  --help             Show this help message and exit.

Examples:
  ./install-vllm.sh --venv
  ./install-vllm.sh --venv --force-reinstall
  ./install-vllm.sh --venv --smoke-test
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1."
      exit 1
      ;;
  esac
done

if [ "$VENV" = true ]; then
  echo "**** Activating virtual environment ****"
  if [[ $OSTYPE = msys ]]; then
    source .venv/Scripts/activate
  else
    source .venv/bin/activate
  fi
fi

# intel-xpu-backend-for-triton project root
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPTS_DIR=$ROOT/scripts
VLLM_PROJ=$ROOT/vllm
VLLM_PIN="$(<"$ROOT/benchmarks/vllm/vllm-pin.txt")"

echo "**** vLLM pin: $VLLM_PIN ****"

############################################################################
# Smoke test function (defined early so it can be called from early-exit path)

function smoke_test_vllm {
  echo "**** Running vLLM smoke tests ****"

  python -c "import torch; print(f'torch={torch.__version__}, xpu={torch.xpu.is_available()}')"
  python -c "import triton; print(f'triton={triton.__version__}')"
  python -c "import vllm; print(f'vllm={getattr(vllm, \"__version__\", \"installed (no version attr)\")}')"

  python -c "
import torch
if torch.xpu.is_available():
    print(f'XPU device: {torch.xpu.get_device_name(0)}')
    x = torch.randn(4, 4, device='xpu')
    print(f'Tensor on XPU: {x.device}')
else:
    print('WARNING: No XPU device available')
"

  python -c "
from vllm.platforms import current_platform
print(f'Platform: {current_platform.device_type}')
"

  echo "**** Smoke tests passed ****"
}

############################################################################
# Check if already installed

if python -m pip show vllm >/dev/null 2>&1; then
  if [ "$FORCE_REINSTALL" = false ]; then
    echo "**** vLLM is already installed, skipping. ****"
    echo "**** Use --force-reinstall to force reinstallation. ****"
    echo "**** To get a clean install: rm -rf $VLLM_PROJ && python -m pip uninstall -y vllm ****"
    if [ "$SMOKE_TEST" = true ]; then
      smoke_test_vllm
    fi
    exit 0
  fi
  echo "**** --force-reinstall: uninstalling existing vLLM. ****"
  python -m pip uninstall -y vllm
fi

############################################################################
# Clone and checkout pinned commit

function clone_vllm {
  echo "**** Cloning vLLM into $VLLM_PROJ ****"
  git clone https://github.com/vllm-project/vllm.git "$VLLM_PROJ"
  cd "$VLLM_PROJ"
  git checkout "$VLLM_PIN"
}

if [ -d "$VLLM_PROJ" ]; then
  if [ "$FORCE_REINSTALL" = true ]; then
    echo "**** Removing existing $VLLM_PROJ ****"
    rm -rf "$VLLM_PROJ"
    clone_vllm
  else
    echo "**** Reusing existing $VLLM_PROJ directory. ****"
  fi
else
  clone_vllm
fi

############################################################################
# Patch for XPU

function patch_vllm {
  cd "$VLLM_PROJ"

  # Apply the main vLLM fix patch (conftest, batched_moe, fused_batched_moe, etc.)
  if git apply --check "$ROOT/benchmarks/vllm/vllm-fix.patch" 2>/dev/null; then
    git apply "$ROOT/benchmarks/vllm/vllm-fix.patch"
    echo "**** Applied vllm-fix.patch ****"
  else
    echo "**** vllm-fix.patch already applied or conflicts, skipping. ****"
  fi

  # Targeted sed transformations for kernel test files
  sed -i 's/device="cuda"/device="xpu"/g' \
    tests/kernels/moe/utils.py \
    tests/kernels/attention/test_triton_unified_attention.py
  sed -i 's/set_default_device("cuda")/set_default_device("xpu")/g' \
    tests/kernels/attention/test_triton_unified_attention.py

  # AST-based XPU adaptation for spec_decode/mrv2 test files
  python "$SCRIPTS_DIR/vllm/vllm_xpu_patch.py" "$VLLM_PROJ"

  echo "**** vLLM patched for XPU ****"
}

patch_vllm

############################################################################
# Install

if [ "$SKIP_INSTALL" = true ]; then
  echo "**** --skip-install: skipping python -m pip install. ****"
  exit 0
fi

function install_vllm {
  cd "$VLLM_PROJ"

  # Verify torch is pre-installed (from nightly wheels)
  if ! python -m pip show torch >/dev/null 2>&1; then
    echo "ERROR: torch must be installed before running this script."
    echo "Install nightly wheels first: triton-utils wheels -D ./wheels --wheel-set torch"
    exit 1
  fi

  # vLLM installs pytest-shard which conflicts with pytest-skip.
  # We need pytest-skip for Triton skip lists (--skip-from-file).
  # Uninstall pytest-shard instead - we don't need it for our runs.
  python -m pip uninstall pytest-shard -y 2>/dev/null || true

  # Strip torch/triton pins from requirements (use pre-installed nightly wheels)
  # Be precise: match torch/torchaudio/torchvision/triton but NOT tritonclient
  sed -i '/^torch[=>= ]/d; /^torchaudio/d; /^torchvision/d; /^triton[=>= ]/d; /extra-index-url.*pytorch/d' requirements/xpu.txt
  sed -i '/^torch[=>= ]/d; /^torchaudio/d; /^torchvision/d; /^triton[=>= ]/d' requirements/test.in

  # Create constraints file to prevent pip from replacing pre-installed torch
  # with a PyPI version. common.txt -> transformers -> torch is the main culprit.
  CONSTRAINTS=$(mktemp)
  python -m pip freeze | grep -iE '^(torch|triton)' > "$CONSTRAINTS" || true
  echo "**** Using constraints: $(cat "$CONSTRAINTS") ****"

  # Dry-run first: verify pip won't replace torch/triton with unwanted versions
  echo "**** Dry-run: checking for unintended torch/triton replacements ****"
  DRY_OUTPUT=$(python -m pip install --dry-run -c "$CONSTRAINTS" -r requirements/xpu.txt 2>&1) || true
  if echo "$DRY_OUTPUT" | grep -iE "Would install.*(torch|triton)" | grep -ivE "torchvision|torchaudio|tritonclient"; then
    echo "WARNING: pip would install/replace torch or triton packages:"
    echo "$DRY_OUTPUT" | grep -iE "torch|triton"
    echo ""
    echo "This likely means a transitive dependency is pulling in torch."
    echo "Check constraints file: $CONSTRAINTS"
    echo "Aborting to prevent nightly wheels from being overwritten."
    rm -f "$CONSTRAINTS"
    exit 1
  fi
  echo "**** Dry-run passed: torch/triton will not be replaced ****"

  # Install XPU requirements with torch constrained
  python -m pip install -c "$CONSTRAINTS" -r requirements/xpu.txt

  # Install XPU test requirements (locked, stripped of pytest-shard and python-incompatible pins)
  sed -i '/pytest-shard/d' requirements/xpu-test.txt
  sed -i '/^torch[=>= ]/d; /^torchaudio/d; /^torchvision/d; /^triton[=>= ]/d' requirements/xpu-test.txt
  # gpt-oss requires Python 3.12+; skip on 3.10
  sed -i '/^gpt-oss/d' requirements/xpu-test.txt
  python -m pip install -c "$CONSTRAINTS" -r requirements/xpu-test.txt

  # Install additional test dependencies not covered by requirements
  python -m pip install cachetools cbor2 blake3 pybase64 openai_harmony tblib

  rm -f "$CONSTRAINTS"

  # Copy tests for benchmark use
  rm -rf "$ROOT/benchmarks/vllm/batched_moe/tests"
  cp -r tests "$ROOT/benchmarks/vllm/batched_moe/tests"

  # Install vLLM in editable mode (--no-deps: don't resolve deps again)
  VLLM_TARGET_DEVICE=xpu python -m pip install --no-deps --no-build-isolation -e .

  echo "**** vLLM installed successfully ****"
}

install_vllm

if [ "$SMOKE_TEST" = true ]; then
  smoke_test_vllm
fi
