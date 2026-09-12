#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly DEFAULT_BRANCH="main"
readonly SCRIPTS_DIR="$ROOT/scripts"
readonly VLLM_PROJ="$ROOT/vllm"

# Provides the `pip` wrapper (pip or `uv pip`).
source "$SCRIPTS_DIR/pip-utils.sh"

# Check if the specified package is installed and matches the pinned commit.
# Returns 0 if the installed package is correct, 1 if it needs to be installed/reinstalled.
check_installed_package() {
  local package="$1"
  local pinned_commit="$2"
  local force="$3"
  local latest="$4"

  if ! pip show "$package" &>/dev/null; then
    return 1
  fi

  if [[ "$latest" == true ]]; then
    echo "*** --latest specified: ignoring installed $package. ***"
    pip uninstall -y "$package"

    return 1
  fi

  local current_commit="$(pip show "$package" | awk '/^Version:/ {print $2}')"
  current_commit="${current_commit#*+g}"
  current_commit="${current_commit%%.*}"
  echo "*** $package is installed at commit: $current_commit. ***"

  if [[ "$pinned_commit" == "$current_commit"* ]]; then
    if [[ "$force" == false ]]; then
      echo "*** Installed $package matches the pinned commit: $pinned_commit. ***"
      return 0
    fi

    echo "*** --force-reinstall specified: ignoring installed $package. ***"
  else
    echo "*** Installed $package commit ($current_commit) does not match the pinned commit ($pinned_commit). ***"
  fi

  if [[ "$force" == false ]]; then
    echo "ERROR: Installed $package does not match the pinned commit and --force-reinstall is not specified." >&2
    exit 1
  fi

  pip uninstall -y "$package"

  return 1
}

show_installs() {
  echo "*** Installed versions: ***"
  echo "vllm: $(pip show vllm | awk '/^Version:/ {print $2}')."
  echo "vllm-xpu-kernels: $(pip show vllm-xpu-kernels | awk '/^Version:/ {print $2}')."
}

update_submodules_and_clean() {
  local repo_dir="$1"

  git -C "$repo_dir" submodule update --init --recursive
  git -C "$repo_dir" clean -xffd
}

# Clone the repository at the specified commit, or main if no commit is provided.
# Returns 0 on success, 1 on failure.
clone_repo() {
  local target_dir="$1"
  local repo_url="$2"
  local pinned_commit="$3"
  local latest="$4"

  rm -rf "$target_dir"
  git clone --single-branch -b "$DEFAULT_BRANCH" "$repo_url" "$target_dir"

  if [[ "$latest" == false ]]; then
    git -C "$target_dir" checkout "$pinned_commit"
  fi

  update_submodules_and_clean "$target_dir"
}

# Prepare the source code for the project by cloning or resetting to the pinned commit.
# Returns 0 on success, 1 on failure.
prepare_source() {
  local target_dir="$1"
  local repo_url="$2"
  local pinned_commit="$3"
  local latest="$4"

  local needs_clone=true
  if [[ -d "$target_dir" ]]; then
    if [[ "$clean" == false ]]; then
      echo "*** --no-clean specified: reusing source at $target_dir without cleanup. ***"
      return 0
    fi

    local reset_ref="${pinned_commit:-$DEFAULT_BRANCH}"
    if [[ "$latest" == true ]]; then
      echo "*** --latest specified: resetting to the latest commit on $DEFAULT_BRANCH. ***"
      reset_ref="origin/$DEFAULT_BRANCH"
    fi

    if (git -C "$target_dir" fetch --recurse-submodules && \
        git -C "$target_dir" reset --hard "$reset_ref" && \
        update_submodules_and_clean "$target_dir"); then
      needs_clone=false
    fi
  fi

  if [[ "$needs_clone" == true ]]; then
    clone_repo "$target_dir" "$repo_url" "$pinned_commit" "$latest"
  fi
}

# Install vLLM in editable mode from the source directory.
#
# `vllm_xpu_kernels` is deliberately left in `requirements/xpu.txt`: vLLM pins an
# exact kernels release there and publishes a matching wheel on its own index, so
# the kernels are installed from vLLM's pin instead of being built here.
install_vllm() {
  if [[ ! -d "$VLLM_PROJ/tests" ]]; then
    echo "ERROR: tests dir not found in vLLM." >&2
    exit 1
  fi

  sed -i \
    -e '/^pytest-shard/d' \
    -e '/^torch/d' \
    -e '/^triton/d' \
    -e '/^xgrammar/d' \
    -e '/^--extra-index-url.*https:\/\/download\.pytorch\.org\/whl/d' \
    "$VLLM_PROJ/requirements/xpu.txt"
  pip install -r "$VLLM_PROJ/requirements/xpu.txt"

  VLLM_TARGET_DEVICE=xpu pip install --no-deps --no-build-isolation -e "$VLLM_PROJ"
}

cd "$ROOT"

prepare_source_only=false
latest=false
force_reinstall=false
use_venv=false
clean=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-source)
      prepare_source_only=true
      shift
      ;;
    --latest)
      latest=true
      shift
      ;;
    --force-reinstall)
      force_reinstall=true
      shift
      ;;
    --venv)
      use_venv=true
      shift
      ;;
    -nc|--no-clean)
      clean=false
      shift
      ;;
    --help)
      cat <<EOF
Usage: $0 [options]

vLLM is built and installed from source as an editable install. vLLM XPU kernels
are installed from the release that vLLM pins in requirements/xpu.txt.

Options:
  --prepare-source               Prepare vLLM source only (clone/reset + patch), without build/install. With
                                 --no-clean and an existing source tree, checkout/reset and patching are
                                 skipped and the tree is reused as-is.

  --latest                       Build vLLM from the latest commit in the $DEFAULT_BRANCH branch.

  --force-reinstall              Force reinstallation of vLLM.

  --venv                         Activate Python virtual environment from .venv/ before installation.

  -nc, --no-clean                Reuse existing vLLM source tree without cleanup; skips checkout/reset and
                                 patching when source exists.

  --help                         Show this help message and exit.

Examples:
  ./install-vllm.sh
  ./install-vllm.sh --prepare-source
  ./install-vllm.sh --prepare-source --latest
  ./install-vllm.sh --latest --venv
EOF
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1." >&2
      exit 1
      ;;
  esac
done

if [[ "$use_venv" == true ]]; then
  echo "*** --venv specified: activating virtual environment from .venv. ***"
  source .venv/bin/activate
fi

vllm_pinned_commit=""

if [[ "$latest" == false ]]; then
  vllm_pinned_commit="$(<"$SCRIPTS_DIR/vllm/vllm-pin.txt")"
  echo "*** Using the pinned vllm commit: $vllm_pinned_commit. ***"
fi

if [[ "$prepare_source_only" == false ]]; then
  if check_installed_package "vllm" "${vllm_pinned_commit:-}" "$force_reinstall" "$latest"; then
    show_installs

    echo "*** vllm is installed at the correct commit. ***"
    exit 0
  fi
fi

echo "*** Base directory: $ROOT. ***"
echo "*** vLLM project: $VLLM_PROJ. ***"

prepare_source "$VLLM_PROJ" "https://github.com/vllm-project/vllm.git" "${vllm_pinned_commit:-}" "$latest"
if [[ "$clean" == true ]]; then
  git -C "$VLLM_PROJ" apply "$SCRIPTS_DIR/vllm/vllm-fix.patch"
  python "$SCRIPTS_DIR/vllm/vllm_xpu_patch.py" "$VLLM_PROJ"
fi

if [[ "$prepare_source_only" == true ]]; then
  echo "*** vLLM source prepared at $VLLM_PROJ. ***"
  echo "*** Current commit: $(git -C "$VLLM_PROJ" rev-parse HEAD). ***"
  exit 0
fi

install_vllm

show_installs
