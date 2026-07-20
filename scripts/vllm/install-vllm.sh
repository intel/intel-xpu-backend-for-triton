#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
readonly DEFAULT_BRANCH="main"
readonly SCRIPTS_DIR="$ROOT/scripts"
readonly VLLM_PROJ="$ROOT/vllm"
readonly VLLM_XPU_KERNELS_PROJ="$ROOT/vllm-xpu-kernels"

# Check if the specified package is installed and matches the pinned commit.
# Returns 0 if the installed package is correct, 1 if it needs to be installed/reinstalled.
check_installed_package() {
  local package="$1"
  local pinned_commit="$2"
  local force="$3"
  local latest="$4"

  if ! python -m pip show "$package" &>/dev/null; then
    return 1
  fi

  if [[ "$latest" == true ]]; then
    echo "*** --latest specified: ignoring installed $package. ***"
    python -m pip uninstall -y "$package"

    return 1
  fi

  local current_commit="$(python -m pip show "$package" | awk '/^Version:/ {print $2}')"
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

  python -m pip uninstall -y "$package"

  return 1
}

show_installs() {
  echo "*** Installed versions: ***"
  echo "vllm: $(python -m pip show vllm | awk '/^Version:/ {print $2}')."
  echo "vllm-xpu-kernels: $(python -m pip show vllm-xpu-kernels | awk '/^Version:/ {print $2}')."
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
install_vllm() {
  if [[ ! -d "$VLLM_PROJ/tests" ]]; then
    echo "ERROR: tests dir not found in vLLM." >&2
    exit 1
  fi

  sed -i \
    -e '/^pytest-shard/d' \
    -e '/^torch/d' \
    -e '/^triton/d' \
    -e '/^vllm[_-]xpu[_-]kernels/d' \
    -e '/^xgrammar/d' \
    -e '/^--extra-index-url.*https:\/\/download\.pytorch\.org\/whl/d' \
    "$VLLM_PROJ/requirements/xpu.txt"
  python -m pip install -r "$VLLM_PROJ/requirements/xpu.txt"

  VLLM_TARGET_DEVICE=xpu python -m pip install --no-deps --no-build-isolation -e "$VLLM_PROJ"
}

cd "$ROOT"

build_vllm=false
prepare_source_only=false
latest=false
force_reinstall=false
check_wheel=false
use_venv=false
clean=true
triton_repo=intel/intel-xpu-backend-for-triton
triton_repo_branch=$DEFAULT_BRANCH

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      build_vllm=true
      shift
      ;;
    --prepare-source)
      build_vllm=true
      prepare_source_only=true
      shift
      ;;
    --latest)
      build_vllm=true
      latest=true
      shift
      ;;
    --force-reinstall)
      force_reinstall=true
      shift
      ;;
    --check-wheel)
      check_wheel=true
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
    --triton-repo)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --triton-repo requires an argument." >&2
        exit 1
      fi
      triton_repo="$2"
      shift 2
      ;;
    --triton-repo-branch)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --triton-repo-branch requires an argument." >&2
        exit 1
      fi
      triton_repo_branch="$2"
      shift 2
      ;;
    --help)
      cat <<EOF
Usage: $0 [options]

Options:
  --source                       Build vLLM XPU kernels from source using pinned commit.

  --prepare-source               Prepare vLLM and vLLM XPU kernels source only (clone/reset + patch), without
                                 build/install. With --no-clean and an existing source tree, checkout/reset and
                                 patching are skipped and the tree is reused as-is.

  --latest                       Build vLLM and vLLM XPU kernels from the latest commits in the $DEFAULT_BRANCH branch.

  --force-reinstall              Force reinstallation of vLLM and vLLM XPU kernels.

  --check-wheel                  Check if a prebuilt vLLM XPU kernels wheel already exists before building.

  --venv                         Activate Python virtual environment from .venv/ before installation.

  -nc, --no-clean                Reuse existing vLLM and vLLM XPU kernels source trees without cleanup; skips
                                 checkout/reset and patching when source exists.

  --triton-repo <repo>           GitHub repo to fetch prebuilt vLLM XPU kernels wheels from
                                 (default: intel/intel-xpu-backend-for-triton).

  --triton-repo-branch <branch>  Branch to fetch prebuilt vLLM XPU kernels wheels from (default: $DEFAULT_BRANCH).

  --help                         Show this help message and exit.

Examples:
  ./install-vllm.sh --source
  ./install-vllm.sh --prepare-source
  ./install-vllm.sh --prepare-source --latest
  ./install-vllm.sh --latest --venv
  ./install-vllm.sh --triton-repo my_fork/intel-xpu-backend-for-triton --triton-repo-branch dev
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
vllm_xpu_kernels_pinned_commit=""

if [[ "$latest" == false ]]; then
  vllm_pinned_commit="$(<"$SCRIPTS_DIR/vllm/vllm-pin.txt")"
  echo "*** Using the pinned vllm commit: $vllm_pinned_commit. ***"

  vllm_xpu_kernels_pinned_commit="$(<"$SCRIPTS_DIR/vllm/vllm-xpu-kernels-pin.txt")"
  echo "*** Using the pinned vllm-xpu-kernels commit: $vllm_xpu_kernels_pinned_commit. ***"
fi

if [[ "$prepare_source_only" == false ]]; then
  correct_vllm_installed=false
  correct_vllm_xpu_kernels_installed=false

  if check_installed_package "vllm" "${vllm_pinned_commit:-}" "$force_reinstall" "$latest"; then
    correct_vllm_installed=true
  fi

  if check_installed_package "vllm-xpu-kernels" "${vllm_xpu_kernels_pinned_commit:-}" "$force_reinstall" "$latest"; then
    correct_vllm_xpu_kernels_installed=true
  fi

  if [[ "$correct_vllm_installed" == true && "$correct_vllm_xpu_kernels_installed" == true ]]; then
    show_installs

    echo "*** Both vllm and vllm-xpu-kernels are installed at the correct commits. ***"
    exit 0
  fi
fi

prepare_source "$VLLM_PROJ" "https://github.com/vllm-project/vllm.git" "${vllm_pinned_commit:-}" "$latest"
if [[ "$clean" == true ]]; then
  git -C "$VLLM_PROJ" apply "$SCRIPTS_DIR/vllm/vllm-fix.patch"
  python "$SCRIPTS_DIR/vllm/vllm_xpu_patch.py" "$VLLM_PROJ"
fi

if [[ "$build_vllm" == false ]]; then
  if ! command -v gh &>/dev/null; then
    echo "ERROR: gh is not installed." >&2
    exit 1
  fi

  if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is not installed." >&2
    exit 1
  fi

  echo "*** Downloading nightly builds. ***"
  run_id="$(gh run list --workflow nightly-wheels.yml --branch "$triton_repo_branch" -R "$triton_repo" --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')"
  temp_dir="$(mktemp -d)"
  wheel_pattern="wheels-vllm-py$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")-*"

  wheel=""
  if [[ -z "$run_id" || "$run_id" == "null" ]]; then
    echo "*** No successful nightly-wheels.yml run found on '$triton_repo_branch'. ***"
  elif ! gh run download "$run_id" \
    --repo "$triton_repo" \
    --pattern "$wheel_pattern" \
    --dir "$temp_dir"; then
    echo "*** Failed to download vllm-xpu-kernels wheel from run $run_id. ***"
  else
    wheel="$(ls "$temp_dir"/$wheel_pattern/vllm_xpu_kernels-*.whl 2>/dev/null | head -n 1)"
  fi

  # Verify the downloaded wheel's built commit matches the pinned commit before using
  # it. The wheel filename encodes the commit as <version>+g<sha7>.d<date>, matching the
  # version parsing in check_installed_package. Otherwise fall back to building from
  # source so a moved pin is always honored -- e.g. during the window after a pin bump
  # merges to main but before the post-merge main rebuild has succeeded, where
  # `gh run list` still returns the pre-merge (stale) wheel.
  wheel_commit=""
  if [[ -n "$wheel" ]]; then
    wheel_commit="$(basename "$wheel")"
    wheel_commit="${wheel_commit#*+g}"
    wheel_commit="${wheel_commit%%.*}"
  fi

  if [[ -n "$wheel_commit" && "$vllm_xpu_kernels_pinned_commit" == "$wheel_commit"* ]]; then
    echo "*** Downloaded vllm-xpu-kernels wheel commit ($wheel_commit) matches the pinned commit. ***"
    echo "*** Installing vLLM XPU kernels from nightly builds. ***"
    python -m pip install "$wheel"
    rm -rf "$temp_dir"
    echo "*** Installing vLLM from source. ***"
    install_vllm
    show_installs

    exit 0
  fi

  echo "*** Downloaded vllm-xpu-kernels wheel commit (${wheel_commit:-unknown}) does not match the pinned commit ($vllm_xpu_kernels_pinned_commit). ***"
  echo "*** Falling back to building vLLM XPU kernels from source. ***"
  rm -rf "$temp_dir"
  build_vllm=true
fi

echo "*** Base directory: $ROOT. ***"
echo "*** vLLM project: $VLLM_PROJ. ***"
echo "*** vLLM XPU kernels project: $VLLM_XPU_KERNELS_PROJ. ***"

if [[ "$prepare_source_only" == true ]]; then
  prepare_source "$VLLM_XPU_KERNELS_PROJ" "https://github.com/vllm-project/vllm-xpu-kernels.git" "${vllm_xpu_kernels_pinned_commit:-}" "$latest"

  echo "*** vLLM source prepared at $VLLM_PROJ. ***"
  echo "*** Current commit: $(git -C "$VLLM_PROJ" rev-parse HEAD). ***"
  echo "*** vLLM XPU kernels source prepared at $VLLM_XPU_KERNELS_PROJ. ***"
  echo "*** Current commit: $(git -C "$VLLM_XPU_KERNELS_PROJ" rev-parse HEAD). ***"
  exit 0
fi

vllm_xpu_kernels_wheel_exists=false
if [[ "$latest" == true ]]; then
  echo "*** --latest specified: skipping wheel check.  ***"
elif [[ -d "$VLLM_XPU_KERNELS_PROJ/dist" ]]; then
  python_version="$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
  wheel_pattern="vllm_xpu_kernels-*+g${vllm_xpu_kernels_pinned_commit:0:7}.d*-cp${python_version}-cp${python_version}-linux_x86_64.whl"
  wheel="$(find "$VLLM_XPU_KERNELS_PROJ/dist" -maxdepth 1 -type f -name "$wheel_pattern" -printf '%f\n' 2>/dev/null | head -n 1)"

  if [[ -n "$wheel" ]]; then
    echo "*** Found vllm_xpu_kernels wheel: $wheel. ***"
    vllm_xpu_kernels_wheel_exists=true
  else
    echo "*** No matching wheel in $VLLM_XPU_KERNELS_PROJ/dist. ***"
  fi
else
  echo "*** $VLLM_XPU_KERNELS_PROJ/dist does not exist. ***"
fi

if [[ "$check_wheel" == false ]] || [[ "$vllm_xpu_kernels_wheel_exists" == false ]]; then
  prepare_source "$VLLM_XPU_KERNELS_PROJ" "https://github.com/vllm-project/vllm-xpu-kernels.git" "${vllm_xpu_kernels_pinned_commit:-}" "$latest"

  sed -i \
    -e '/"torch/d' \
    "$VLLM_XPU_KERNELS_PROJ/pyproject.toml"

  sed -i \
    -e '/^torch/d' \
    -e '/^triton/d' \
    -e '/^--extra-index-url.*https:\/\/download\.pytorch\.org\/whl/d' \
    "$VLLM_XPU_KERNELS_PROJ/requirements.txt"
  python -m pip install -r "$VLLM_XPU_KERNELS_PROJ/requirements.txt"
  VLLM_TARGET_DEVICE=xpu python -m build --wheel --no-isolation "$VLLM_XPU_KERNELS_PROJ"
fi

install_vllm
python -m pip install "$VLLM_XPU_KERNELS_PROJ"/dist/*.whl

show_installs
