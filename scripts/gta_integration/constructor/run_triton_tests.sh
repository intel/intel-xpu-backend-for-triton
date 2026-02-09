#!/usr/bin/env bash

saved_args=("$@")

install_only=0
for a in "$@"; do
  [[ "$a" == "--install-only" ]] && install_only=1 && break
done

if (( install_only && $# != 1 )); then
  echo "Error: --install-only must be the only argument." >&2
  return 2 2>/dev/null || exit 2
fi

if [[ "$install_only" -eq 1 ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Error: to persist activation, run:" >&2
    echo "  source $0 --install-only" >&2
    exit 2
  fi
fi

# Clear all positional arguments to prevent them from being passed as an argument to the conda activation script
set --

echo "Pythonpath: $PYTHONPATH"
echo "Replace python path"
export OLD_PATH=$PYTHONPATH
echo "Env old_path: $OLD_PATH"
export PYTHONPATH=/home/gta/setup_triton/lib/python3.12/site-packages
echo "New pythonpath: $PYTHONPATH"

if [ -z "$HOME" ]; then
    echo "HOME variable is not set or empty, setting it to /root"
    export HOME="/root"
fi

installer="TritonXPU-GTA.sh"
installer_hash_file="TritonXPU-GTA.sh.sha256"
installed_hash_file="$HOME/triton-xpu/$installer_hash_file"

install_triton_env() {
  echo "Installing Triton xpu test env in $HOME/triton-xpu"
  rm -rf "$HOME/triton-xpu"
  bash "$installer"
  cp "$installer_hash_file" "$HOME/triton-xpu/"
  echo "Installed Triton xpu test env contents:"
  ls -la "$HOME/triton-xpu/"
}

echo "Checking if Triton xpu test env is already created in $HOME/triton-xpu"
if [ -d "$HOME/triton-xpu" ] && [ -f "$installed_hash_file" ]; then
  if cmp -s "$installer_hash_file" "$installed_hash_file"; then
    echo "Triton xpu test env is already created and installer hash file matches."
  else
    echo "Installer hash file mismatch. Reinstalling Triton xpu test env."
    install_triton_env
  fi
else
  install_triton_env
fi

export TRITON_PROJ="$HOME/triton-xpu/intel-xpu-backend-for-triton"
# Fix the git error fatal: detected dubious ownership in repository at '$TRITON_PROJ'
git config --global --add safe.directory "$TRITON_PROJ"

source "$HOME/triton-xpu/bin/activate"
source /opt/intel/oneapi/setvars.sh --force

if [[ "$install_only" -eq 1 ]]; then
  return 0
fi

cd $TRITON_PROJ
echo "Working directory: $(pwd)"

# This env variable might be no longer needed
export TRITON_RELAX_PROFILING_CHECK=1

rm -rf "$HOME/reports"

echo "Forwarding arguments: ${saved_args[@]}"
python scripts/test_triton.py --home-dir $HOME "${saved_args[@]}"

exit_code=$?

conda deactivate

set PYTHONPATH=$OLD_PATH
echo "Revert python path to old value $PYTHONPATH"

exit $exit_code
