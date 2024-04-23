#!/usr/bin/env bash

set -euo pipefail

# Select which tests to run.
VENV=false
ARGS=
for arg in "$@"; do
  case $arg in
    --venv)
      VENV=true
      shift
      ;;
    --help)
      echo "Example usage: ./docs-triton.sh [ --venv ]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

if [ ! -v BASE ]; then
  echo "**** BASE is not given *****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

if [ "$VENV" = true ]; then
  source .venv/bin/activate
fi

export TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
export TRITON_PROJ_BUILD=$TRITON_PROJ/python/build
export SCRIPTS_DIR=$(cd $(dirname "$0") && pwd)

python3 -m pip install lit
python3 -m pip install pytest pytest-xdist pytest-rerunfailures pytest-select pytest-select

source $SCRIPTS_DIR/pytest-utils.sh
$SCRIPTS_DIR/compile-pytorch-ipex.sh --pinned $ARGS

if [ ! -d "$TRITON_PROJ_BUILD" ]
then
  echo "****** ERROR: Build Triton first ******"
  exit 1
fi


run_build_docs() {
  echo "***************************************************"
  echo "************   Building Triton Docs    ************"
  echo "***************************************************"
  python3 -m pip install matplotlib pandas tabulate sphinx sphinx_rtd_theme sphinx_gallery sphinx_multiversion myst_parser -q
  cd $TRITON_PROJ/docs
  python3 -m sphinx . _build/html/mai
}

run_build_docs
