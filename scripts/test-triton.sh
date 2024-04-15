#!/usr/bin/env bash

set -euo pipefail

# Select which tests to run.
TEST_CORE=false
TEST_TUTORIAL=false
TEST_UNIT=false
VENV=false
TRITON_TEST_REPORTS=false
ARGS=
for arg in "$@"; do
  case $arg in
    --core)
      TEST_CORE=true
      shift
      ;;
    --tutorial)
      TEST_TUTORIAL=true
      shift
      ;;
    --unit)
      TEST_UNIT=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --reports)
      TRITON_TEST_REPORTS=true
      shift
      ;;
    --help)
      echo "Example usage: ./test-triton.sh [--core | --tutorial | --unit | --venv | --reports]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

if [ "$TEST_CORE" = false ] && [ "$TEST_TUTORIAL" = false ] && [ "$TEST_UNIT" = false ]; then
  TEST_CORE=true
  TEST_TUTORIAL=true
  TEST_UNIT=true
fi

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

run_unit_tests() {
  echo "***************************************************"
  echo "******      Running Triton CXX unittests     ******"
  echo "***************************************************"
  UNIT_TEST_DIR=$TRITON_PROJ_BUILD/"$(ls $TRITON_PROJ_BUILD)"
  if [ ! -d "${UNIT_TEST_DIR}" ]; then
    echo "Not found '${UNIT_TEST_DIR}'. Build Triton please" ; exit 2
  fi
  cd $UNIT_TEST_DIR
  ctest .

  echo "***************************************************"
  echo "******       Running Triton LIT tests        ******"
  echo "***************************************************"
  LIT_TEST_DIR=$TRITON_PROJ_BUILD/"$(ls $TRITON_PROJ_BUILD)"/test
  if [ ! -d "${LIT_TEST_DIR}" ]; then
    echo "Not found '${LIT_TEST_DIR}'. Build Triton please" ; exit 4
  fi
  lit -v "${LIT_TEST_DIR}"
}

run_core_tests() {
  echo "***************************************************"
  echo "******      Running Triton Core tests        ******"
  echo "***************************************************"
  CORE_TEST_DIR=$TRITON_PROJ/python/test/unit
  if [ ! -d "${CORE_TEST_DIR}" ]; then
    echo "Not found '${CORE_TEST_DIR}'. Build Triton please" ; exit 3
  fi
  cd ${CORE_TEST_DIR}

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=language \
  pytest -vvv -n 8 --device xpu language/ --deselect-from-file ../../../scripts/core.exclude-list --ignore=language/test_line_info.py --ignore=language/test_subprocess.py

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess \
  pytest -vvv -n 8 language/test_subprocess.py

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=runtime \
  pytest --verbose --device xpu runtime/

  # run test_line_info.py separately with TRITON_DISABLE_LINE_INFO=0
  TRITON_DISABLE_LINE_INFO=0 TRITON_TEST_SUITE=line_info \
  pytest --verbose --device xpu language/test_line_info.py

  TRITON_INTERPRET=1 TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=interpreter \
  pytest -vvv -n 16 -m interpreter --deselect-from-file ../../../scripts/interpreter.exclude-list language/test_core.py language/test_standard.py \
  language/test_random.py operators/test_flash_attention.py::test_op --device cpu

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=operators \
  pytest -n 8 --verbose --device xpu operators/
}

run_regression_tests() {
  echo "***************************************************"
  echo "******   Running Triton Regression tests     ******"
  echo "***************************************************"
  REGRESSION_TEST_DIR=$TRITON_PROJ/python/test/regression
  if [ ! -d "${REGRESSION_TEST_DIR}" ]; then
    echo "Not found '${REGRESSION_TEST_DIR}'. Build Triton please" ; exit 3
  fi
  cd ${REGRESSION_TEST_DIR}

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=regression \
  pytest -vvv -s --device xpu . --reruns 10 --ignore=test_performance.py
}

run_tutorial_test() {
  echo
  echo "****** Running $1 test ******"
  echo
  python $1.py
}

run_tutorial_tests() {
  echo "***************************************************"
  echo "**** Running Triton Tutorial tests           ******"
  echo "***************************************************"
  python3 -m pip install matplotlib pandas tabulate -q

  TUTORIAL_TEST_DIR=$TRITON_PROJ/python/tutorials
  if [ ! -d "${TUTORIAL_TEST_DIR}" ]; then
    echo "Not found '${TUTORIAL_TEST_DIR}'." ; exit 5
  fi
  cd $TUTORIAL_TEST_DIR

  run_tutorial_test "01-vector-add"
  run_tutorial_test "02-fused-softmax"
  run_tutorial_test "03-matrix-multiplication"
  run_tutorial_test "04-low-memory-dropout"
  run_tutorial_test "05-layer-norm"
  run_tutorial_test "06-fused-attention"
  run_tutorial_test "07-extern-functions"
  run_tutorial_test "08-grouped-gemm"
  run_tutorial_test "09-experimental-block-pointer"
}

test_triton() {
  if [ "$TEST_UNIT" = true ]; then
    run_unit_tests
  fi
  if [ "$TEST_CORE" = true ]; then
    run_core_tests
    run_regression_tests
  fi
  if [ "$TEST_TUTORIAL" = true ]; then
    run_tutorial_tests
  fi
}

test_triton
