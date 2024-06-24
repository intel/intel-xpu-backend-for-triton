#!/usr/bin/env bash

set -euo pipefail

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Select which tests to run.
TEST_MICRO_BENCHMARKS=false
TEST_CORE=false
TEST_TUTORIAL=false
TEST_UNIT=false
VENV=false
TRITON_TEST_REPORTS=false
TRITON_TEST_WARNING_REPORTS=false
TRITON_TEST_IGNORE_ERRORS=false
SKIP_DEPS=false
ARGS=
for arg in "$@"; do
  case $arg in
    --microbench)
      TEST_MICRO_BENCHMARKS=true
      shift
      ;;
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
    --skip-deps)
      SKIP_DEPS=true
      shift
      ;;
    --reports)
      TRITON_TEST_REPORTS=true
      shift
      ;;
    --warning-reports)
      TRITON_TEST_WARNING_REPORTS=true
      shift
      ;;
    --ignore-errors)
      TRITON_TEST_IGNORE_ERRORS=true
      shift
      ;;
    --help)
      echo "Example usage: ./test-triton.sh [--core | --tutorial | --unit | --microbench | --venv | --reports | --warning-reports | --ignore-errors]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

if [ "$TEST_MICRO_BENCHMARKS" = false ] && [ "$TEST_CORE" = false ] && [ "$TEST_TUTORIAL" = false ] && [ "$TEST_UNIT" = false ]; then
  TEST_MICRO_BENCHMARKS=true
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

python3 -m pip install lit pytest pytest-xdist pytest-rerunfailures pytest-select setuptools==69.5.1

if [ "$TRITON_TEST_WARNING_REPORTS" == true ]; then
    python3 -m pip install git+https://github.com/kwasd/pytest-capturewarnings-ng@v1.2.0
fi

source $SCRIPTS_DIR/pytest-utils.sh
if [ "$TRITON_TEST_REPORTS" == true ]; then
    capture_runtime_env
fi

$SKIP_DEPS || $SCRIPTS_DIR/compile-pytorch-ipex.sh --pinned $ARGS

if [ ! -d "$TRITON_PROJ_BUILD" ]
then
  echo "****** ERROR: Build Triton first ******"
  exit 1
fi

run_benchmark_tests() {
  echo "****************************************************"
  echo "*****   Running Triton Micro Benchmark tests   *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/micro_benchmarks
  if [ ! -d "${BENCHMARK_TEST_DIR}" ]; then
    echo "Not found '${BENCHMARK_TEST_DIR}'." ; exit 5
  fi
  python ${BENCHMARK_TEST_DIR}/run_benchmarks.py
}

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
  pytest -vvv -n 8 --device xpu language/ --ignore=language/test_line_info.py --ignore=language/test_subprocess.py

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess \
  pytest -vvv -n 8 language/test_subprocess.py

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=runtime \
  pytest --verbose --device xpu runtime/

  # run test_line_info.py separately with TRITON_DISABLE_LINE_INFO=0
  TRITON_DISABLE_LINE_INFO=0 TRITON_TEST_SUITE=line_info \
  pytest --verbose --device xpu language/test_line_info.py
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
  run_tutorial_test "10-experimental-block-pointer"
  run_tutorial_test "10i-experimental-block-pointer"
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
  if [ "$TEST_MICRO_BENCHMARKS" = true ]; then
    run_benchmark_tests
  fi
}

test_triton
