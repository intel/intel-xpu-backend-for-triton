#!/usr/bin/env bash

set -euo pipefail

# Select which tests to run.
TEST_CORE=false
TEST_TUTORIAL=false
TEST_UNIT=false
VENV=false
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
    --help)
      echo "Example usage: ./test-triton.sh [--core | --tutorial | --unit | --venv]"
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
export SCRIPTS_DIR=$(dirname "$0")

python3 -m pip install lit
python3 -m pip install pytest pytest-xdist pytest-rerunfailures

$SCRIPTS_DIR/compile-pytorch-ipex.sh $ARGS
if [ $? -ne 0 ]; then
  echo "FAILED: return code $?"
  exit $?
fi

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
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  echo "***************************************************"
  echo "******       Running Triton LIT tests        ******"
  echo "***************************************************"
  LIT_TEST_DIR=$TRITON_PROJ_BUILD/"$(ls $TRITON_PROJ_BUILD)"/test
  if [ ! -d "${LIT_TEST_DIR}" ]; then
    echo "Not found '${LIT_TEST_DIR}'. Build Triton please" ; exit 4
  fi
  lit -v "${LIT_TEST_DIR}"
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi
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

  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest -n 8 --verbose --device xpu language/ --ignore=language/test_line_info.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest --verbose --device xpu runtime/ --ignore=runtime/test_jit.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  # run test_line_info.py separately with TRITON_DISABLE_LINE_INFO=0
  TRITON_DISABLE_LINE_INFO=0 python3 -m pytest --verbose --device xpu language/test_line_info.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  TRITON_INTERPRET=1 TRITON_DISABLE_LINE_INFO=1 python3 -m pytest -vvv -n 4 -m interpreter language/test_core.py --device cpu
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  TRITON_INTERPRET=1 TRITON_DISABLE_LINE_INFO=1 python3 -m pytest -n 8 -m interpreter -vvv -s operators/test_flash_attention.py::test_op --device cpu
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest -n 8 --verbose --device xpu operators/
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi
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

  python3 -m pytest -vvv -s --device xpu . --reruns 10 --ignore=test_performance.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi
}

run_tutorial_test() {
  echo
  echo "****** Running $1 test ******"
  echo
  python $2
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

}

run_tutorial_tests() {
  echo "***************************************************"
  echo "**** Running Triton Tutorial tests           ******"
  echo "***************************************************"
  python3 -m pip install matplotlib pandas tabulate -q
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  TUTORIAL_TEST_DIR=$TRITON_PROJ/python/tutorials
  if [ ! -d "${TUTORIAL_TEST_DIR}" ]; then
    echo "Not found '${TUTORIAL_TEST_DIR}'." ; exit 5
  fi
  cd $TUTORIAL_TEST_DIR

  run_tutorial_test "01-vector-add" 01-vector-add.py
  run_tutorial_test "02-fused-softmax" 02-fused-softmax.py
  run_tutorial_test "03-matrix-multiplication" 03-matrix-multiplication.py
  run_tutorial_test "04-low-memory-dropout" 04-low-memory-dropout.py
  run_tutorial_test "05-layer-norm" 05-layer-norm.py
  run_tutorial_test "06-fused-attention.py" 06-fused-attention.py
  run_tutorial_test "07-extern-functions" 07-extern-functions.py
  run_tutorial_test "08-experimental-block-pointer" 08-experimental-block-pointer.py
  run_tutorial_test "09-experimental-tma-matrix-multiplication" 09-experimental-tma-matrix-multiplication.py
  run_tutorial_test "10-experimental-tma-store-matrix-multiplication" 10-experimental-tma-store-matrix-multiplication.py
  run_tutorial_test "11-grouped-gemm" 11-grouped-gemm.py
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
