#!/usr/bin/env bash

# Select which tests to run.
TEST_CORE=false
TEST_TUTORIAL=false
TEST_UNIT=false
VENV=false
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

set +o xtrace
if [ ! -d "$BASE" ]; then
  echo "**** BASE is not given *****"
  echo "**** Default BASE is set to /iusers/$USER ****"
  BASE=/iusers/$USER
fi

if [ "$VENV" = true ]; then
  source .venv/bin/activate
fi

export TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
export TRITON_PROJ_BUILD=$TRITON_PROJ/python/build

python3 -m pip install lit
python3 -m pip install pytest pytest-xdist
python3 -m pip install torch==2.1.0a0+cxx11.abi intel_extension_for_pytorch==2.1.10+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
if [ $? -ne 0 ]; then
  echo "FAILED: return code $?"
  exit $?
fi

if [ ! -d "$TRITON_PROJ_BUILD" ]
then
  echo "****** ERROR: Build Triton first ******"
  exit 1
fi

function run_unit_tests {
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

function run_core_tests {
  echo "***************************************************"
  echo "******      Running Triton Core tests        ******"
  echo "***************************************************"
  CORE_TEST_DIR=$TRITON_PROJ/python/test/unit
  if [ ! -d "${CORE_TEST_DIR}" ]; then
    echo "Not found '${CORE_TEST_DIR}'. Build Triton please" ; exit 3
  fi
  cd ${CORE_TEST_DIR}

  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest -n 8 --verbose --device xpu language/ --ignore=language/test_line_info.py --ignore=language/test_subprocess.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  # run test_subprocess serially to avoid issues with print buffer handling.
  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest --verbose --device xpu language/test_subprocess.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest --verbose runtime/
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  TRITON_DISABLE_LINE_INFO=1 python3 -m pytest -n 8 --verbose operators/
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

  # run test_line_info.py separately with TRITON_DISABLE_LINE_INFO=0
  TRITON_DISABLE_LINE_INFO=0 python3 -m pytest --verbose language/test_line_info.py
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi
}

function run_tutorial_test {
  echo
  echo "****** Running $1 test ******"
  echo
  python $2
  if [ $? -ne 0 ]; then
    echo "FAILED: return code $?" ; exit $?
  fi

}

function run_tutorial_tests {
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
  run_tutorial_test "04-low-memory-dropout" 04-low-memory-dropout.py
  run_tutorial_test "05-layer-norm" 05-layer-norm.py
  run_tutorial_test "06-fused-attention.py" 06-fused-attention.py
  run_tutorial_test "07-math-functions" 07-math-functions.py
}

function test_triton {
  if [ "$TEST_UNIT" = true ]; then
    run_unit_tests
  fi
  if [ "$TEST_CORE" = true ]; then
    run_core_tests
  fi
  if [ "$TEST_TUTORIAL" = true ]; then
    run_tutorial_tests
  fi
}

test_triton
