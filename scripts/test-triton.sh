#!/usr/bin/env bash

set -euo pipefail

export SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPTS_DIR/pytest-utils.sh"

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Select which tests to run.
TEST_UNIT=false
TEST_CORE=false
TEST_INTERPRETER=false
TEST_TUTORIAL=false
TEST_MICRO_BENCHMARKS=false
TEST_BENCHMARK_SOFTMAX=false
TEST_BENCHMARK_GEMM=false
TEST_BENCHMARK_ATTENTION=false
TEST_INSTRUMENTATION=false
VENV=false
TRITON_TEST_REPORTS=false
TRITON_TEST_WARNING_REPORTS=false
TRITON_TEST_IGNORE_ERRORS=false
SKIP_PIP=false
SKIP_PYTORCH=false
TEST_UNSKIP=false
ARGS=
for arg in "$@"; do
  case $arg in
    --unskip)
      TEST_UNSKIP=true
      shift
      ;;
    --unit)
      TEST_UNIT=true
      shift
      ;;
    --core)
      TEST_CORE=true
      shift
      ;;
    --interpreter)
      TEST_INTERPRETER=true
      shift
      ;;
    --tutorial)
      TEST_TUTORIAL=true
      shift
      ;;
    --microbench)
      TEST_MICRO_BENCHMARKS=true
      shift
      ;;
    --softmax)
      TEST_BENCHMARK_SOFTMAX=true
      shift
      ;;
    --gemm)
      TEST_BENCHMARK_GEMM=true
      shift
      ;;
    --attention)
      TEST_BENCHMARK_ATTENTION=true
      shift
      ;;
    --instrumentation)
      TEST_INSTRUMENTATION=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --skip-pip-install)
      SKIP_PIP=true
      shift
      ;;
    --skip-pytorch-install)
      SKIP_PYTORCH=true
      shift
      ;;
    --reports)
      TRITON_TEST_REPORTS=true
      shift
      ;;
    --reports-dir)
      TRITON_TEST_REPORTS=true
      TRITON_TEST_REPORTS_DIR="$2"
      shift 2
      ;;
    --warning-reports)
      TRITON_TEST_WARNING_REPORTS=true
      shift
      ;;
    --ignore-errors)
      TRITON_TEST_IGNORE_ERRORS=true
      shift
      ;;
    --skip-list)
      TRITON_TEST_SKIPLIST_DIR="$2"
      shift 2
      ;;
    --help)
      err "Example usage: ./test-triton.sh [--core | --tutorial | --unit | --microbench | --softmax | --gemm | --attention | --venv | --reports | --warning-reports | --ignore-errors]"
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

# Only run interpreter test when $TEST_INTERPRETER is true
if [ "$TEST_UNIT" = false ] && [ "$TEST_CORE" = false ] && [ "$TEST_INTERPRETER" = false ] && [ "$TEST_TUTORIAL" = false ] && [ "$TEST_MICRO_BENCHMARKS" = false ] && [ "$TEST_BENCHMARK_SOFTMAX" = false ] && [ "$TEST_BENCHMARK_GEMM" = false ] && [ "$TEST_BENCHMARK_ATTENTION" = false ] && [ "$TEST_INSTRUMENTATION" = false ]; then
  TEST_UNIT=true
  TEST_CORE=true
  TEST_TUTORIAL=true
  TEST_MICRO_BENCHMARKS=true
fi

if [ ! -v BASE ]; then
  echo "**** BASE is not given ****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

if [ "$VENV" = true ]; then
  source .venv/bin/activate
fi

export TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
export TRITON_PROJ_BUILD=$TRITON_PROJ/python/build

$SKIP_PIP || {
  python3 -m pip install lit pytest pytest-xdist pytest-rerunfailures pytest-select pytest-timeout setuptools==69.5.1 defusedxml

  $TRITON_TEST_WARNING_REPORTS || python3 -m pip install git+https://github.com/kwasd/pytest-capturewarnings-ng@v1.2.0
}

if [ "$TRITON_TEST_REPORTS" == true ]; then
    capture_runtime_env
fi

test -d "$TRITON_PROJ_BUILD" || err "****** ERROR: Build Triton first ******"

install_deps() {
  if [ "$SKIP_PIP" = true ]; then
    echo "**** Skipping installation of pip dependencies ****"
  else
    echo "**** Installing pip dependencies ****"
    python -m pip install -r "$SCRIPTS_DIR/requirements-test.txt"

    if [ "$TRITON_TEST_WARNING_REPORTS" == true ]; then
      python -m pip install git+https://github.com/kwasd/pytest-capturewarnings-ng@v1.2.0
    fi
  fi

  if [ "$SKIP_PYTORCH" = true ]; then
    echo "**** Skipping installation of pytorch ****"
  else
    echo "**** Installing pytorch ****"
    if [ "$TEST_BENCHMARK_SOFTMAX" = true ] || [ "$TEST_BENCHMARK_GEMM" = true ] || [ "$TEST_BENCHMARK_ATTENTION" = true ]; then
      $SCRIPTS_DIR/compile-pytorch-ipex.sh --pytorch --ipex --pinned --source $([ $VENV = true ] && echo "--venv")
    else
      $SCRIPTS_DIR/install-pytorch.sh $([ $VENV = true ] && echo "--venv")
    fi
  fi
}

run_unit_tests() {
  echo "***************************************************"
  echo "******      Running Triton CXX unittests     ******"
  echo "***************************************************"

  UNIT_TEST_DIR="$(ls -1d $TRITON_PROJ_BUILD/bdist*)" || err "Not found '${UNIT_TEST_DIR}'. Build Triton please"
  cd $UNIT_TEST_DIR
  ctest .

  echo "***************************************************"
  echo "******       Running Triton LIT tests        ******"
  echo "***************************************************"
  LIT_TEST_DIR=$(ls -1d $TRITON_PROJ_BUILD/cmake*/test) || err "Not found '${LIT_TEST_DIR}'. Build Triton please"
  lit -v "${LIT_TEST_DIR}"
}

run_core_tests() {
  echo "***************************************************"
  echo "******      Running Triton Core tests        ******"
  echo "***************************************************"
  CORE_TEST_DIR=$TRITON_PROJ/python/test/unit

  test -d "${CORE_TEST_DIR}" || err "Not found '${CORE_TEST_DIR}'. Build Triton please"

  cd ${CORE_TEST_DIR}
  ensure_spirv_dis
  export TEST_UNSKIP

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=language \
  pytest -vvv -n 8 --device xpu language/ --ignore=language/test_line_info.py --ignore=language/test_subprocess.py

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess \
  pytest -vvv -n 8 --device xpu language/test_subprocess.py

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
  export TEST_UNSKIP

  test -d "${REGRESSION_TEST_DIR}" || err "Not found '${REGRESSION_TEST_DIR}'. Build Triton please"
  cd ${REGRESSION_TEST_DIR}

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=regression \
  pytest -vvv -s --device xpu . --reruns 10 --ignore=test_performance.py
}

run_interpreter_tests() {
  echo "***************************************************"
  echo "******   Running Triton Interpreter tests    ******"
  echo "***************************************************"
  INTERPRETER_TEST_DIR=$TRITON_PROJ/python/test/unit

  test -d "${INTERPRETER_TEST_DIR}" || err "Not found '${INTERPRETER_TEST_DIR}'. Build Triton please"
  cd ${INTERPRETER_TEST_DIR}
  export TEST_UNSKIP
  TRITON_INTERPRET=1 TRITON_TEST_SUITE=interpreter \
    pytest -vvv -n 16 -m interpreter language/test_core.py language/test_standard.py \
  language/test_random.py --device cpu
}

run_tutorial_tests() {
  echo "***************************************************"
  echo "**** Running Triton Tutorial tests           ******"
  echo "***************************************************"
  python -m pip install matplotlib pandas tabulate -q

  TUTORIAL_TEST_DIR=$TRITON_PROJ/python/tutorials
  test -d "${TUTORIAL_TEST_DIR}" || "Not found '${TUTORIAL_TEST_DIR}'."
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

run_microbench_tests() {
  echo "****************************************************"
  echo "*****   Running Triton Micro Benchmark tests   *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/micro_benchmarks
  test -d "${BENCHMARK_TEST_DIR}" || err "Not found '${BENCHMARK_TEST_DIR}'."
  python ${BENCHMARK_TEST_DIR}/run_benchmarks.py
}

run_benchmark_softmax() {
  echo "****************************************************"
  echo "*****             Running Softmax              *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/triton_kernels_benchmark
  test -d "${BENCHMARK_TEST_DIR}" || err "Not found '${BENCHMARK_TEST_DIR}'."
  cd $TRITON_PROJ/benchmarks; python setup.py install
  python ${BENCHMARK_TEST_DIR}/fused_softmax.py
}

run_benchmark_gemm() {
  echo "****************************************************"
  echo "*****              Running GEMM                *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/triton_kernels_benchmark
  test -d "${BENCHMARK_TEST_DIR}" || err "Not found '${BENCHMARK_TEST_DIR}'."
  cd $TRITON_PROJ/benchmarks; python setup.py install
  echo "Default path:"
  TRITON_INTEL_ADVANCED_PATH=0 \
  TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
  IGC_VISAOptions=" -enableBCR -nolocalra" \
  IGC_DisableLoopUnroll=1 \
  python ${BENCHMARK_TEST_DIR}/gemm_benchmark.py

  echo "Advanced path:"
  TRITON_INTEL_ADVANCED_PATH=1 \
  TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
  IGC_VISAOptions=" -enableBCR -nolocalra" \
  IGC_DisableLoopUnroll=1 \
  python ${BENCHMARK_TEST_DIR}/gemm_benchmark.py
}

run_benchmark_attention() {
  echo "****************************************************"
  echo "*****            Running ATTENTION             *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/triton_kernels_benchmark
  test -d "${BENCHMARK_TEST_DIR}" || err "Not found '${BENCHMARK_TEST_DIR}'."
  cd $TRITON_PROJ/benchmarks; python setup.py install
  echo "Default path:"
  TRITON_INTEL_ADVANCED_PATH=0 \
  TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
  IGC_VISAOptions=" -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
  IGC_DisableLoopUnroll=1 \
  python ${BENCHMARK_TEST_DIR}/flash_attention_fwd_benchmark.py

  echo "Advanced path:"
  TRITON_INTEL_ADVANCED_PATH=1 \
  TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
  IGC_VISAOptions=" -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
  IGC_DisableLoopUnroll=1 \
  python ${BENCHMARK_TEST_DIR}/flash_attention_fwd_benchmark.py
}

run_instrumentation_tests() {
  set -x
  # FIXME: the "instrumentation" test suite currently contains only one test, when all tests
  # are skipped pytest reports an error. If the only test is the skip list, then we shouldn't
  # run pytest at all. This must be changed when there is more than one instrumentation test.
  if [[ $TEST_UNSKIP = false && -s $TRITON_TEST_SKIPLIST_DIR/instrumentation.txt ]]; then
    return
  fi

  SHARED_LIB_DIR=$(ls -1d $TRITON_PROJ/python/build/*lib*/triton/_C) || err "Could not find '${SHARED_LIB_DIR}'"

  cd $TRITON_PROJ/python/test/unit

  TRITON_TEST_SUITE=instrumentation \
  TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0 LLVM_PASS_PLUGIN_PATH=${SHARED_LIB_DIR}/libGPUHello.so \
    pytest -vvv --device xpu instrumentation/test_gpuhello.py
}

test_triton() {
  if [ "$TEST_UNIT" = true ]; then
    run_unit_tests
  fi
  if [ "$TEST_CORE" = true ]; then
    run_core_tests
    run_regression_tests
  fi
  if [ "$TEST_INTERPRETER" = true ]; then
    run_interpreter_tests
  fi
  if [ "$TEST_TUTORIAL" = true ]; then
    run_tutorial_tests
  fi
  if [ "$TEST_MICRO_BENCHMARKS" = true ]; then
    run_microbench_tests
  fi
  if [ "$TEST_BENCHMARK_SOFTMAX" = true ]; then
    run_benchmark_softmax
  fi
  if [ "$TEST_BENCHMARK_GEMM" = true ]; then
    run_benchmark_gemm
  fi
  if [ "$TEST_BENCHMARK_ATTENTION" = true ]; then
    run_benchmark_attention
  fi
  if [ "$TEST_INSTRUMENTATION" == true ]; then
    run_instrumentation_tests
  fi
}

install_deps
test_triton
