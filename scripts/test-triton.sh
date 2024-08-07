#!/usr/bin/env bash

set -euo pipefail

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Select which tests to run.
TEST_UNIT=false
TEST_CORE=false
TEST_INTERPRETER=false
TEST_TUTORIAL=false
TEST_MICRO_BENCHMARKS=false
TEST_BENCHMARK_SOFTMAX=false
TEST_BENCHMARK_GEMM=false
VENV=false
TRITON_TEST_REPORTS=false
TRITON_TEST_WARNING_REPORTS=false
TRITON_TEST_IGNORE_ERRORS=false
SKIP_DEPS=false
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

# Only run interpreter test when $TEST_INTERPRETER is ture
if [ "$TEST_UNIT" = false ] && [ "$TEST_CORE" = false ] && [ "$TEST_INTERPRETER" = false ] && [ "$TEST_TUTORIAL" = false ] && [ "$TEST_MICRO_BENCHMARKS" = false ] && [ "$TEST_BENCHMARK_SOFTMAX" = false ] && [ "$TEST_BENCHMARK_GEMM" = false ]; then
  TEST_UNIT=true
  TEST_CORE=true
  TEST_TUTORIAL=true
  TEST_MICRO_BENCHMARKS=true
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

python3 -m pip install lit pytest pytest-xdist pytest-rerunfailures pytest-select pytest-timeout setuptools==69.5.1

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

  if [ ! -d "${REGRESSION_TEST_DIR}" ]; then
    echo "Not found '${REGRESSION_TEST_DIR}'. Build Triton please" ; exit 3
  fi
  cd ${REGRESSION_TEST_DIR}

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=regression \
  pytest -vvv -s --device xpu . --reruns 10 --ignore=test_performance.py
}

run_interpreter_tests() {
  echo "***************************************************"
  echo "******   Running Triton Interpreter tests    ******"
  echo "***************************************************"
  INTERPRETER_TEST_DIR=$TRITON_PROJ/python/test/unit

  if [ ! -d "${INTERPRETER_TEST_DIR}" ]; then
    echo "Not found '${INTERPRETER_TEST_DIR}'. Build Triton please" ; exit 3
  fi
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

run_microbench_tests() {
  echo "****************************************************"
  echo "*****   Running Triton Micro Benchmark tests   *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/micro_benchmarks
  if [ ! -d "${BENCHMARK_TEST_DIR}" ]; then
    echo "Not found '${BENCHMARK_TEST_DIR}'." ; exit 5
  fi
  python ${BENCHMARK_TEST_DIR}/run_benchmarks.py
}

run_benchmark_softmax() {
  echo "****************************************************"
  echo "*****             Running Softmax              *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/triton_kernels_benchmark
  if [ ! -d "${BENCHMARK_TEST_DIR}" ]; then
    echo "Not found '${BENCHMARK_TEST_DIR}'." ; exit 5
  fi
  python ${BENCHMARK_TEST_DIR}/fused_softmax.py
}

run_benchmark_gemm() {
  echo "****************************************************"
  echo "*****              Running GEMM                *****"
  echo "****************************************************"
  BENCHMARK_TEST_DIR=$TRITON_PROJ/benchmarks/triton_kernels_benchmark
  if [ ! -d "${BENCHMARK_TEST_DIR}" ]; then
    echo "Not found '${BENCHMARK_TEST_DIR}'." ; exit 5
  fi
  cd $TRITON_PROJ/benchmarks; python setup.py install
  TRITON_INTEL_ADVANCED_PATH=0 \
  TRITON_INTEL_ENABLE_FAST_PREFETCH=1 \
  TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
  IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC -abiver 2" \
  IGC_DisableLoopUnroll=1 \
  SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' " \
  python ${BENCHMARK_TEST_DIR}/gemm_benchmark.py

  TRITON_INTEL_ADVANCED_PATH=1 \
  TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
  IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC -abiver 2" \
  IGC_DisableLoopUnroll=1 \
  SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' " \
  python ${BENCHMARK_TEST_DIR}/gemm_benchmark.py
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
}

test_triton
