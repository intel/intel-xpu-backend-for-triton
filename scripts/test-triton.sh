#!/usr/bin/env bash

set -euo pipefail

err() {
    echo $@
    exit 1
}

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
while [ -v $1 ]; do
  case "$1" in
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
      ARGS+="$1 "
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

export SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPTS_DIR/pytest-utils.sh"
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
  cd $TRITON_PROJ_BUILD/bdist*
  ctest .

  echo "***************************************************"
  echo "******       Running Triton LIT tests        ******"
  echo "***************************************************"
  cd $TRITON_PROJ_BUILD/cmake*/test
  lit -v .
}

run_core_tests() {
  echo "***************************************************"
  echo "******      Running Triton Core tests        ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit
  ensure_spirv_dis

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
  cd $TRITON_PROJ/python/test/regression

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=regression \
    pytest -vvv -s --device xpu . --reruns 10 --ignore=test_performance.py
}

run_interpreter_tests() {
  echo "***************************************************"
  echo "******   Running Triton Interpreter tests    ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_INTERPRET=1 TRITON_TEST_SUITE=interpreter \
    pytest -vvv -n 16 -m interpreter language/test_core.py language/test_standard.py \
    language/test_random.py --device cpu
}

run_tutorial_tests() {
  echo "***************************************************"
  echo "**** Running Triton Tutorial tests           ******"
  echo "***************************************************"
  python -m pip install matplotlib pandas tabulate -q
  cd $TRITON_PROJ/python/tutorials

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
  python $TRITON_PROJ/benchmarks/micro_benchmarks/run_benchmarks.py
}

run_benchmark_softmax() {
  echo "****************************************************"
  echo "*****             Running Softmax              *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  python setup.py install
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/fused_softmax.py
}

run_benchmark_gemm() {
  echo "****************************************************"
  echo "*****              Running GEMM                *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  python setup.py install

  echo "Default path:"
  TRITON_INTEL_ADVANCED_PATH=0 \
    TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
    IGC_VISAOptions=" -enableBCR -nolocalra" \
    IGC_DisableLoopUnroll=1 \
    python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/gemm_benchmark.py

  echo "Advanced path:"
  TRITON_INTEL_ADVANCED_PATH=1 \
    TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
    IGC_VISAOptions=" -enableBCR -nolocalra" \
    IGC_DisableLoopUnroll=1 \
    python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/gemm_benchmark.py
}

run_benchmark_attention() {
  echo "****************************************************"
  echo "*****            Running ATTENTION             *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  python setup.py install

  echo "Default path:"
  TRITON_INTEL_ADVANCED_PATH=0 \
    TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
    IGC_VISAOptions=" -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
    IGC_DisableLoopUnroll=1 \
    python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_fwd_benchmark.py

  echo "Advanced path:"
  TRITON_INTEL_ADVANCED_PATH=1 \
    TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 \
    TRITON_INTEL_ENABLE_INSTR_SCHED=1 \
    IGC_VISAOptions=" -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
    IGC_DisableLoopUnroll=1 \
    python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_fwd_benchmark.py
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
