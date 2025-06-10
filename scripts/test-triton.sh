#!/usr/bin/env bash

set -euo pipefail

HELP="\
Example usage: ./test-triton.sh [TEST]... [OPTION]...

TEST:
    --unit          default
    --core          default
    --tutorial      default
    --microbench    default
    --minicore      part of core
    --mxfp          part of core
    --scaled-dot    part of core
    --interpreter
    --benchmarks
    --softmax
    --gemm
    --attention
    --instrumentation
    --inductor

OPTION:
    --unskip
    --venv
    --skip-pip-install
    --skip-pytorch-install
    --reports
    --reports-dir DIR
    --warning-reports
    --ignore-errors
    --skip-list SKIPLIST
    --select-from-file SELECTFILE
"

err() {
    echo "$@"
    exit 1
}

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Select which tests to run.
TEST_DEFAULT=true
TEST_UNIT=false
TEST_CORE=false
TEST_MINICORE=false
TEST_MXFP=false
TEST_SCALED_DOT=false
TEST_INTERPRETER=false
TEST_TUTORIAL=false
TEST_MICRO_BENCHMARKS=false
TEST_BENCHMARKS=false
TEST_BENCHMARK_SOFTMAX=false
TEST_BENCHMARK_GEMM=false
TEST_BENCHMARK_ATTENTION=false
TEST_INSTRUMENTATION=false
TEST_INDUCTOR=false
VENV=false
TRITON_TEST_REPORTS=false
TRITON_TEST_WARNING_REPORTS=false
TRITON_TEST_IGNORE_ERRORS=false
SKIP_PIP=false
SKIP_PYTORCH=false
TEST_UNSKIP=false

while (( $# != 0 )); do
  case "$1" in
    --unskip)
      TEST_UNSKIP=true
      shift
      ;;
    --unit)
      TEST_UNIT=true
      TEST_DEFAULT=false
      shift
      ;;
    --core)
      TEST_CORE=true
      TEST_DEFAULT=false
      shift
      ;;
    --minicore)
      TEST_MINICORE=true
      TEST_DEFAULT=false
      shift
      ;;
    --mxfp)
      TEST_MXFP=true
      TEST_DEFAULT=false
      shift
      ;;
    --scaled-dot)
      TEST_SCALED_DOT=true
      TEST_DEFAULT=false
      shift
      ;;
    --interpreter)
      TEST_INTERPRETER=true
      TEST_DEFAULT=false
      shift
      ;;
    --tutorial)
      TEST_TUTORIAL=true
      TEST_DEFAULT=false
      shift
      ;;
    --microbench)
      TEST_MICRO_BENCHMARKS=true
      TEST_DEFAULT=false
      shift
      ;;
    --benchmarks)
      TEST_BENCHMARKS=true
      TEST_DEFAULT=false
      shift
      ;;
    --softmax)
      TEST_BENCHMARK_SOFTMAX=true
      TEST_DEFAULT=false
      shift
      ;;
    --gemm)
      TEST_BENCHMARK_GEMM=true
      TEST_DEFAULT=false
      shift
      ;;
    --attention)
      TEST_BENCHMARK_ATTENTION=true
      TEST_DEFAULT=false
      shift
      ;;
    --instrumentation)
      TEST_INSTRUMENTATION=true
      TEST_DEFAULT=false
      shift
      ;;
    --inductor)
      TEST_INDUCTOR=true
      TEST_DEFAULT=false
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
      # Must be absolute
      TRITON_TEST_REPORTS_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
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
      # Must be absolute
      TRITON_TEST_SKIPLIST_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
      shift 2
      ;;
    --select-from-file)
      # Must be absolute
      TRITON_TEST_SELECTFILE="$(realpath "$2")"
      shift 2
      ;;
    --help)
      echo "$HELP"
      exit 0
      ;;
    *)
      err "Unknown argument: $1."
      ;;
  esac
done

if [ "$TEST_DEFAULT" = true ]; then
  TEST_UNIT=true
  TEST_CORE=true
  TEST_TUTORIAL=true
  TEST_MICRO_BENCHMARKS=true
fi

if [ "$VENV" = true ]; then
  if [[ $OSTYPE = msys ]]; then
    source .venv/Scripts/activate
  else
    source .venv/bin/activate
  fi
fi

TRITON_PROJ="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )"
SCRIPTS_DIR="$TRITON_PROJ/scripts"
source "$SCRIPTS_DIR/pytest-utils.sh"

if [ "$TRITON_TEST_REPORTS" == true ]; then
    capture_runtime_env
fi

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
    $SCRIPTS_DIR/install-pytorch.sh $([ $VENV = true ] && echo "--venv")
  fi
}

run_unit_tests() {
  echo "***************************************************"
  echo "******      Running Triton CXX unittests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/build/cmake* || err "****** ERROR: Build Triton first ******"
  ctest .

  echo "***************************************************"
  echo "******       Running Triton LIT tests        ******"
  echo "***************************************************"
  cd $TRITON_PROJ/build/cmake*/test
  lit -v . || $TRITON_TEST_IGNORE_ERRORS
}

run_pytest_command() {
  if [[ -n "$TRITON_TEST_SELECTFILE" ]]; then
    if pytest "$@" --collect-only > /dev/null 2>&1; then
      pytest "$@"
    fi
  else
    pytest "$@"
  fi
}

run_regression_tests() {
  echo "***************************************************"
  echo "******   Running Triton Regression tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/regression

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=regression \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} -s --device xpu . --ignore=test_performance.py
}

run_minicore_tests() {
  echo "***************************************************"
  echo "******    Running Triton mini core tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit
  ensure_spirv_dis

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=language \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu language/ --ignore=language/test_line_info.py --ignore=language/test_subprocess.py --ignore=language/test_warp_specialization.py \
    -k "not test_mxfp and not test_scaled_dot"

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu language/test_subprocess.py

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=runtime \
    run_pytest_command -k "not test_within_2gb" --verbose --device xpu runtime/ --ignore=runtime/test_cublas.py

  TRITON_TEST_SUITE=debug \
    run_pytest_command --verbose -n ${PYTEST_MAX_PROCESSES:-8} test_debug.py test_debug_dump.py --forked --device xpu

  TRITON_TEST_SUITE=warnings \
    run_pytest_command --verbose -n ${PYTEST_MAX_PROCESSES:-8} test_perf_warning.py --device xpu

  # run test_line_info.py separately with TRITON_DISABLE_LINE_INFO=0
  TRITON_DISABLE_LINE_INFO=0 TRITON_TEST_SUITE=line_info \
    run_pytest_command -k "not test_line_info_interpreter" --verbose --device xpu language/test_line_info.py

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=tools \
    run_pytest_command -n ${PYTEST_MAX_PROCESSES:-8} -k "not test_disam_cubin" --verbose tools

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=intel \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu intel/ --ignore=intel/test_mxfp_matmul.py

  cd $TRITON_PROJ/third_party/intel/python/test
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=third_party \
    run_pytest_command --device xpu .

  run_regression_tests
}

run_mxfp_tests() {
  echo "***************************************************"
  echo "******    Running Triton matmul mxfp tests   ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=mxfp \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu intel/test_mxfp_matmul.py
}

run_scaled_dot_tests() {
  echo "***************************************************"
  echo "******    Running Triton scaled_dot tests    ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=scaled_dot \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu language/ --ignore=language/test_line_info.py --ignore=language/test_subprocess.py --ignore=language/test_warp_specialization.py --ignore=language/test_frontend.py\
    -k "test_scaled_dot"
}

run_core_tests() {
  echo "***************************************************"
  echo "******      Running Triton Core tests        ******"
  echo "***************************************************"
  run_minicore_tests
  run_mxfp_tests
  run_scaled_dot_tests
}

run_interpreter_tests() {
  echo "***************************************************"
  echo "******   Running Triton Interpreter tests    ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_INTERPRET=1 TRITON_TEST_SUITE=interpreter \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-16} -m interpreter language/test_core.py language/test_standard.py \
    language/test_random.py language/test_line_info.py --device cpu
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
  run_tutorial_test "09-persistent-matmul"
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
  pip install .
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/fused_softmax.py
}

run_benchmark_gemm() {
  echo "****************************************************"
  echo "*****              Running GEMM                *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  pip install .

  echo "Default path:"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/gemm_benchmark.py

  echo "GEMM with tensor of pointer:"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/gemm_tensor_of_ptr_benchmark.py

  echo "GEMM with tensor descriptor:"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/gemm_tensor_desc_benchmark.py
}

run_benchmark_attention() {
  echo "****************************************************"
  echo "*****            Running ATTENTION             *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  pip install .

  echo "Forward - Default path:"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py

  echo "Forward - with tensor descriptor:"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_tensor_desc_benchmark.py

  echo "Forward - Advanced path:"
  TRITON_INTEL_ADVANCED_PATH=1 \
    IGC_VISAOptions=" -enableBCR" \
    python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py

  echo "Backward - Default path:"
  FA_KERNEL_MODE="bwd" \
    python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py
}

run_benchmarks() {
  cd $TRITON_PROJ/benchmarks
  pip install .
  for file in $TRITON_PROJ/benchmarks/triton_kernels_benchmark/*.py; do
    benchmark=$(basename -- "$file" .py)
    if [[ $benchmark = @("__init__"|"benchmark_shapes_parser"|"benchmark_testing"|"benchmark_utils"|"build_report") ]]; then
      continue
    fi
    echo
    echo "****** Running ${benchmark} ******"
    echo
    python $file
  done
}

run_instrumentation_tests() {
  INSTRUMENTATION_LIB_DIR=$(ls -1d $TRITON_PROJ/build/*lib*/triton/instrumentation) || err "Could not find $TRITON_PROJ/build/*lib*/triton/instrumentation, build Triton first"
  INSTRUMENTATION_LIB_NAME=$(ls -1 $INSTRUMENTATION_LIB_DIR/*GPUInstrumentationTestLib* | head -n1)

  cd $TRITON_PROJ/python/test/unit

  # FIXME: `-n 1` is not required, but a workaround for pytest-skip, which does report a false positive skip list item not matching to any test.
  TRITON_TEST_SUITE=instrumentation \
    TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_LINE_INFO=0 LLVM_PASS_PLUGIN_PATH=${INSTRUMENTATION_LIB_NAME} \
    run_pytest_command -vvv -n 1 --device xpu instrumentation/test_gpuhello.py
}

run_inductor_tests() {
  test -d pytorch || (
    git clone https://github.com/pytorch/pytorch
    rev=$(cat .github/pins/pytorch.txt)
    cd pytorch
    git checkout $rev
  )

  pip install pyyaml pandas scipy numpy psutil pyre_extensions torchrec

  # TODO: Find the fastest Hugging Face model
  ZE_AFFINITY_MASK=0 python pytorch/benchmarks/dynamo/huggingface.py --accuracy --float32 -dxpu -n10 --no-skip --dashboard --inference --freezing --total-partitions 1 --partition-id 0 --only AlbertForMaskedLM --backend=inductor --timeout=4800 --output=$(pwd -P)/inductor_log.csv

  cat inductor_log.csv
  grep AlbertForMaskedLM inductor_log.csv | grep -q ,pass,
}

test_triton() {
  if [ "$TEST_UNIT" = true ]; then
    run_unit_tests
  fi

  # core suite consists of minicore, mxfp, scaled_dot
  if [ "$TEST_CORE" = true ]; then
    run_core_tests
  else
    if [ "$TEST_MINICORE" = true ]; then
        run_minicore_tests
    fi
    if [ "$TEST_MXFP" = true ]; then
        run_mxfp_tests
    fi
    if [ "$TEST_SCALED_DOT" = true ]; then
        run_scaled_dot_tests
    fi
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
  if [ "$TEST_BENCHMARKS" = true ]; then
    run_benchmarks
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
  if [ "$TEST_INDUCTOR" == true ]; then
    run_inductor_tests
  fi
}

install_deps
test_triton
