#!/usr/bin/env bash

set -euo pipefail

HELP="\
Example usage: ./test-triton.sh [TEST]... [OPTION]...

TEST:
    --unit            default
    --core            default
    --tutorial        default
    --microbench      default
    --triton-kernels  default
    --minicore        part of core
    --mxfp            part of core
    --scaled-dot      part of core
    --gluon
    --interpreter
    --benchmarks
    --softmax
    --gemm
    --flash-attention
    - tutorial-fa-64
    - tutorial-fa-128-fwdfp8
    - tutorial-fa-128-nofwdfp8
    --flex-attention
    --instrumentation
    --inductor
    --sglang
    --liger
    --vllm

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
    --extra-skip-list-suffixes SEMICOLON-SEPARATED LIST OF SUFFIXES
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
TEST_GLUON=false
TEST_INTERPRETER=false
TEST_TUTORIAL=false
TEST_MICRO_BENCHMARKS=false
TEST_BENCHMARKS=false
TEST_BENCHMARK_SOFTMAX=false
TEST_BENCHMARK_GEMM=false
TEST_BENCHMARK_FLASH_ATTENTION=false
TEST_BENCHMARK_FLEX_ATTENTION=false
TEST_INSTRUMENTATION=false
TEST_INDUCTOR=false
TEST_SGLANG=false
TEST_LIGER=false
TEST_VLLM=false
TEST_TRITON_KERNELS=false
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
    --gluon)
      TEST_GLUON=true
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
    --tutorial-fa-64)
      TEST_TUTORIAL=true
      TEST_TUTORIAL_FA=true
      FA_CONFIG="HEAD_DIM=64"
      TEST_DEFAULT=false
      shift
      ;;
    --tutorial-fa-128-fwdfp8)
      TEST_TUTORIAL=true
      TEST_TUTORIAL_FA=true
      FA_CONFIG="HEAD_DIM=128 FWD_FP8_ONLY=1"
      TEST_DEFAULT=false
      shift
      ;;
    --tutorial-fa-128-nofwdfp8)
      TEST_TUTORIAL=true
      TEST_TUTORIAL_FA=true
      FA_CONFIG="HEAD_DIM=128 FWD_FP8_SKIP=1"
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
    --flash-attention)
      TEST_BENCHMARK_FLASH_ATTENTION=true
      TEST_DEFAULT=false
      shift
      ;;
    --flex-attention)
      TEST_BENCHMARK_FLEX_ATTENTION=true
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
    --sglang)
      TEST_SGLANG=true
      TEST_DEFAULT=false
      shift
      ;;
    --liger)
      TEST_LIGER=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm)
      TEST_VLLM=true
      TEST_DEFAULT=false
      shift
      ;;
    --triton-kernels)
      TEST_TRITON_KERNELS=true
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
      export TRITON_TEST_REPORTS_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
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
    --extra-skip-list-suffixes)
      TRITON_EXTRA_SKIPLIST_SUFFIXES="$2"
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
  TEST_TRITON_KERNELS=true
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
    cat /crisim/gfx-driver/version.txt
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
    -k "not test_mxfp and not test_preshuffle_scale_mxfp_cdna4 and not test_scaled_dot"

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu language/test_subprocess.py

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=runtime \
    run_pytest_command -k "not test_within_2gb" --verbose --device xpu runtime/ --ignore=runtime/test_cublas.py

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

run_gluon_tests() {
  echo "***************************************************"
  echo "******         Running Gluon tests          ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/gluon

  TRITON_TEST_SUITE=gluon \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu .
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

  tutorials=(
    "01-vector-add"
    "02-fused-softmax"
    "03-matrix-multiplication"
    "04-low-memory-dropout"
    "05-layer-norm"
    "06-fused-attention"
    "07-extern-functions"
    "08-grouped-gemm"
    "09-persistent-matmul"
    "10-experimental-block-pointer"
    "10i-experimental-block-pointer"
  )
  if [ "${TEST_TUTORIAL_FA:-false}" = true ]; then
    tutorials=(
      "06-fused-attention"
    )

    if [ -n "${FA_CONFIG:-}" ]; then
      # Containst specific config for Fused attention tutorial
      export $FA_CONFIG
    fi
  fi

  for tutorial in "${tutorials[@]}"; do
    if [[ -f $TRITON_TEST_SELECTFILE ]] && ! grep -qF "$tutorial" "$TRITON_TEST_SELECTFILE"; then
        continue
    fi

    for i in $(seq 0 ${PYTEST_RERUNS:-0}); do
      run_tutorial_test "$tutorial" && break || [[ $i -lt ${PYTEST_RERUNS:-0} ]]
      echo "Rerunning $tutorial"
    done
  done
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

run_benchmark_flash_attention() {
  echo "****************************************************"
  echo "*****          Running FlashAttention          *****"
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

run_benchmark_flex_attention() {
  echo "****************************************************"
  echo "*****          Running FlexAttention           *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  pip install .

  echo "FlexAttention - causal mask:"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flex_attention_benchmark_causal_mask.py
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

  pip install pyyaml pandas scipy 'numpy==1.26.4' psutil pyre_extensions torchrec

  # TODO: Find the fastest Hugging Face model
  ZE_AFFINITY_MASK=0 python pytorch/benchmarks/dynamo/huggingface.py --accuracy --float32 -dxpu -n10 --no-skip --dashboard --inference --freezing --total-partitions 1 --partition-id 0 --only AlbertForMaskedLM --backend=inductor --timeout=4800 --output=$(pwd -P)/inductor_log.csv

  cat inductor_log.csv
  grep AlbertForMaskedLM inductor_log.csv | grep -q ,pass,
}

run_sglang_tests() {
  echo "***************************************************"
  echo "******    Running SGLang Triton tests        ******"
  echo "***************************************************"

  if ! [ -d "./sglang" ]; then
    git clone https://github.com/sgl-project/sglang.git
  fi
  cd sglang

  if ! pip list | grep "sglang" ; then
    git apply $TRITON_PROJ/benchmarks/third_party/sglang/sglang-fix.patch
    pip install "./python[dev_xpu]"

    # SGLang installation breaks the default PyTorch and Triton versions, so we need to reinstall them.
    $SCRIPTS_DIR/install-pytorch.sh --force-reinstall
    $SCRIPTS_DIR/compile-triton.sh --triton
  fi

  pip install pytest pytest-xdist
  run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-4} test/srt/test_triton_attention_kernels.py
}

run_liger_tests() {
  echo "************************************************"
  echo "******    Running Liger Triton tests      ******"
  echo "************************************************"

  if ! [ -d "./Liger-Kernel" ]; then
    git clone https://github.com/linkedin/Liger-Kernel
  fi

  if ! pip list | grep "liger_kernel" ; then
    pip install pytest pytest-xdist pytest-cov transformers pandas pytest datasets -e Liger-Kernel
  fi

  run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-4} Liger-Kernel/test/
}

run_vllm_tests() {
  echo "************************************************"
  echo "******    Running VLLM Triton tests       ******"
  echo "************************************************"

  if ! [ -d "./vllm" ]; then
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout "$(<../benchmarks/third_party/vllm/vllm-pin.txt)"
    git apply $TRITON_PROJ/benchmarks/third_party/vllm/vllm-fix.patch
    cd ..
  fi

  if ! pip list | grep "vllm" ; then
    pip install -r vllm/requirements/xpu.txt

    git clone https://github.com/vllm-project/vllm-xpu-kernels
    cd vllm-xpu-kernels
    git checkout "$(<../benchmarks/third_party/vllm/vllm-kernels-pin.txt)"
    sed -i '/pytorch\|torch/d' requirements.txt
    pip install -r requirements.txt
    VLLM_TARGET_DEVICE=xpu pip install -e .
    cd ..

    VLLM_TARGET_DEVICE=xpu pip install --no-deps vllm
  fi

  cd vllm
  pip install pytest pytest-cov pytest-xdist cachetools cbor2 blake3 pybase64 openai_harmony tblib

  run_pytest_command -vvv tests/kernels/moe/test_batched_moe.py tests/kernels/attention/test_triton_unified_attention.py
}

run_triton_kernels_tests() {
  echo "***************************************************"
  echo "******    Running Triton Kernels tests      ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/triton_kernels/tests

  # available after `capture_runtime_env` call
  gpu_file="$TRITON_TEST_REPORTS_DIR/gpu.txt"
  if [[ -f "$gpu_file" ]] && grep -q "B580" "$gpu_file"; then
    # Using any other number of processes results in an error on the BMG due to insufficient resources.
    # FIXME: reconsider in the future
    max_procs=1
  else
    max_procs=${PYTEST_MAX_PROCESSES:-4}
  fi

  TRITON_TEST_SUITE=triton_kernels \
    run_pytest_command -vvv -n $max_procs --device xpu .
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

  if [ "$TEST_GLUON" == true ]; then
    run_gluon_tests
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
  if [ "$TEST_BENCHMARK_FLASH_ATTENTION" = true ]; then
    run_benchmark_flash_attention
  fi
  if [ "$TEST_BENCHMARK_FLEX_ATTENTION" = true ]; then
    run_benchmark_flex_attention
  fi
  if [ "$TEST_INSTRUMENTATION" == true ]; then
    run_instrumentation_tests
  fi
  if [ "$TEST_INDUCTOR" == true ]; then
    run_inductor_tests
  fi
  if [ "$TEST_SGLANG" == true ]; then
    run_sglang_tests
  fi
  if [ "$TEST_LIGER" == true ]; then
    run_liger_tests
  fi
  if [ "$TEST_VLLM" == true ]; then
    run_vllm_tests
  fi
  if [ "$TEST_TRITON_KERNELS" == true ]; then
    run_triton_kernels_tests
  fi
}

install_deps
test_triton
