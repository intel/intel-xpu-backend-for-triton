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
    --intel           part of core
    --language        part of core
    --matmul          part of core
    --scaled-dot      part of core
    --runtime         part of core
    --debug           part of core
    --warnings        part of core
    --tools           part of core
    --regression      part of core
    --gluon
    --interpreter
    --proton
    --benchmarks
    --softmax
    --gemm
    --flash-attention
    --tutorial06-run-mode MODE   FA run mode: all, skip, fa_only, fp8_only, skip_fp8
    --flex-attention
    --instrumentation
    --inductor
    --vllm
    --vllm-spec-decode
    --vllm-mrv2
    --vllm-moe
    --vllm-triton-attn
    --vllm-gdn-attn
    --vllm-mamba
    --vllm-quant
    --vllm-linear-attn
    --vllm-deepgemm
    --vllm-kda
    --install-vllm
    --sglang
    --install-sglang
    --liger
    --install-liger

OPTION:
    --unskip
    --venv
    --skip-pip-install
    --skip-pytorch-install
    --reports
    --reports-dir DIR
    --warning-reports
    --ignore-errors
    --run-all
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
TEST_INTEL=false
TEST_LANGUAGE=false
TEST_MATMUL=false
TEST_SCALED_DOT=false
TEST_RUNTIME=false
TEST_DEBUG=false
TEST_WARNINGS=false
TEST_TOOLS=false
TEST_REGRESSION=false
TEST_GLUON=false
TEST_INTERPRETER=false
TEST_PROTON=false
TEST_TUTORIAL=false
TUTORIAL06_RUN_MODE=all
TEST_MICRO_BENCHMARKS=false
TEST_BENCHMARKS=false
TEST_BENCHMARK_SOFTMAX=false
TEST_BENCHMARK_GEMM=false
TEST_BENCHMARK_FLASH_ATTENTION=false
TEST_BENCHMARK_FLEX_ATTENTION=false
TEST_INSTRUMENTATION=false
TEST_INDUCTOR=false
TEST_SGLANG=false
INSTALL_SGLANG=false
TEST_LIGER=false
INSTALL_LIGER=false
TEST_VLLM=false
TEST_VLLM_SPEC_DECODE=false
TEST_VLLM_MRV2=false
TEST_VLLM_MOE=false
TEST_VLLM_TRITON_ATTN=false
TEST_VLLM_GDN_ATTN=false
TEST_VLLM_MAMBA=false
TEST_VLLM_QUANT=false
TEST_VLLM_LINEAR_ATTN=false
TEST_VLLM_DEEPGEMM=false
TEST_VLLM_KDA=false
INSTALL_VLLM=false
TEST_TRITON_KERNELS=false
VENV=false
TRITON_TEST_REPORTS=false
TRITON_TEST_WARNING_REPORTS=false
TRITON_TEST_IGNORE_ERRORS=false
TRITON_TEST_RUN_ALL=false
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
    --intel)
      TEST_INTEL=true
      TEST_DEFAULT=false
      shift
      ;;
    --language)
      TEST_LANGUAGE=true
      TEST_DEFAULT=false
      shift
      ;;
    --matmul)
      TEST_MATMUL=true
      TEST_DEFAULT=false
      shift
      ;;
    --scaled-dot)
      TEST_SCALED_DOT=true
      TEST_DEFAULT=false
      shift
      ;;
    --runtime)
      TEST_RUNTIME=true
      TEST_DEFAULT=false
      shift
      ;;
    --debug)
      TEST_DEBUG=true
      TEST_DEFAULT=false
      shift
      ;;
    --warnings)
      TEST_WARNINGS=true
      TEST_DEFAULT=false
      shift
      ;;
    --tools)
      TEST_TOOLS=true
      TEST_DEFAULT=false
      shift
      ;;
    --regression)
      TEST_REGRESSION=true
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
    --proton)
      TEST_PROTON=true
      TEST_DEFAULT=false
      shift
      ;;
    --tutorial)
      TEST_TUTORIAL=true
      TEST_DEFAULT=false
      shift
      ;;
    --tutorial06-run-mode)
      TEST_TUTORIAL=true
      if [ "$#" -lt 2 ] || [ -z "${2-}" ]; then
        err "--tutorial06-run-mode requires an argument: one of all, skip, fa_only, fp8_only, skip_fp8."
      fi
      case "$2" in
        all|skip|fa_only|fp8_only|skip_fp8) ;;
        *) err "Invalid value for --tutorial06-run-mode: '$2'. Expected one of: all, skip, fa_only, fp8_only, skip_fp8." ;;
      esac
      TUTORIAL06_RUN_MODE="$2"
      TEST_DEFAULT=false
      shift 2
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
    --install-sglang)
      INSTALL_SGLANG=true
      TEST_DEFAULT=false
      shift
      ;;
    --liger)
      TEST_LIGER=true
      TEST_DEFAULT=false
      shift
      ;;
    --install-liger)
      INSTALL_LIGER=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm)
      TEST_VLLM=true
      TEST_DEFAULT=false
      shift
      ;;
    --install-vllm)
      INSTALL_VLLM=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-spec-decode)
      TEST_VLLM_SPEC_DECODE=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-mrv2)
      TEST_VLLM_MRV2=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-moe)
      TEST_VLLM_MOE=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-triton-attn)
      TEST_VLLM_TRITON_ATTN=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-gdn-attn)
      TEST_VLLM_GDN_ATTN=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-mamba)
      TEST_VLLM_MAMBA=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-quant)
      TEST_VLLM_QUANT=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-linear-attn)
      TEST_VLLM_LINEAR_ATTN=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-deepgemm)
      TEST_VLLM_DEEPGEMM=true
      TEST_DEFAULT=false
      shift
      ;;
    --vllm-kda)
      TEST_VLLM_KDA=true
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
    --run-all)
      TRITON_TEST_RUN_ALL=true
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
  lit -v . || handle_test_error
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

run_intel_tests() {
  echo "***************************************************"
  echo "******   Running Triton Intel tests     ******"
  echo "***************************************************"

  cd $TRITON_PROJ/python/test/unit

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=intel \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu intel/

  cd $TRITON_PROJ/third_party/intel/python/test
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=third_party \
    run_pytest_command --device xpu .
}

run_language_test() {
  local suite_name="$1"
  shift
  local extra_args=("$@")
  echo "***************************************************"
  echo "******     Running Triton ${suite_name} tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE="$suite_name" \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu "language/test_${suite_name}.py" "${extra_args[@]}"
}

# run test_line_info.py separately with TRITON_DISABLE_LINE_INFO=0
run_line_info_tests() {
  echo "***************************************************"
  echo "******     Running Triton line_info tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit
  ensure_spirv_dis

  TRITON_DISABLE_LINE_INFO=0 TRITON_TEST_SUITE=line_info \
    run_pytest_command -k "not test_line_info_interpreter" --verbose --device xpu language/test_line_info.py
}

run_language_tests() {
  echo "***************************************************"
  echo "******     Running Triton Language tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=language \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu language/ \
    --ignore=language/test_line_info.py --ignore=language/test_matmul.py --ignore=language/test_subprocess.py --ignore=language/test_warp_specialization.py \
    -k "not test_mxfp and not test_scaled_dot"

  run_language_test subprocess
  run_line_info_tests
}

run_matmul_tests() {
  run_language_test matmul -k "not test_mxfp and not test_preshuffle_scale_mxfp_cdna4 or test_mxfp8_mxfp4_matmul"
}

run_scaled_dot_tests() {
  echo "***************************************************"
  echo "******     Running Triton scaled_dot tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=scaled_dot \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} --device xpu language/test_core.py \
    -k "test_scaled_dot"
}

run_runtime_tests() {
  echo "***************************************************"
  echo "******   Running Triton Runtime tests     ******"
  echo "***************************************************"

  cd $TRITON_PROJ/python/test/unit

  # run runtime tests serially to avoid race condition with cache handling.
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=runtime \
    run_pytest_command -k "not test_within_2gb" --verbose --device xpu runtime/ --ignore=runtime/test_cublas.py
}

run_debug_tests() {
  echo "***************************************************"
  echo "******   Running Triton Debug tests     ******"
  echo "***************************************************"

  cd $TRITON_PROJ/python/test/unit

  TRITON_TEST_SUITE=debug \
    run_pytest_command --verbose -n ${PYTEST_MAX_PROCESSES:-8} test_debug.py test_debuginfo.py test_debug_dump.py --forked --device xpu
}

run_warnings_tests() {
  echo "***************************************************"
  echo "******   Running Triton Warnings tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit

  TRITON_TEST_SUITE=warnings \
    run_pytest_command --verbose -n ${PYTEST_MAX_PROCESSES:-8} test_perf_warning.py --device xpu
}

run_tools_tests() {
  echo "***************************************************"
  echo "******    Running Triton Tools tests      ******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/test/unit
  ensure_spirv_dis

  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=tools \
    run_pytest_command -n ${PYTEST_MAX_PROCESSES:-8} -k "not test_disam_cubin" --verbose tools
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
  run_intel_tests
  run_runtime_tests
  run_warnings_tests
  run_tools_tests
  run_regression_tests
}

run_core_tests() {
  echo "***************************************************"
  echo "******      Running Triton Core tests        ******"
  echo "***************************************************"
  run_minicore_tests
  run_language_tests
  run_matmul_tests
  run_scaled_dot_tests
  run_debug_tests
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

run_proton_tests() {
  echo "***************************************************"
  echo "******      Running Triton Proton tests     ******"
  echo "***************************************************"
  cd $TRITON_PROJ/third_party/proton/test

  run_pytest_command -vvv test_api.py test_cmd.py test_lib.py test_profile.py test_viewer.py --device xpu -s
}

run_tutorial_tests() {
  echo "***************************************************"
  echo "**** Running Triton Tutorial tests           ******"
  echo "***************************************************"
  python -m pip install matplotlib 'pandas<3.0' tabulate -q

  cd $TRITON_PROJ/python/test/tutorials

  # For FA-specific runs, place the report in a subdirectory so each CI
  # matrix job's tutorials.xml has a unique path within the upload artifact.
  local saved_reports_dir="$TRITON_TEST_REPORTS_DIR"
  if [[ "$TUTORIAL06_RUN_MODE" != "all" && "$TUTORIAL06_RUN_MODE" != "skip" ]]; then
    TRITON_TEST_REPORTS_DIR="$TRITON_TEST_REPORTS_DIR/test-report-tutorials-${TUTORIAL06_RUN_MODE//_/-}"
  fi

  # For reading them via os.environ for benchmark CSV redirection.
  export TRITON_TEST_REPORTS
  export TRITON_TEST_REPORTS_DIR

  # Run tutorials serially (no -n flag): tutorials execute heavy GPU kernels with
  # autotuning, sys.argv manipulation, and global allocator changes that are not
  # safe to parallelize with pytest-xdist.
  TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=tutorials \
    run_pytest_command -vvv --device xpu test_tutorials.py --tutorial06-mode "$TUTORIAL06_RUN_MODE"

  # Restore the original reports directory.
  TRITON_TEST_REPORTS_DIR="$saved_reports_dir"
}

run_microbench_tests() {
  echo "****************************************************"
  echo "*****   Running Triton Micro Benchmark tests   *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  pip install --no-build-isolation .
  python $TRITON_PROJ/benchmarks/micro_benchmarks/run_benchmarks.py
}

run_benchmark_softmax() {
  echo "****************************************************"
  echo "*****             Running Softmax              *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  pip install --no-build-isolation .
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
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/gemm_benchmark.py
}

run_benchmark_flash_attention() {
  echo "****************************************************"
  echo "*****          Running FlashAttention          *****"
  echo "****************************************************"
  cd $TRITON_PROJ/benchmarks
  pip install .

  echo "Forward - Default path (with tensor descriptor):"
  python $TRITON_PROJ/benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py

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

  pip install pyyaml 'pandas<3.0' scipy 'numpy==1.26.4' psutil pyre_extensions torchrec

  # TODO: Find the fastest Hugging Face model
  ZE_AFFINITY_MASK=0 python pytorch/benchmarks/dynamo/huggingface.py --accuracy --float32 -dxpu -n10 --no-skip --dashboard --inference --freezing --total-partitions 1 --partition-id 0 --only AlbertForMaskedLM --backend=inductor --timeout=4800 --output=$(pwd -P)/inductor_log.csv

  cat inductor_log.csv
  grep AlbertForMaskedLM inductor_log.csv | grep -q ,pass,
}

run_test_deps_install() {
  pip install pytest pytest-cov pytest-xdist
}

run_sglang_install() {
  echo "************************************************"
  echo "******    Installing SGLang                 ****"
  echo "************************************************"

  if pip show sglang >/dev/null 2>&1; then
    echo "WARNING: sglang is already installed, skipping installation."
    echo "To get clean installation, run:"
    echo "  rm -rf ./sglang && pip uninstall -y sglang"
    return
  fi

  if [ -d "./sglang" ]; then
    echo "WARNING: ./sglang directory already exists, installing from it."
    echo "To get clean installation, run:"
    echo "  rm -rf ./sglang && pip uninstall -y sglang"
  else
    git clone https://github.com/sgl-project/sglang.git
    cd sglang
    git checkout "$(<../benchmarks/third_party/sglang/sglang-pin.txt)"
    git apply ../benchmarks/third_party/sglang/sglang-test-fix.patch
    git apply ../benchmarks/third_party/sglang/sglang-bench-fix.patch

    # That's how sglang assumes we'll pick out platform for now
    cp python/pyproject_xpu.toml python/pyproject.toml
    # We should remove all torch libraries from requirements to avoid reinstalling triton & torch
    # We remove sgl kernel due to a bug in the current environment probably due to using newer torch, we don't currently use it anyway
    # We remove timm because it depends on torchvision, which depends on torch==2.9
    sed -i '/pytorch\|torch\|sgl-kernel\|timm/d' python/pyproject.toml
    cat python/pyproject.toml
    cd ..
  fi

  pip install -e "./sglang/python"
}

run_sglang_tests() {
  echo "***************************************************"
  echo "******    Running SGLang Triton tests        ******"
  echo "***************************************************"

  run_sglang_install
  run_test_deps_install
  cd sglang
  run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-4} test/srt/test_triton_attention_kernels.py
}

run_liger_install() {
  echo "************************************************"
  echo "******    Installing Liger-Kernel         ******"
  echo "************************************************"

  if ! [ -d "./Liger-Kernel" ]; then
    git clone https://github.com/linkedin/Liger-Kernel
    cd Liger-Kernel
    echo "Liger-Kernels commit: '$(git rev-parse HEAD)'"
    git apply ../benchmarks/third_party/liger/liger-fix.patch --allow-empty
    cd ..
  fi

  if ! pip list | grep "liger_kernel" ; then
    # Liger requires transformers<5.0
    # https://github.com/linkedin/Liger-Kernel/issues/978
    pip install 'transformers<5.0' 'pandas<3.0' datasets -e Liger-Kernel
  fi
}


run_liger_tests() {
  echo "************************************************"
  echo "******    Running Liger-Kernel tests      ******"
  echo "************************************************"

  run_liger_install
  run_test_deps_install
  run_pytest_command -vvv Liger-Kernel/test/
}

run_vllm_install() {
  echo "************************************************"
  echo "******    Installing VLLM                 ******"
  echo "************************************************"
  echo "VLLM pin: $(<"$TRITON_PROJ/benchmarks/vllm/vllm-pin.txt")"

  cd "$TRITON_PROJ"

  CLEAN_MSG="To get a clean install, run: \n    rm -rf $TRITON_PROJ/vllm && pip uninstall -y vllm"

  local has_vllm_pip=false
  pip show vllm >/dev/null 2>&1 && has_vllm_pip=true

  # vllm already installed — nothing to do
  if [ "$has_vllm_pip" = true ]; then
    echo "WARNING: vllm is already installed, skipping installation."
    echo -e $CLEAN_MSG
    return
  fi

  # vllm not installed — proceed, reusing existing directory if present
  if [ -d "./vllm" ]; then
    echo "WARNING: ./vllm directory already exists, installing from it."
    echo -e $CLEAN_MSG
  else
    git clone https://github.com/vllm-project/vllm.git

    # Checkout the pinned commit, apply necessary patches and modify tests to run on xpu
    cd vllm
    git checkout "$(<../benchmarks/vllm/vllm-pin.txt)"
    git apply ../benchmarks/vllm/vllm-fix.patch
    sed -i 's/device="cuda"/device="xpu"/g' \
      tests/kernels/moe/utils.py \
      tests/kernels/attention/test_triton_unified_attention.py

    sed -i 's/set_default_device("cuda")/set_default_device("xpu")/g' \
      tests/kernels/attention/test_triton_unified_attention.py

    cd ..
  fi

  # FIXME: temporary workaround — pytest-shard (from vLLM deps) conflicts with
  # pytest-skip (from triton CI, needed for --skip-from-file). Uninstall pytest-shard
  # instead of pytest-skip so skip lists work. Long-term: resolve the conflict properly.
  echo "WARNING: Uninstalling pytest-shard to preserve pytest-skip (temporary workaround)"
  pip uninstall pytest-shard -y 2>/dev/null || true

  # These files contain specific versions of pytorch and triton, so let's remove them
  # vllm_xpu_kernels wheel URL is preserved and installed from pre-built wheel
  sed -i '/pytorch\|torch\|triton/d' vllm/requirements/xpu.txt
  sed -i '/pytorch\|torch\|triton/d' vllm/requirements/test.in
  pip install -r vllm/requirements/xpu.txt
  # Let's not install whole test requirements for now, they are very large and overwrite torch
  # pip install -r vllm/requirements/test.in
  pip install cachetools cbor2 blake3 pybase64 openai_harmony tblib
  rm -rf benchmarks/vllm/batched_moe/tests
  cp -r vllm/tests benchmarks/vllm/batched_moe/tests
  VLLM_TARGET_DEVICE=xpu pip install --no-deps --no-build-isolation -e vllm
}


run_vllm_tests() {
  echo "************************************************"
  echo "******    Running VLLM Triton tests       ******"
  echo "************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # FIXME: Make batched_moe and triton_unified_attention proper test suites.
  # run_vllm_tests should eventually run all vllm testsuites.
  run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
    tests/kernels/moe/test_batched_moe.py \
    tests/kernels/attention/test_triton_unified_attention.py
}


run_vllm_spec_decode_tests() {
  echo "********************************************************"
  echo "******  Running VLLM Spec Decode tests           *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # Include test_max_len.py fully (small file, extra tests won't hurt) to keep
  # everything in a single pytest command and avoid overwriting the junit report.
  VLLM_USE_V2_MODEL_RUNNER=1 TRITON_TEST_SUITE=vllm_spec_decode \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/v1/spec_decode/test_eagle.py \
      tests/v1/spec_decode/test_mtp.py \
      tests/v1/spec_decode/test_max_len.py \
      tests/v1/spec_decode/test_speculators_eagle3.py \
      tests/v1/spec_decode/test_synthetic_rejection_sampler_utils.py \
      tests/v1/sample/test_rejection_sampler.py
}


run_vllm_mrv2_tests() {
  echo "********************************************************"
  echo "******  Running VLLM MRv2 tests                  *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  VLLM_USE_V2_MODEL_RUNNER=1 TRITON_TEST_SUITE=vllm_mrv2 \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/v1/worker/test_gpu_model_runner.py \
      tests/v1/worker/test_gpu_input_batch.py \
      tests/v1/worker/test_gpu_model_runner_v2_eplb.py \
      tests/v1/sample/test_sampler.py \
      tests/v1/sample/test_logprobs.py
}


run_vllm_moe_tests() {
  echo "********************************************************"
  echo "******  Running VLLM MOE Triton kernel tests     *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # MOE Triton kernels: moe_mmk, expert_triton_kernel, batched_triton_kernel,
  # write_zeros_to_output, count_expert_num_tokens, fused_moe_kernel,
  # fused_moe_kernel_gpta_awq, _silu_mul_fp8_quant_deep_gemm, apply_expert_map,
  # _fwd_kernel_ep_scatter_1, _fwd_kernel_ep_scatter_2, _fwd_kernel_ep_gather
  TRITON_TEST_SUITE=vllm_moe \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/kernels/moe/test_batched_moe.py \
      tests/kernels/moe/test_count_expert_num_tokens.py \
      tests/kernels/moe/test_moe.py \
      tests/kernels/moe/test_triton_moe_no_act_mul.py \
      tests/kernels/moe/test_triton_moe_ptpc_fp8.py \
      tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py \
      tests/kernels/moe/test_batched_deepgemm.py \
      tests/kernels/moe/test_gpt_oss_triton_kernels.py
}


run_vllm_triton_attn_tests() {
  echo "********************************************************"
  echo "******  Running VLLM Triton Attention tests      *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # Triton attention kernels: merge_attn_states_kernel, _fwd_kernel_stage1,
  # _fwd_grouped_kernel_stage1, _fwd_kernel_stage2, kernel_unified_attention_2d,
  # kernel_unified_attention_3d, reduce_segments
  TRITON_TEST_SUITE=vllm_triton_attn \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/kernels/attention/test_merge_attn_states.py \
      tests/kernels/attention/test_triton_decode_attention.py \
      tests/kernels/attention/test_triton_unified_attention.py \
      tests/kernels/attention/test_triton_prefill_attention.py
}


run_vllm_gdn_attn_tests() {
  echo "********************************************************"
  echo "******  Running VLLM GDN Attention tests         *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # GDN (Gated Delta Net) attention kernels used by Qwen3-Next:
  # chunk_gated_delta_rule_fwd_kernel, chunk_fwd_kernel_o,
  # chunk_scaled_dot_kkt_fwd_kernel, chunk_local_cumsum_*_kernel,
  # fused_recurrent_gated_delta_rule_fwd_kernel, l2norm_fwd_kernel*,
  # layer_norm_fwd_kernel, solve_tril_16x16_kernel, merge_*_inverse_kernel,
  # recompute_w_u_fwd_kernel
  TRITON_TEST_SUITE=vllm_gdn_attn \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/v1/attention/test_gdn_metadata_builder.py
}


run_vllm_mamba_tests() {
  echo "********************************************************"
  echo "******  Running VLLM Mamba tests                 *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # Mamba kernels: _causal_conv1d_fwd_kernel, _causal_conv1d_update_kernel,
  # fused_gdn_gating_kernel, _selective_scan_update_kernel, softplus,
  # bmm_chunk_fwd_kernel, chunk_scan_fwd_kernel, chunk_cumsum_fwd_kernel,
  # _chunk_state_fwd_kernel, chunk_state_varlen_kernel, state_passing_fwd_kernel
  TRITON_TEST_SUITE=vllm_mamba \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/kernels/mamba/test_causal_conv1d.py \
      tests/kernels/mamba/test_mamba_ssm.py \
      tests/kernels/mamba/test_mamba_ssm_ssd.py \
      tests/kernels/mamba/test_mamba_mixer2.py
}


run_vllm_quant_tests() {
  echo "********************************************************"
  echo "******  Running VLLM Quantization Triton tests   *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # Quantization Triton kernels: scaled_mm_kernel, awq_dequantize_kernel,
  # awq_gemm_kernel, round_int8, _per_token_quant_int8,
  # _per_token_group_quant_int8, _w8a8_block_int8_matmul,
  # _per_token_group_quant_fp8, _per_token_group_quant_fp8_colmajor,
  # _w8a8_block_fp8_matmul
  TRITON_TEST_SUITE=vllm_quant \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/kernels/quantization/test_triton_scaled_mm.py \
      tests/kernels/quantization/test_awq_triton.py \
      tests/kernels/quantization/test_int8_kernel.py \
      tests/kernels/quantization/test_block_int8.py \
      tests/kernels/quantization/test_fp8_quant.py \
      tests/kernels/quantization/test_fp8_quant_group.py \
      tests/kernels/quantization/test_block_fp8.py
}


run_vllm_linear_attn_tests() {
  echo "********************************************************"
  echo "******  Running VLLM Linear Attention tests      *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # Linear attention kernels (MiniMax-Text / Lightning Attention):
  # _fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce,
  # _fwd_none_diag_kernel, linear_attn_decode_kernel
  TRITON_TEST_SUITE=vllm_linear_attn \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/kernels/attention/test_lightning_attn.py
}


run_vllm_deepgemm_tests() {
  echo "********************************************************"
  echo "******  Running VLLM DeepGemm tests              *******"
  echo "********************************************************"

  run_vllm_install
  run_test_deps_install

  cd vllm
  # DeepGemm MOE kernels: _silu_mul_fp8_quant_deep_gemm, apply_expert_map,
  # _fwd_kernel_ep_scatter_1, _fwd_kernel_ep_scatter_2, _fwd_kernel_ep_gather
  TRITON_TEST_SUITE=vllm_deepgemm \
    run_pytest_command -vvv -n ${PYTEST_MAX_PROCESSES:-8} \
      tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py \
      tests/kernels/moe/test_batched_deepgemm.py \
      tests/kernels/moe/test_deepgemm.py
}


run_vllm_kda_tests() {
  echo "********************************************************"
  echo "******  Running VLLM KDA tests                   *******"
  echo "********************************************************"

  # No dedicated kernel tests exist yet — KDA is model-level integration only.
  # This is a placeholder for when kernel-level tests are added.
  echo "WARNING: No dedicated KDA kernel tests available. Skipping."
}


run_triton_kernels_tests() {
  echo "***************************************************"
  echo "******    Running Triton Kernels tests      *******"
  echo "***************************************************"
  cd $TRITON_PROJ/python/triton_kernels/tests

  # available after `capture_runtime_env` call
  gpu_file="$TRITON_TEST_REPORTS_DIR/gpu.txt"
  # BMG, LNL, ARLs, A770
  if [[ -f "$gpu_file" ]] && grep -Eq "(B580|64a0|7d6|7d5|770)" "$gpu_file"; then
    # Using any other number of processes results in an error on small GPUs due to insufficient resources.
    # FIXME: reconsider in the future
    max_procs=1
  else
    # Using any other number of processes results in an error on the PVC due to insufficient resources.
    # FIXME: reconsider in the future
    max_procs=${PYTEST_MAX_PROCESSES:-4}
  fi
  # skipping mxfp, they are part of mxfp_tests suite
  TRITON_TEST_SUITE=triton_kernels \
    run_pytest_command -vvv -n $max_procs --device xpu . -k 'not test_mxfp'
}

test_triton() {
  if [ "$TEST_UNIT" = true ]; then
    run_unit_tests
  fi
  if [ "$TEST_CORE" = true ]; then
    run_core_tests
  fi
  if [ "$TEST_MINICORE" = true ]; then
    run_minicore_tests
  fi
  if [ "$TEST_INTEL" = true ]; then
    run_intel_tests
  fi
  if [ "$TEST_LANGUAGE" = true ]; then
    run_language_tests
  fi
  if [ "$TEST_MATMUL" = true ]; then
    run_matmul_tests
  fi
  if [ "$TEST_SCALED_DOT" = true ]; then
    run_scaled_dot_tests
  fi
  if [ "$TEST_RUNTIME" = true ]; then
    run_runtime_tests
  fi
  if [ "$TEST_DEBUG" = true ]; then
    run_debug_tests
  fi
  if [ "$TEST_WARNINGS" = true ]; then
    run_warnings_tests
  fi
  if [ "$TEST_TOOLS" = true ]; then
    run_tools_tests
  fi
  if [ "$TEST_REGRESSION" = true ]; then
    run_regression_tests
  fi
  if [ "$TEST_GLUON" == true ]; then
    run_gluon_tests
  fi
  if [ "$TEST_INTERPRETER" = true ]; then
    run_interpreter_tests
  fi
  if [ "$TEST_PROTON" == true ]; then
    run_proton_tests
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
  if [ "$INSTALL_SGLANG" == true ]; then
    run_sglang_install
  fi
  if [ "$TEST_SGLANG" == true ]; then
    run_sglang_tests
  fi
  if [ "$INSTALL_LIGER" == true ]; then
    run_liger_install
  fi
  if [ "$TEST_LIGER" == true ]; then
    run_liger_tests
  fi
  if [ "$INSTALL_VLLM" == true ]; then
    run_vllm_install
  fi
  if [ "$TEST_VLLM" == true ]; then
    run_vllm_tests
  fi
  if [ "$TEST_VLLM_SPEC_DECODE" == true ]; then
    run_vllm_spec_decode_tests
  fi
  if [ "$TEST_VLLM_MRV2" == true ]; then
    run_vllm_mrv2_tests
  fi
  if [ "$TEST_VLLM_MOE" == true ]; then
    run_vllm_moe_tests
  fi
  if [ "$TEST_VLLM_TRITON_ATTN" == true ]; then
    run_vllm_triton_attn_tests
  fi
  if [ "$TEST_VLLM_GDN_ATTN" == true ]; then
    run_vllm_gdn_attn_tests
  fi
  if [ "$TEST_VLLM_MAMBA" == true ]; then
    run_vllm_mamba_tests
  fi
  if [ "$TEST_VLLM_QUANT" == true ]; then
    run_vllm_quant_tests
  fi
  if [ "$TEST_VLLM_LINEAR_ATTN" == true ]; then
    run_vllm_linear_attn_tests
  fi
  if [ "$TEST_VLLM_DEEPGEMM" == true ]; then
    run_vllm_deepgemm_tests
  fi
  if [ "$TEST_VLLM_KDA" == true ]; then
    run_vllm_kda_tests
  fi
  if [ "$TEST_TRITON_KERNELS" == true ]; then
    run_triton_kernels_tests
  fi
}

install_deps
test_triton
exit $TRITON_TEST_EXIT_CODE
