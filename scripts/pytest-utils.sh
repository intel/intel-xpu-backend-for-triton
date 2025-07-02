TIMESTAMP="$(date '+%Y%m%d%H%M%S')"

SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TRITON_TEST_REPORTS="${TRITON_TEST_REPORTS:-false}"
TRITON_TEST_REPORTS_DIR="${TRITON_TEST_REPORTS_DIR:-$HOME/reports/$TIMESTAMP}"
TRITON_TEST_SKIPLIST_DIR="${TRITON_TEST_SKIPLIST_DIR:-$SCRIPTS_DIR/skiplist/default}"
TRITON_TEST_SELECTFILE="${TRITON_TEST_SELECTFILE:=}"
TRITON_TEST_WARNING_REPORTS="${TRITON_TEST_WARNING_REPORTS:-false}"
TRITON_TEST_IGNORE_ERRORS="${TRITON_TEST_IGNORE_ERRORS:-false}"

if [[ $TEST_UNSKIP = true ]]; then
    TRITON_TEST_IGNORE_ERRORS=true
fi
# absolute path for the selected skip list
TRITON_TEST_SKIPLIST_DIR="$(cd "$TRITON_TEST_SKIPLIST_DIR" && pwd)"

pytest() {
    pytest_extra_args=(
        "--dist=worksteal"
    )

    if [[ -v TRITON_TEST_SUITE && $TRITON_TEST_REPORTS = true ]]; then
        mkdir -p "$TRITON_TEST_REPORTS_DIR"
        pytest_extra_args+=(
            "--junitxml=$TRITON_TEST_REPORTS_DIR/$TRITON_TEST_SUITE.xml"
        )
    fi

    if [[ -v TRITON_TEST_SUITE && $TRITON_TEST_WARNING_REPORTS = true ]]; then
        mkdir -p "$TRITON_TEST_REPORTS_DIR"
        pytest_extra_args+=(
            "--warnings-json-output-file=$TRITON_TEST_REPORTS_DIR/${TRITON_TEST_SUITE}-warnings.json"
        )
    fi

    if [[ -f $TRITON_TEST_SELECTFILE ]]; then
        pytest_extra_args+=(
            "--select-from-file=$TRITON_TEST_SELECTFILE"
        )
    fi

    if [[ ! -f $TRITON_TEST_SELECTFILE && -v TRITON_TEST_SUITE && -f $TRITON_TEST_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt ]]; then
        if [[ $TEST_UNSKIP = false ]]; then
            pytest_extra_args+=(
                "--skip-from-file=$TRITON_TEST_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt"
                "--select-fail-on-missing"
            )
        else
            pytest_extra_args+=(
                "--timeout=500"
                "--max-worker-restart=500"
            )
        fi
    fi

    export TEST_UNSKIP
    python -u -m pytest "${pytest_extra_args[@]}" "$@" || $TRITON_TEST_IGNORE_ERRORS
}

run_tutorial_test() {
    if [[ -f $TRITON_TEST_SELECTFILE ]] && ! grep -qF "$1" "$TRITON_TEST_SELECTFILE"; then
        return
    fi

    echo
    echo "****** Running $1 test ******"
    echo

    run_tutorial_args=(
        "--skip-list=$TRITON_TEST_SKIPLIST_DIR/tutorials.txt"
        "$1.py"
    )

    if [[ $TRITON_TEST_REPORTS = true ]]; then
        run_tutorial_args+=(
            "--reports=$TRITON_TEST_REPORTS_DIR"
        )
    fi

    python -u "$SCRIPTS_DIR/run_tutorial.py" "${run_tutorial_args[@]}" || $TRITON_TEST_IGNORE_ERRORS
}

capture_runtime_env() {
    mkdir -p "$TRITON_TEST_REPORTS_DIR"

    set +u
    echo "$CMPLR_ROOT" > $TRITON_TEST_REPORTS_DIR/cmplr_version.txt
    echo "$MKLROOT" > $TRITON_TEST_REPORTS_DIR/mkl_version.txt
    set -u

    # Exit script execution as long as one of those components is not found.
    local TRITON_COMMIT=""
    WHEELS=($SCRIPTS_DIR/../dist/*.whl)
    # This covers cases when multiple whls are found, it will get the commit id only when they have the same commit id
    # otherwise this script fail to execute
    if [[ "${#WHEELS[@]}" -gt 1 ]]; then
        TRITON_COMMIT=$(echo "${WHEELS[0]}" | sed -n 's/.*git\([a-zA-Z0-9]*\)[^a-zA-Z0-9].*/\1/p')
        for file in "${WHEELS[@]}"; do
            CUR_TRITON_COMMIT=$(echo "$file" | sed -n 's/.*git\([a-zA-Z0-9]*\)[^a-zA-Z0-9].*/\1/p')
            if [[ "$TRITON_COMMIT" != "$CUR_TRITON_COMMIT" ]]; then
                echo "ERROR: Multiple wheels found"
                exit 1
            fi
        done
    fi
    # This covers 3 cases: no whl, one whl with commit, one whl without commit
    if [[ "${#WHEELS[@]}" -eq 1 ]]; then
        TRITON_COMMIT=$(echo "${WHEELS[0]}" | sed -n 's/.*git\([a-zA-Z0-9]*\)[^a-zA-Z0-9].*/\1/p')
    fi

    if [[ $TRITON_PROJ && ! $TRITON_COMMIT ]]; then
        TRITON_COMMIT=$(cd $TRITON_PROJ && git rev-parse --short HEAD)
    fi
    if [[ ! $TRITON_COMMIT ]]; then
        echo "ERROR: Triton wheel package or source code is not found"
        exit 1
    fi
    echo "$TRITON_COMMIT" > $TRITON_TEST_REPORTS_DIR/triton_commit_id.txt
    cp $TRITON_TEST_REPORTS_DIR/triton_commit_id.txt $TRITON_TEST_REPORTS_DIR/tests_commit_id.txt

    source $SCRIPTS_DIR/capture-hw-details.sh --quiet
    echo "$LIBIGC1_VERSION" > $TRITON_TEST_REPORTS_DIR/libigc1_version.txt
    echo "$LEVEL_ZERO_VERSION" > $TRITON_TEST_REPORTS_DIR/level-zero_version.txt
    echo "$AGAMA_VERSION" > $TRITON_TEST_REPORTS_DIR/agama_driver_version.txt
    echo "$GPU_DEVICE" > $TRITON_TEST_REPORTS_DIR/gpu.txt

    python -c 'import platform; print(platform.python_version())' > $TRITON_TEST_REPORTS_DIR/triton_version.txt
    python -c 'import triton; print(triton.__version__)' >  $TRITON_TEST_REPORTS_DIR/triton_version.txt
    python -c 'import torch; print(torch.__version__)' > $TRITON_TEST_REPORTS_DIR/pytorch_version.txt
}

ensure_spirv_dis() {
    # Does not work on Windows
    if [[ $OSTYPE = msys ]]; then
        return
    fi
    export PATH="$HOME/.local/bin:$PATH"
    local spirv_dis="$(which spirv-dis || true)"
    if [[ $spirv_dis ]]; then
        echo "Found spirv-dis at $spirv_dis"
        return
    fi
    echo "Installing spirv-dis to $HOME/.local/bin"
    mkdir -p ~/.local/bin
    curl -sSL https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.xz | tar Jxf - -C $HOME/.local/bin --strip-components 3 --no-anchored spirv-dis
}
