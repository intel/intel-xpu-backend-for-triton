TIMESTAMP="$(date '+%Y%m%d%H%M%S')"

SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TRITON_TEST_REPORTS="${TRITON_TEST_REPORTS:-false}"
TRITON_TEST_REPORTS_DIR="${TRITON_TEST_REPORTS_DIR:-$HOME/reports/$TIMESTAMP}"
TRITON_TEST_SKIPLIST_DIR="${TRITON_TEST_SKIPLIST_DIR:-$SCRIPTS_DIR/skiplist/default}"
TRITON_EXTRA_SKIPLIST_SUFFIXES="${TRITON_EXTRA_SKIPLIST_SUFFIXES:=}"
TRITON_TEST_SELECTFILE="${TRITON_TEST_SELECTFILE:=}"
TRITON_TEST_WARNING_REPORTS="${TRITON_TEST_WARNING_REPORTS:-false}"
TRITON_TEST_IGNORE_ERRORS="${TRITON_TEST_IGNORE_ERRORS:-false}"
TRITON_TEST_RUN_ALL="${TRITON_TEST_RUN_ALL:-false}"
TRITON_TEST_EXIT_CODE=0

if [[ $TEST_UNSKIP = true ]]; then
    TRITON_TEST_IGNORE_ERRORS=true
fi

# Handle test errors based on the selected mode:
#   --ignore-errors: swallow error, continue, exit 0
#   --run-all:       record error, continue, exit non-zero at end
#   default:         fail immediately (set -e kills the script)
handle_test_error() {
    if [[ $TRITON_TEST_IGNORE_ERRORS == true ]]; then
        return 0
    elif [[ $TRITON_TEST_RUN_ALL == true ]]; then
        TRITON_TEST_EXIT_CODE=1
        return 0
    else
        return 1
    fi
}
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
            SKIPFILES="$TRITON_TEST_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt"
            if [[ -n "$TRITON_EXTRA_SKIPLIST_SUFFIXES" ]]; then
                IFS=',' read -ra SUFFIXES <<< "$TRITON_EXTRA_SKIPLIST_SUFFIXES"
                for SUFFIX in "${SUFFIXES[@]}"; do
                    SKIPFILE="$TRITON_TEST_SKIPLIST_DIR/${TRITON_TEST_SUITE}-${SUFFIX}.txt"
                    if [[ -f "$SKIPFILE" ]]; then
                        SKIPFILES+=";$SKIPFILE"
                    else
                        echo "ERROR: $SKIPFILE not found"
                        exit 1
                    fi
                done
            fi
            pytest_extra_args+=(
                "--skip-from-file=$SKIPFILES"
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
    python -u -m pytest "${pytest_extra_args[@]}" "$@" || handle_test_error
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

# Hex-encoded SPIR-V module declaring OpCapability PredicatedIOINTEL (6257) and
# using OpPredicatedLoad/StoreINTEL. spirv-dis disassembles it (exit 0) only when
# it understands SPV_INTEL_predicated_io; older builds fail with
# "Invalid capability operand: 6257". spirv-dis reads this hex text from stdin,
# so no binary fixture is needed. Never pass --handle-unknown-opcodes: it would
# re-emit unknown ops and exit 0, hiding the missing support.
PREDICATED_IO_PROBE="\
07230203 00010000 00000000 0000000c 00000000 00020011 00000004 00020011 00000006 \
00020011 00001871 0007000a 5f565053 45544e49 72705f4c 63696465 64657461 006f695f \
0003000e 00000001 00000002 0005000f 00000006 00000008 626f7270 00000065 00020013 \
00000001 00030021 00000002 00000001 00020014 00000003 00040015 00000004 00000020 \
00000000 00040020 00000005 00000007 00000004 00030029 00000003 00000006 0004002b \
00000004 00000007 00000000 00050036 00000001 00000008 00000000 00000002 000200f8 \
00000009 0004003b 00000005 0000000a 00000007 00061872 00000004 0000000b 0000000a \
00000006 00000007 00041873 0000000a 0000000b 00000006 000100fd 00010038"

spirv_dis_supports_predicated_io() {
    local spirv_dis="$1"
    [[ -x "$spirv_dis" ]] || command -v -- "$spirv_dis" >/dev/null 2>&1 || return 1
    printf '%s' "$PREDICATED_IO_PROBE" | "$spirv_dis" - >/dev/null 2>&1
}

find_spirv_dis_with_predicated_io() {
    local path_dir candidate_dir candidate
    local IFS=:
    for path_dir in $PATH; do
        [[ -n "$path_dir" ]] || path_dir=.
        candidate_dir="$(cd -- "$path_dir" 2>/dev/null && pwd -P)" || continue
        candidate="$candidate_dir/spirv-dis"
        if spirv_dis_supports_predicated_io "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

ensure_spirv_dis() {
    # Does not work on Windows
    if [[ $OSTYPE = msys || $OSTYPE = cygwin ]]; then
        return
    fi

    export PATH="$HOME/.local/bin:$PATH"

    # Make a capable spirv-dis the first one on PATH: triton.knobs.intel.spirv_dis
    # resolves via shutil.which("spirv-dis") and does not check the extension, so
    # the probed binary must win PATH order. A user may point at one explicitly
    # via TRITON_SPIRV_DIS_PATH.
    if [[ -n "${TRITON_SPIRV_DIS_PATH:-}" ]]; then
        if spirv_dis_supports_predicated_io "$TRITON_SPIRV_DIS_PATH"; then
            export PATH="$(cd -- "$(dirname -- "$TRITON_SPIRV_DIS_PATH")" && pwd -P):$PATH"
            echo "Using spirv-dis from TRITON_SPIRV_DIS_PATH: $TRITON_SPIRV_DIS_PATH"
            return
        fi
        echo "TRITON_SPIRV_DIS_PATH does not support SPV_INTEL_predicated_io: $TRITON_SPIRV_DIS_PATH" >&2
        return 1
    fi

    local capable_spirv_dis
    if capable_spirv_dis="$(find_spirv_dis_with_predicated_io)"; then
        export PATH="$(cd -- "$(dirname -- "$capable_spirv_dis")" && pwd -P):$PATH"
        echo "Using existing spirv-dis with SPV_INTEL_predicated_io support: $capable_spirv_dis"
        return
    fi

    # No capable spirv-dis on PATH: build from source.
    # The commit is pinned because SPV_INTEL_predicated_io support is not in any
    # tagged SPIRV-Tools release yet (post-v2026.2, KhronosGroup/SPIRV-Tools#6665).
    # FIXME: Switch back to Vulkan SDK tarball once a released SDK includes it.
    echo "Building spirv-dis from source to $HOME/.local/bin"
    mkdir -p "$HOME/.local/bin"
    (
        set -e
        build_dir="$(mktemp -d)"
        trap 'rm -rf "$build_dir"' EXIT
        git clone https://github.com/KhronosGroup/SPIRV-Tools.git "$build_dir/SPIRV-Tools"
        git -C "$build_dir/SPIRV-Tools" checkout 4c2ec2a09b7fbeff1dc64cb9f857d77403a3c25f
        python3 "$build_dir/SPIRV-Tools/utils/git-sync-deps"
        cmake -B "$build_dir/build" -S "$build_dir/SPIRV-Tools" -DCMAKE_BUILD_TYPE=Release -DSPIRV_SKIP_TESTS=ON
        cmake --build "$build_dir/build" -j"$(nproc)" --target spirv-dis
        cp "$build_dir/build/tools/spirv-dis" "$HOME/.local/bin/"
    ) || return 1

    local built_spirv_dis="$HOME/.local/bin/spirv-dis"
    if ! spirv_dis_supports_predicated_io "$built_spirv_dis"; then
        echo "Built spirv-dis does not support SPV_INTEL_predicated_io: $built_spirv_dis" >&2
        return 1
    fi
}
