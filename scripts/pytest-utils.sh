TIMESTAMP="$(date '+%Y%m%d%H%M%S')"

TRITON_TEST_REPORTS="${TRITON_TEST_REPORTS:-false}"
TRITON_TEST_REPORTS_DIR="${TRITON_TEST_REPORTS_DIR:-$BASE/reports/$TIMESTAMP}"

pytest() {
    pytest_extra_args=()

    if [[ -v TRITON_TEST_SUITE && $TRITON_TEST_REPORTS = true ]]; then
        mkdir -p "$TRITON_TEST_REPORTS_DIR"
        pytest_extra_args+=(
            "--junitxml=$TRITON_TEST_REPORTS_DIR/$TRITON_TEST_SUITE.xml"
        )
    fi

    python3 -m pytest "${pytest_extra_args[@]}" "$@"
}
