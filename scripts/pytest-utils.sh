TIMESTAMP="$(date '+%Y%m%d%H%M%S')"

SCRIPTS_DIR="${SCRIPTS_DIR:-$PWD/scripts}"
TRITON_TEST_REPORTS="${TRITON_TEST_REPORTS:-false}"
TRITON_TEST_REPORTS_DIR="${TRITON_TEST_REPORTS_DIR:-$HOME/reports/$TIMESTAMP}"
TRITON_TEST_SKIPLIST_DIR="${TRITON_TEST_SKIPLIST_DIR:-$SCRIPTS_DIR/skiplist/default}"

# absolute path for the selected skip list
TRITON_TEST_SKIPLIST_DIR="$(cd "$TRITON_TEST_SKIPLIST_DIR" && pwd)"
# absolute path for the current skip list
CURRENT_SKIPLIST_DIR="$SCRIPTS_DIR/skiplist/current"

pytest() {
    pytest_extra_args=()

    if [[ -v TRITON_TEST_SUITE && $TRITON_TEST_REPORTS = true ]]; then
        mkdir -p "$TRITON_TEST_REPORTS_DIR"
        pytest_extra_args+=(
            "--junitxml=$TRITON_TEST_REPORTS_DIR/$TRITON_TEST_SUITE.xml"
        )
    fi

    if [[ -v TRITON_TEST_SUITE && -f $TRITON_TEST_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt ]]; then
        mkdir -p "$CURRENT_SKIPLIST_DIR"
        # skip comments in the skiplist
        sed -e '/^#/d' "$TRITON_TEST_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt" > "$CURRENT_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt"
#        pytest_extra_args+=(
#            "--deselect-from-file=$CURRENT_SKIPLIST_DIR/$TRITON_TEST_SUITE.txt"
#            "--select-fail-on-missing"
#        )
    fi

    python3 -m pytest "${pytest_extra_args[@]}" "$@" || true
}
