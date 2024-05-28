# generate skiplist for all test suite on a new platform
CORE_LOG=${TRITON_PROJ}/${GPU_TYPE}_${AGAMA_TYPE}_core.log
REGRESSION_LOG=${TRITON_PROJ}/${GPU_TYPE}_${AGAMA_TYPE}_regression.log

echo "Generating core test log ..."
run_core_tests > "${CORE_LOG}"
echo "Done generation"

echo "Generating regression test log ..."
run_regression_tests > "${REGRESSION_LOG}"
echo "Done generation"

mkdir -p "$TRITON_TEST_SKIPLIST_DIR"

parse_log() {
    local keyword="$1"
    local output_file="$2"
    if grep -q "SKIPPED $keyword" "$CORE_LOG"; then
        grep "SKIPPED $keyword" "$CORE_LOG" |
        sed -E "s/^(.*)$keyword(.*)/test\/unit\/$keyword\2/g; s/ //g" > "$output_file"
    fi
}

# language.txt & operators.txt
parse_log "language" "${TRITON_TEST_SKIPLIST_DIR}/language.txt"
parse_log "operators" "${TRITON_TEST_SKIPLIST_DIR}/operators.txt"

# subprocess.txt
if grep -q "test_subprocess.py" "${TRITON_TEST_SKIPLIST_DIR}/language.txt"; then
    grep "test_subprocess.py" "${TRITON_TEST_SKIPLIST_DIR}/language.txt" > ${TRITON_TEST_SKIPLIST_DIR}/subprocess.txt
    sed -i '/test_subprocess.py/d' ${TRITON_TEST_SKIPLIST_DIR}/language.txt
fi

# regression.txt
if grep -q "SKIPPED" ${REGRESSION_LOG}; then
    grep "SKIPPED" ${REGRESSION_LOG} |
    sed -E 's/^(.*)SKIPPED(.*)/test\/unit\/language\/\1/g; s/ //g' > ${TRITON_TEST_SKIPLIST_DIR}/regression.txt
fi

# DOTO: runtime.txt
