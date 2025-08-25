#!/bin/bash

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Input arguments
RESULT_DIR=$1
TORCH_DIR=$2
SUITES=${3:-'["huggingface", "torchbench", "timm_models"]'}
MODES=${4:-'["training", "inference"]'}
DTYPES=${5:-'["float32", "bfloat16", "float16", "amp_bf16", "amp_fp16"]'}

# Function to convert JSON array to space-separated string
convert_json_array() {
    local input="$1"
    # Check if input looks like a JSON array
    if [[ "$input" =~ ^\[.*\]$ ]]; then
        python3 -c "import json; print(' '.join(json.loads('$input')))"
    else
        echo "$input"
    fi
}

# Check if required arguments are provided
if [ -z "$RESULT_DIR" ] || [ -z "$TORCH_DIR" ]; then
    echo "Usage: $0 <RESULT_DIR> <TORCH_DIR> [SUITES] [MODES] [DTYPES]" >&2
    exit 1
fi

# Validate that directories exist
if [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: RESULT_DIR '$RESULT_DIR' does not exist" >&2
    exit 1
fi

if [ ! -d "$TORCH_DIR" ]; then
    echo "ERROR: TORCH_DIR '$TORCH_DIR' does not exist" >&2
    exit 1
fi

# Check if the Python script exists
PYTHON_SCRIPT="$TORCH_DIR/.github/ci_expected_accuracy/check_expected.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script '$PYTHON_SCRIPT' not found" >&2
    exit 1
fi

# Convert JSON arrays to space-separated strings if needed
SUITES=$(convert_json_array "$SUITES")
MODES=$(convert_json_array "$MODES")
DTYPES=$(convert_json_array "$DTYPES")

# Convert space-separated strings to arrays
IFS=' ' read -ra suites <<< "$SUITES"
IFS=' ' read -ra modes <<< "$MODES"
IFS=' ' read -ra dtypes <<< "$DTYPES"

# Variable to collect all failed model information
failed_models_output=""
missing_files=()
exit_code=0

echo "$SUITES"
echo "$MODES"
echo "$DTYPES"

# Nested loops
for suite in "${suites[@]}"; do
    for mode in "${modes[@]}"; do
        # Skip inference-with-freezing mode since there is reference for it in pytorch project
        if [ "$mode" = "inference-with-freezing" ]; then
            echo "Skipping mode: $mode"
            continue
        fi

        for dtype in "${dtypes[@]}"; do
            CSV_FILE="$RESULT_DIR/logs-$suite-$dtype-$mode-accuracy/$suite/$dtype/inductor_${suite}_${dtype}_${mode}_xpu_accuracy.csv"

            # Check if CSV file exists
            if [ ! -f "$CSV_FILE" ]; then
                echo "Missing: $CSV_FILE"
                missing_files+=("$CSV_FILE")
                continue
            fi

            echo "Processing: $suite, $mode, $dtype"

            # Run the Python script and capture output
            output=$(python "$PYTHON_SCRIPT" \
                --driver rolling \
                --suite "$suite" \
                --mode "$mode" \
                --dtype "$dtype" \
                --csv_file "$CSV_FILE")

            # Print the output
            echo "$output"

            # Extract and concatenate summary lines
            summary_lines=$(echo "$output" | grep -E "(Real failed models:|Summary for)")
            if [ -n "$summary_lines" ]; then
                failed_models_output="${failed_models_output}${summary_lines}"$'\n'
            fi

        done
    done
done


echo "========================================="
echo "Summary of all results:"
echo "$failed_models_output"

# Find lines with actual failures (not "Real failed models: 0")
failed_line=$(echo "$failed_models_output" | grep "Real failed models:" | grep -v "Real failed models: 0" || true)

echo "========================================="
echo "Summary of only failed models:"
if [ -n "$failed_line" ]; then
    echo "$failed_line"
    echo "ERROR: Found failed models!"
    exit_code=1
else
    echo "SUCCESS: All models passed!"
fi

# Check for missing files first
if [ ${#missing_files[@]} -gt 0 ]; then
    echo "========================================="
    echo "ERROR: Missing files detected:"
    for file in "${missing_files[@]}"; do
        echo "  $file"
    done
    echo "Total missing files: ${#missing_files[@]}"
    exit_code=1
fi

exit $exit_code
