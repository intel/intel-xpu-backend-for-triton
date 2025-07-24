#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u

# --- Trap Handler ---
# No temporary files to clean up in this version.

# --- Script Start ---
start_time=$(date +%s)

# Record script start timestamp
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$0")"

# Define the full path to the results directory
RESULTS_DIR="$TIMESTAMP-weekly"

# Define paths to the helper scripts (assuming they are in the same directory as this script)
COLLECT_SCRIPT="$SCRIPT_DIR/collect_tests.py"
RUNNER_SCRIPT="$SCRIPT_DIR/run_tests_sequentially.sh"

# Define the specific test files you want to run
TEST_FILES=(
    "../hoshibara-pytorch/test/inductor/test_flex_attention.py"
    "../hoshibara-pytorch/test/inductor/test_flex_decoding.py"
)

# Check if the specified test files exist before starting
for TEST_FILE in "${TEST_FILES[@]}"; do
    if [ ! -f "$TEST_FILE" ]; then
        echo "Error: Specified test file not found: $TEST_FILE" >&2
        echo "Please ensure the path is correct relative to where you run this script." >&2
        exit 1 # Exit if any of the specified test files are missing
    fi
done


# --- Setup ---

# Create the timestamped directory for results
echo "Creating results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR" # Use -p to avoid error if directory already exists (unlikely with timestamp, but safe)


# Collect environment information
echo "Collecting environment information..."
# Assuming collect_env.py is one directory level up from where this script is located
COLLECT_ENV_SCRIPT="$SCRIPT_DIR/../collect_env.py"
# Check if the environment collection script exists and is executable
if [ ! -s "$COLLECT_ENV_SCRIPT" ]; then
    echo "Warning: collect_env.py script not found or not executable at $COLLECT_ENV_SCRIPT. Skipping environment collection." >&2
    # Do not exit, just warn and continue script execution
else
    # Redirect output to the results directory
    set +e
    python "$COLLECT_ENV_SCRIPT" > "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ pip list | grep transformers\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    pip list | grep transformers >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep igc\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep igc >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep dkms\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep dkms >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo "Done collecting environment info." >> "$RESULTS_DIR/collect_env.log" # Append Done message to the log file
    echo "Environment log saved to $RESULTS_DIR/collect_env.log"
    set -e
fi


TORCH_LOGS="+output_code" python run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 10 \
  --input-tokens 1024 --num-warmup 2 --num-iter 4 --compile --profile >> "$RESULTS_DIR/llama31.compile.xpu.profile.log" 2>&1

# --- Test Execution Loop ---

# Loop through each specified test file and run the collection and execution sequence
for TEST_FILE in "${TEST_FILES[@]}"; do
    echo "--- Processing tests for $TEST_FILE ---"

    # Check if the test file exists (already checked, but defensive)
     if [ ! -f "$TEST_FILE" ]; then
        echo "Error: Test file not found at path $TEST_FILE. Skipping this iteration." >&2
        continue # Skip to the next file
     fi

    # Define the permanent file name for the collected commands within the results directory.
    # Derived from the test file's base name and placed inside $RESULTS_DIR.
    COMMANDS_FILE_PERMANENT="$RESULTS_DIR/$(basename "$TEST_FILE" .py).commands.txt"

    # --- Step 1: Collect Test Commands ---
    # Call the collect_tests.py script. It takes the original test file path
    # and the path where to save the commands.
    # Paths are relative to the directory where THIS script is run.
    echo "Collecting commands for $TEST_FILE into $COMMANDS_FILE_PERMANENT..."
    # The python script exit status is checked by 'set -e'.
    python "$COLLECT_SCRIPT" "$TEST_FILE" "$COMMANDS_FILE_PERMANENT"
    echo "Finished collecting commands."

    # Check if the commands file was actually populated by the collector script.
    # '-s' checks if the file exists and has a size greater than zero.
    if [ ! -s "$COMMANDS_FILE_PERMANENT" ]; then
         echo "Warning: No test commands were collected or found for $TEST_FILE. Skipping execution for this file." >&2
         # The command file exists but is empty, it remains in the results directory.
         continue # Skip to the next test file in the loop.
    fi


    # --- Step 2: Run Collected Commands Sequentially ---
    # Call the run_tests_sequentially.sh script. It takes the commands file path
    # and the original test file path.
    # Paths are relative to the directory where THIS script is run.
    # We redirect its entire standard output and standard error to the test log file
    # within the results directory.
    TEST_LOG_FILE="$RESULTS_DIR/$(basename "$TEST_FILE" .py).result.log"
    echo "Running collected commands from $COMMANDS_FILE_PERMANENT for $TEST_FILE."
    echo "Full log will be saved to $TEST_LOG_FILE"
    # The bash runner script exit status is checked by 'set -e'.
    bash "$RUNNER_SCRIPT" "$COMMANDS_FILE_PERMANENT" > "$TEST_LOG_FILE" 2>&1
    # Note: Any 'tee' commands within run_tests_sequentially.sh will now have their
    # console output redirected into the log file as well.
    echo "Finished running commands for $TEST_FILE."


    echo "" # Add a blank line for readability between test file runs

done # End loop over test files


# --- Script End ---

# Create the final marker file in the results directory.
echo "Done" > "$RESULTS_DIR/finish.log"
echo "All specified test runs completed."
echo "Detailed logs and results, including collected commands, are located in the '$RESULTS_DIR/' directory."
echo "Command files: $RESULTS_DIR/*.commands.txt"
echo "Test logs: $RESULTS_DIR/*.test.log"


# No temporary file cleanup needed as commands files are permanent results.
# Trap handler is not needed as no temporary files are created.

# Exit with status 0 to indicate overall script success (unless set -e caused an earlier exit).
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds."
echo "Total elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds." >> "$RESULTS_DIR/finish.log"
echo "Script completed successfully at $(date '+%Y-%m-%d %H:%M:%S')."

exit 0