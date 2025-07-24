#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    # Print usage to stderr
    echo "Usage: $0 <commands_file>" >&2
    exit 1
fi

COMMANDS_FILE="$1"

# Check if the commands file exists and is readable
if [ ! -r "$COMMANDS_FILE" ]; then
    # Print error to stderr
    echo "Error: Commands file not found or not readable: $COMMANDS_FILE" >&2
    exit 1
fi

# --- Execution Start Messages (to stdout) ---
# These messages will be captured by the calling script's redirection
echo "--- Starting test execution at $(date) ---"
echo "--- Commands read from: $COMMANDS_FILE ---"
# We no longer know the final log file name here, so omit that message


# Read commands line by line from the file and execute them
# The < "$COMMANDS_FILE" redirects the file content to the while loop's stdin
while IFS= read -r command || [ -n "$command" ]; do
  # Trim leading/trailing whitespace from the command line
  command=$(echo "$command" | xargs)

  if [ -z "$command" ]; then # Skip empty or whitespace-only lines
    continue
  fi # <--- Corrected: Use 'fi' to close the if block

  # --- Log Command Execution (to stdout) ---
  echo ""
  echo "--- Running: $command ---"
  start_time=$(date +%s) # Capture start time

  # --- Execute the command safely (disable set -e temporarily) ---
  # Disable immediate exit on error, so the script continues even if 'eval "$command"' fails.
  set +e
  eval "$command"
  exit_status=$? # Capture exit status *immediately* after the command
  # Re-enable immediate exit on error for commands *after* this eval (e.g., the next 'echo' or file operations).
  set -e
  # --- End safe execution block ---

  end_time=$(date +%s) # Capture end time
  duration=$((end_time - start_time)) # Calculate duration

  # --- Log Execution Result (to stdout) ---
  echo "--- Finished: $command ---"
  echo "--- Exit Status: $exit_status (Took ${duration} seconds) ---"
  echo ""

done < "$COMMANDS_FILE"


# --- Execution End Message (to stdout) ---
echo ""
echo "--- Test execution finished at $(date) ---"

# Exit with a success status for the runner script itself.
# Individual test failures are indicated by the Exit Status in the log.
exit 0