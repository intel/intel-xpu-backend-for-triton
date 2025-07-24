#!/usr/bin/env python3
import sys
import subprocess
import os

def collect_and_save_test_commands(test_file_path, output_file_path):
    """
    Uses pytest --collect-only --quiet to list test cases, parse
    file::Class::method format, and saves commands in
    'python <file> <Class.method>' format to a file.

    Args:
        test_file_path: Path to the Python test file.
        output_file_path: Path to the file where commands will be saved.

    Returns:
        True if commands were collected and saved successfully, False otherwise.
    """
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found: {test_file_path}", file=sys.stderr)
        return False

    # Command to get the quiet collection output (file::id format)
    command = ["pytest", "--quiet", "--collect-only", test_file_path]

    try:
        # Run pytest and capture output
        # Use check=True to raise an error if the command fails
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().splitlines()

        commands_to_save = []
        for line in output_lines:
            # Skip empty lines
            if not line:
                continue

            # Expecting format like: test/file.py::TestClass::test_method
            # Split only on the first '::' to separate file path and the rest of the ID
            parts = line.split('::', 1)

            if len(parts) == 2:
                file_part = parts[0] # e.g., test/inductor/test_flex_attention.py
                id_part = parts[1]   # e.g., TestFlexAttentionXPU::test_GQA_score_mod3_xpu_float16

                # Replace remaining '::' in the ID part with '.'
                formatted_id = id_part.replace('::', '.') # e.g., TestFlexAttentionXPU.test_GQA_score_mod3_xpu_float16

                # Construct the desired command format
                # Note: This specific command format 'python <file> <dotted_id>'
                # is not the standard way pytest is typically invoked for a single test case.
                # Standard is usually 'pytest <file::id>' or 'pytest <file>::<Class>::<method>'.
                # This assumes your test setup can handle being run via 'python <file>'
                # and correctly interpret the dotted ID argument to run the specific test.
                command_line = f"python ../hoshibara-pytorch/{file_part} {formatted_id}"
                commands_to_save.append(command_line)
            else:
                # This line didn't match the expected file::id format, maybe a warning or skip
                print(f"Warning: Skipping unexpected output line from pytest: {line}", file=sys.stderr)


        if not commands_to_save:
            print(f"Warning: No parseable test cases found in the output for {test_file_path}.", file=sys.stderr)
            # Continue to file writing to create/overwrite an empty file
            pass

        # Write commands to the output file
        with open(output_file_path, 'w') as f:
            for cmd in commands_to_save:
                f.write(cmd + '\n')

        print(f"Successfully collected {len(commands_to_save)} test commands and saved to {output_file_path}", file=sys.stderr)
        return True

    except FileNotFoundError:
        print(f"Error: pytest command not found. Is pytest installed and in your PATH?", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during pytest collection for {test_file_path}:", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        # Print collected stdout if there were no errors but check=True failed for some reason
        # print(f"Collected stdout so far:\n{result.stdout}", file=sys.stderr)
        return False
    except IOError as e:
        print(f"Error writing commands to file {output_file_path}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python collect_tests.py <test_file_path> <output_commands_file>", file=sys.stderr)
        sys.exit(1)

    test_file = sys.argv[1]
    output_file = sys.argv[2]

    if collect_and_save_test_commands(test_file, output_file):
        sys.exit(0)
    else:
        sys.exit(1)