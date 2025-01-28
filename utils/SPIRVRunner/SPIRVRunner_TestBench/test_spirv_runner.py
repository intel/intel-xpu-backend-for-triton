import os
import subprocess
import pytest

# Define the root directory (current directory where the script is run)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Derive paths relative to the root directory
TEST_RESULTS_DIR = os.path.join(ROOT_DIR, ".", "test_results")
print(ROOT_DIR)
print(TEST_RESULTS_DIR)
'''
# Ensure test results directory exists
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def run_spirv_runner(spv_file):
    """Run the SPIRVRunner utility on a SPIR-V binary."""
    result = subprocess.run([SPIRVRUNNER_PATH, spv_file], capture_output=True, text=True)
    return result.stdout

def read_expected_output(test_name):
    """Read the expected output for a test case."""
    expected_output_path = os.path.join(EXPECTED_OUTPUTS_DIR, f"{test_name}.txt")
    with open(expected_output_path, "r") as f:
        return f.read()

def generate_test_cases():
    """Generate test cases dynamically based on SPIR-V binaries."""
    test_cases = []
    for spv_file in os.listdir(SPIRV_BINARIES_DIR):
        if spv_file.endswith(".spv"):
            test_name = os.path.splitext(spv_file)[0]
            test_cases.append((spv_file, test_name))
    return test_cases

@pytest.mark.parametrize("spv_file, test_name", generate_test_cases())
def test_spirv_runner(spv_file, test_name):
    """Dynamic test case for SPIRVRunner."""
    spv_path = os.path.join(SPIRV_BINARIES_DIR, spv_file)
    output = run_spirv_runner(spv_path)
    expected_output = read_expected_output(test_name)

    # Write output to test results directory
    result_path = os.path.join(TEST_RESULTS_DIR, f"{test_name}_result.txt")
    with open(result_path, "w") as f:
        f.write(output)

    # Assert that the output matches the expected output
    assert output == expected_output, f"Test {test_name} failed: Output does not match expected result."
'''
