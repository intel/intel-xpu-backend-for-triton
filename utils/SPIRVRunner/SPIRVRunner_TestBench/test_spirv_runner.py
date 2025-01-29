import pytest
import subprocess
import os

# Path to the SPIRVRunner executable
SPIRV_RUNNER_PATH = "/data/kballeda/Kali/0122_ci_enable/intel-xpu-backend-for-triton/utils/SPIRVRunner/build/SPIRVRunner"
SPIRV_RUNNER_TESTS= "/data/kballeda/Kali/0122_ci_enable/intel-xpu-backend-for-triton/utils/SPIRVRunner/SPIRVRunner_TestBench"
# Define CLI arguments per directory
SPIRV_CLI_ARGS = {
    SPIRV_RUNNER_TESTS + f"/add_kernel": ["-o", "tensor_2", "-p"]
}

@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
def test_argument_parsing():
    """Test that SPIRVRunner correctly handles argument parsing."""
    try:
        print("Running SPIRVRunner with --help...")
        result = subprocess.run([SPIRV_RUNNER_PATH, "--help"], capture_output=True, text=True, check=True)
        print("SPIRVRunner output:", result.stdout)
        assert "Usage" in result.stdout, "Help message not displayed correctly"
    except subprocess.CalledProcessError as e:
        print("Error executing SPIRVRunner:", e)
        pytest.fail(f"SPIRVRunner failed to execute: {e}")

@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
def test_invalid_argument():
    """Test SPIRVRunner's response to an invalid argument."""
    try:
        print("Running SPIRVRunner with an invalid argument...")
        result = subprocess.run([SPIRV_RUNNER_PATH, "--invalid-arg"], capture_output=True, text=True)
        print("SPIRVRunner stderr:", result.stderr)
        assert result.returncode != 0, "Invalid argument should result in an error"
        assert "error" in result.stderr.lower(), "Error message not displayed for invalid argument"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing SPIRVRunner:", e)
        pytest.fail(f"SPIRVRunner failed unexpectedly: {e}")

@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
@pytest.mark.parametrize("spirv_test_dir", SPIRV_CLI_ARGS.keys())
def test_spirv_execution(spirv_test_dir):
    """Test SPIRVRunner's ability to execute SPIR-V files from multiple directories with specific arguments."""
    if not os.path.exists(spirv_test_dir):
        print(f"Skipping test: Directory {spirv_test_dir} not found")
        pytest.skip(f"Test SPIR-V directory {spirv_test_dir} not found")    
    cli_args = SPIRV_CLI_ARGS.get(spirv_test_dir, [])   
    result = subprocess.run([SPIRV_RUNNER_PATH] + cli_args, capture_output=True, text=True, check=True, cwd=spirv_test_dir)
    print("SPIRVRunner output:", result.stdout)

if __name__ == "__main__":
    # Run the tests using pytest
    print("Starting pytest execution...")
    pytest.main()
