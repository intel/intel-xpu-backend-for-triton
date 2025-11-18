import pytest
import subprocess
import os

# Path to the SPIRVRunner executable
SPIRV_RUNNER_PATH = os.getenv("SPIRV_RUNNER_PATH")
SPIRV_RUNNER_TESTS = os.getenv("SPIRV_RUNNER_TESTS")

# Define CLI arguments per directory
SPIRV_CLI_ARGS = {
    os.path.join(SPIRV_RUNNER_TESTS, "add_kernel"): ["-o", "tensor_2", "-p", "-v", "expected_output.pt"],
    os.path.join(SPIRV_RUNNER_TESTS, "dot"): ["-o", "tensor_3", "-p", "-v", "expected_output.pt"]
}


@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
def test_argument_parsing():
    """Test that SPIRVRunner correctly handles argument parsing."""
    try:
        print("Running SPIRVRunner with --help...")
        result = subprocess.run([SPIRV_RUNNER_PATH, "--help"], capture_output=True, text=True, check=True)
        print("SPIRVRunner output:", result.stdout)
        assert "USAGE" in result.stdout, "Help message not displayed correctly"
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
        assert "unknown command" in result.stderr.lower(), "Error message not displayed for invalid argument"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing SPIRVRunner:", e)
        pytest.fail(f"SPIRVRunner failed unexpectedly: {e}")


@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
def test_args_json_gen():
    """Test generation of serialized arguments in JSON format."""
    try:
        print("Running SPIRVRunner to generate serialized args/Tensor data from Triton ...")
        os.environ['TRITON_XPU_DUMP_SPIRV_KERNEL_ARGS'] = './'
        target_dir = os.path.join(SPIRV_RUNNER_TESTS, "add_kernel")
        os.chdir(target_dir)
        result = subprocess.run(["python3", "01-vector-add.py"], capture_output=True, text=True)
        print("SPIRVRunner stderr:", result.stderr)
        os.environ.pop("TRITON_XPU_DUMP_SPIRV_KERNEL_ARGS", None)
        result = subprocess.run([SPIRV_RUNNER_PATH, "-o", "tensor_2", "-v", "expected_output.pt"], capture_output=True,
                                text=True)
        print("SPIRVRunner stderr:", result.stderr)
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

    try:
        result = subprocess.run([SPIRV_RUNNER_PATH] + cli_args, capture_output=True, text=True, check=True,
                                cwd=spirv_test_dir)
        print("SPIRVRunner output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"SPIRVRunner failed with return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        print(f"Command: {e.cmd}")
        print(f"Working directory: {spirv_test_dir}")
        print(f"Files in directory: {os.listdir(spirv_test_dir)}")
        raise


@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
def test_sycl_ls_with_trace():
    """Test sycl-ls command with SYCL_PI_TRACE enabled."""
    try:
        print("Running sycl-ls with SYCL_PI_TRACE=1...")

        env = os.environ.copy()
        env['SYCL_PI_TRACE'] = '1'

        result = subprocess.run(['sycl-ls'], capture_output=True, text=True, env=env, timeout=30)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

        assert result.returncode == 0, f"sycl-ls failed with return code {result.returncode}"

    except subprocess.TimeoutExpired:
        pytest.fail("sycl-ls command timed out")
    except FileNotFoundError:
        pytest.skip("sycl-ls command not found in PATH")
    except Exception as e:
        print(f"Unexpected error: {e}")
        pytest.fail(f"sycl-ls test failed: {e}")


def main():
    # Check if SPIRV_RUNNER_TESTS is set
    if not SPIRV_RUNNER_TESTS:
        raise EnvironmentError("SPIRV_RUNNER_TESTS environment variable is not set")
    """Main function to run all tests."""
    pytest.main()


if __name__ == "__main__":
    # Run the tests using pytest
    print("Starting pytest execution...")
    main()
