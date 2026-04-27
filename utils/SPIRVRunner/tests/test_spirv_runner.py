import pytest
import subprocess
import os
import tempfile
import json

# Path to the SPIRVRunner executable
SPIRV_RUNNER_PATH = os.getenv("SPIRV_RUNNER_PATH")
SPIRV_RUNNER_TESTS = os.getenv("SPIRV_RUNNER_TESTS")

# Define CLI arguments per directory
SPIRV_CLI_ARGS = {os.path.join(SPIRV_RUNNER_TESTS, "dot"): ["-o", "tensor_3", "-p", "-v", "expected_output.pt"]}


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
        with tempfile.TemporaryDirectory(prefix="spirv_runner_cache_") as cache_root:
            target_dir = os.path.join(SPIRV_RUNNER_TESTS, "add_kernel")
            env = os.environ.copy()
            env["TRITON_XPU_ENABLE_DUMP_SPIRV_KERNEL_ARGS"] = "1"
            env["TRITON_CACHE_DIR"] = cache_root
            result = subprocess.run(["python3", "01-vector-add.py"], capture_output=True, text=True, cwd=target_dir,
                                    env=env)
            print("SPIRVRunner stderr:", result.stderr)

            dump_dir = None
            for root, _, files in os.walk(cache_root):
                if "args_data.json" not in files:
                    continue
                args_path = os.path.join(root, "args_data.json")
                with open(args_path) as args_file:
                    args_data = json.load(args_file)
                if args_data.get("spv_name") == "add_kernel.spv":
                    dump_dir = root
                    break

            assert dump_dir is not None, f"args_data.json for add_kernel.spv not found under cache root: {cache_root}"
            result = subprocess.run([SPIRV_RUNNER_PATH, "-d", dump_dir, "-o", "tensor_2"], capture_output=True,
                                    text=True, check=True)
            print("SPIRVRunner stderr:", result.stderr)
            assert any(name.startswith("cpp_outs_") and name.endswith(".pt") for name in os.listdir(dump_dir)), \
                f"No SPIRVRunner output tensor file found in dump dir: {dump_dir}"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing SPIRVRunner:", e)
        if e.stdout:
            print("SPIRVRunner stdout:", e.stdout)
        if e.stderr:
            print("SPIRVRunner stderr:", e.stderr)
        pytest.fail(f"SPIRVRunner failed unexpectedly: {e}")


@pytest.mark.skipif(not os.path.exists(SPIRV_RUNNER_PATH), reason="SPIRVRunner executable not found")
@pytest.mark.parametrize("spirv_test_dir", SPIRV_CLI_ARGS.keys())
def test_spirv_execution(spirv_test_dir):
    """Test SPIRVRunner's ability to execute SPIR-V files from multiple directories with specific arguments."""
    if not os.path.exists(spirv_test_dir):
        print(f"Skipping test: Directory {spirv_test_dir} not found")
        pytest.skip(f"Test SPIR-V directory {spirv_test_dir} not found")
    cli_args = SPIRV_CLI_ARGS.get(spirv_test_dir, [])
    result = subprocess.run([SPIRV_RUNNER_PATH] + cli_args, capture_output=True, text=True, check=True,
                            cwd=spirv_test_dir)
    print("SPIRVRunner output:", result.stdout)


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
