import pytest
import subprocess
import os
import tempfile
import json

from triton.language.target_info import is_xpu_cri

# Path to the TritonKernelRunner executable
TRITON_KERNEL_RUNNER_PATH = os.getenv("TRITON_KERNEL_RUNNER_PATH")
TRITON_KERNEL_RUNNER_TESTS = os.getenv("TRITON_KERNEL_RUNNER_TESTS")

# Define CLI arguments per directory
SPIRV_CLI_ARGS = {os.path.join(TRITON_KERNEL_RUNNER_TESTS, "dot"): ["-o", "tensor_3", "-p", "-v", "expected_output.pt"]}


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
def test_argument_parsing():
    """Test that TritonKernelRunner correctly handles argument parsing."""
    try:
        print("Running TritonKernelRunner with --help...")
        result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "--help"], capture_output=True, text=True, check=True)
        print("TritonKernelRunner output:", result.stdout)
        assert "USAGE" in result.stdout, "Help message not displayed correctly"
    except subprocess.CalledProcessError as e:
        print("Error executing TritonKernelRunner:", e)
        pytest.fail(f"TritonKernelRunner failed to execute: {e}")


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
def test_invalid_argument():
    """Test TritonKernelRunner's response to an invalid argument."""
    try:
        print("Running TritonKernelRunner with an invalid argument...")
        result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "--invalid-arg"], capture_output=True, text=True)
        print("TritonKernelRunner stderr:", result.stderr)
        assert result.returncode != 0, "Invalid argument should result in an error"
        assert "unknown command" in result.stderr.lower(), "Error message not displayed for invalid argument"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing TritonKernelRunner:", e)
        pytest.fail(f"TritonKernelRunner failed unexpectedly: {e}")


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
def test_args_json_gen():
    """Test generation of serialized arguments in JSON format."""
    try:
        print("Running TritonKernelRunner to generate serialized args/Tensor data from Triton ...")
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            target_dir = os.path.join(TRITON_KERNEL_RUNNER_TESTS, "add_kernel")
            env = os.environ.copy()
            env["TRITON_XPU_ENABLE_DUMP_KERNEL_ARGS"] = "1"
            env["TRITON_CACHE_DIR"] = cache_root
            result = subprocess.run(["python3", "01-vector-add.py"], capture_output=True, text=True, cwd=target_dir,
                                    env=env)
            print("TritonKernelRunner stderr:", result.stderr)

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
            result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "-d", dump_dir, "-o", "tensor_2"], capture_output=True,
                                    text=True, check=True)
            print("TritonKernelRunner stderr:", result.stderr)
            assert any(name.startswith("cpp_outs_") and name.endswith(".pt") for name in os.listdir(dump_dir)), \
                f"No TritonKernelRunner output tensor file found in dump dir: {dump_dir}"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing TritonKernelRunner:", e)
        if e.stdout:
            print("TritonKernelRunner stdout:", e.stdout)
        if e.stderr:
            print("TritonKernelRunner stderr:", e.stderr)
        pytest.fail(f"TritonKernelRunner failed unexpectedly: {e}")


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
@pytest.mark.parametrize("spirv_test_dir", SPIRV_CLI_ARGS.keys())
def test_spirv_execution(spirv_test_dir):
    """Test TritonKernelRunner's ability to execute SPIR-V files from multiple directories with specific arguments."""
    if not os.path.exists(spirv_test_dir):
        print(f"Skipping test: Directory {spirv_test_dir} not found")
        pytest.skip(f"Test SPIR-V directory {spirv_test_dir} not found")
    cli_args = SPIRV_CLI_ARGS.get(spirv_test_dir, [])
    result = subprocess.run([TRITON_KERNEL_RUNNER_PATH] + cli_args, capture_output=True, text=True, check=True,
                            cwd=spirv_test_dir)
    print("TritonKernelRunner output:", result.stdout)


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
def test_dumped_build_flags_reflect_auto_grf_retry():
    """The repro dump must capture the GRF mode Triton actually ran with.

    `load_binary`'s runtime auto-large-GRF retry (driver.c) rewrites
    `build_flags` once a kernel spills past `REBUILD_SPILL_BYTES_PER_THREAD`.
    If `args_data.json` still had the pre-retry flags, replaying the dump here
    would build a different (un-escalated) binary than the one Triton
    actually executed -- the alignment issue #5529 reported.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            target_dir = os.path.join(TRITON_KERNEL_RUNNER_TESTS, "auto_grf")
            env = os.environ.copy()
            env["TRITON_XPU_ENABLE_DUMP_KERNEL_ARGS"] = "1"
            env["TRITON_CACHE_DIR"] = cache_root
            result = subprocess.run(["python3", "01-auto-grf.py"], capture_output=True, text=True, cwd=target_dir,
                                    env=env, check=True)
            print("auto-grf kernel stderr:", result.stderr)

            dump_dir = None
            for root, _, files in os.walk(cache_root):
                if "args_data.json" not in files:
                    continue
                args_path = os.path.join(root, "args_data.json")
                with open(args_path) as args_file:
                    args_data = json.load(args_file)
                if args_data.get("spv_name") == "auto_grf_kernel.spv":
                    dump_dir = root
                    break

            assert dump_dir is not None, f"args_data.json for auto_grf_kernel.spv not found under cache root: {cache_root}"
            with open(os.path.join(dump_dir, "args_data.json")) as args_file:
                args_data = json.load(args_file)

            large_grf_flag = "-cl-intel-512-GRF-per-thread" if is_xpu_cri() else "-cl-intel-256-GRF-per-thread"
            build_flags = args_data["build_flags"]
            assert large_grf_flag in build_flags, (f"dumped build_flags do not reflect the auto-large-GRF retry "
                                                   f"(expected {large_grf_flag!r} in {build_flags!r})")

            result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "-d", dump_dir, "-o", "tensor_0"], capture_output=True,
                                    text=True, check=True)
            print("TritonKernelRunner stderr:", result.stderr)
            assert any(name.startswith("cpp_outs_") and name.endswith(".pt") for name in os.listdir(dump_dir)), \
                f"No TritonKernelRunner output tensor file found in dump dir: {dump_dir}"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing test:", e)
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        pytest.fail(f"test_dumped_build_flags_reflect_auto_grf_retry failed unexpectedly: {e}")


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
def test_native_binary_dump_and_execution():
    """TritonKernelRunner must be able to replay a native-binary (AOT/zebin) dump.

    A kernel compiled with `generate_native_code=True` goes through
    `make_zebin`/`ocloc` instead of JIT SPIR-V, and its cached binary is a
    zebin, not a `.spv` file (see `metadata["binary_ext"] = "zebin"` in
    `make_zebin`). The repro dump has to say so (`is_native_binary`) and
    TritonKernelRunner has to load it with `ZE_MODULE_FORMAT_NATIVE` rather than
    `_IL_SPIRV` -- otherwise the "native binary" leg of issue #5529's
    alignment is unsupported, not just untested.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            target_dir = os.path.join(TRITON_KERNEL_RUNNER_TESTS, "native_binary")
            env = os.environ.copy()
            env["TRITON_XPU_ENABLE_DUMP_KERNEL_ARGS"] = "1"
            env["TRITON_CACHE_DIR"] = cache_root
            result = subprocess.run(["python3", "01-native-binary.py"], capture_output=True, text=True, cwd=target_dir,
                                    env=env, check=True)
            print("native-binary kernel stderr:", result.stderr)

            dump_dir = None
            for root, _, files in os.walk(cache_root):
                if "args_data.json" not in files:
                    continue
                args_path = os.path.join(root, "args_data.json")
                with open(args_path) as args_file:
                    args_data = json.load(args_file)
                if args_data.get("spv_name") == "native_binary_kernel.zebin":
                    dump_dir = root
                    break

            assert dump_dir is not None, \
                f"args_data.json for native_binary_kernel.zebin not found under cache root: {cache_root}"
            with open(os.path.join(dump_dir, "args_data.json")) as args_file:
                args_data = json.load(args_file)

            assert args_data.get("is_native_binary") is True, \
                f"dump does not record the kernel as a native binary: {args_data!r}"
            assert os.path.exists(os.path.join(dump_dir, "native_binary_kernel.zebin")), \
                f"dumped zebin file not found in {dump_dir}"

            result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "-d", dump_dir, "-o", "tensor_2"], capture_output=True,
                                    text=True, check=True)
            print("TritonKernelRunner stderr:", result.stderr)
            assert any(name.startswith("cpp_outs_") and name.endswith(".pt") for name in os.listdir(dump_dir)), \
                f"No TritonKernelRunner output tensor file found in dump dir: {dump_dir}"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing test:", e)
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        pytest.fail(f"test_native_binary_dump_and_execution failed unexpectedly: {e}")


@pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH), reason="TritonKernelRunner executable not found")
def test_native_binary_dump_reflects_auto_grf_retry():
    """A native binary that spills must dump/replay at its *escalated* GRF mode.

    `make_zebin` has its own auto-large-GRF retry (measuring real spill via
    `extract_spill_size_from_zebin`/`accepts_default_grf`), separate from
    driver.c's JIT retry. This is the native-binary analogue of
    `test_dumped_build_flags_reflect_auto_grf_retry`: it exercises the
    `generate_native_code=True` leg of issue #5529's alignment, which
    `test_native_binary_dump_and_execution` does not (that kernel is too
    small to spill).
    """
    try:
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            target_dir = os.path.join(TRITON_KERNEL_RUNNER_TESTS, "native_binary_auto_grf")
            env = os.environ.copy()
            env["TRITON_XPU_ENABLE_DUMP_KERNEL_ARGS"] = "1"
            env["TRITON_CACHE_DIR"] = cache_root
            result = subprocess.run(["python3", "01-native-binary-auto-grf.py"], capture_output=True, text=True,
                                    cwd=target_dir, env=env, check=True)
            print("native-binary auto-grf kernel stderr:", result.stderr)

            dump_dir = None
            for root, _, files in os.walk(cache_root):
                if "args_data.json" not in files:
                    continue
                args_path = os.path.join(root, "args_data.json")
                with open(args_path) as args_file:
                    args_data = json.load(args_file)
                if args_data.get("spv_name") == "native_binary_auto_grf_kernel.zebin":
                    dump_dir = root
                    break

            assert dump_dir is not None, (f"args_data.json for native_binary_auto_grf_kernel.zebin not found "
                                          f"under cache root: {cache_root}")
            with open(os.path.join(dump_dir, "args_data.json")) as args_file:
                args_data = json.load(args_file)

            assert args_data.get("is_native_binary") is True, \
                f"dump does not record the kernel as a native binary: {args_data!r}"

            large_grf_flag = "-cl-intel-512-GRF-per-thread" if is_xpu_cri() else "-cl-intel-256-GRF-per-thread"
            build_flags = args_data["build_flags"]
            assert large_grf_flag in build_flags, (
                f"dumped build_flags do not reflect make_zebin's auto-large-GRF retry "
                f"(expected {large_grf_flag!r} in {build_flags!r})")

            result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "-d", dump_dir, "-o", "tensor_0"], capture_output=True,
                                    text=True, check=True)
            print("TritonKernelRunner stderr:", result.stderr)
            assert any(name.startswith("cpp_outs_") and name.endswith(".pt") for name in os.listdir(dump_dir)), \
                f"No TritonKernelRunner output tensor file found in dump dir: {dump_dir}"
    except subprocess.CalledProcessError as e:
        print("Unexpected error executing test:", e)
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        pytest.fail(f"test_native_binary_dump_reflects_auto_grf_retry failed unexpectedly: {e}")


def main():
    # Check if TRITON_KERNEL_RUNNER_TESTS is set
    if not TRITON_KERNEL_RUNNER_TESTS:
        raise EnvironmentError("TRITON_KERNEL_RUNNER_TESTS environment variable is not set")
    """Main function to run all tests."""
    pytest.main()


if __name__ == "__main__":
    # Run the tests using pytest
    print("Starting pytest execution...")
    main()
