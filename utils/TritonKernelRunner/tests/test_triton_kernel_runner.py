import json
import os
import subprocess
import tempfile
from contextlib import contextmanager

import pytest

from triton.language.target_info import is_xpu_cri

TRITON_KERNEL_RUNNER_PATH = os.getenv("TRITON_KERNEL_RUNNER_PATH")
TRITON_KERNEL_RUNNER_TESTS = os.getenv("TRITON_KERNEL_RUNNER_TESTS")
if not TRITON_KERNEL_RUNNER_TESTS:
    raise EnvironmentError("TRITON_KERNEL_RUNNER_TESTS environment variable is not set")

SPIRV_CLI_ARGS = {os.path.join(TRITON_KERNEL_RUNNER_TESTS, "dot"): ["-o", "tensor_3", "-p", "-v", "expected_output.pt"]}

LARGE_GRF_FLAG = "-cl-intel-512-GRF-per-thread" if is_xpu_cri() else "-cl-intel-256-GRF-per-thread"

requires_runner = pytest.mark.skipif(not os.path.exists(TRITON_KERNEL_RUNNER_PATH),
                                     reason="TritonKernelRunner executable not found")


@contextmanager
def report_subprocess_errors(test_name):
    try:
        yield
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        pytest.fail(f"{test_name} failed unexpectedly: {e}")


def run_kernel_script(script, test_subdir, cache_root):
    target_dir = os.path.join(TRITON_KERNEL_RUNNER_TESTS, test_subdir)
    env = os.environ.copy()
    env["TRITON_XPU_ENABLE_DUMP_KERNEL_ARGS"] = "1"
    env["TRITON_CACHE_DIR"] = cache_root
    result = subprocess.run(["python3", script], capture_output=True, text=True, cwd=target_dir, env=env, check=True)
    print(f"{script} stderr:", result.stderr)


def find_dump_dir(cache_root, binary_name):
    for root, _, files in os.walk(cache_root):
        if "args_data.json" not in files:
            continue
        if load_args_data(root).get("binary_name") == binary_name:
            return root
    return None


def load_args_data(dump_dir):
    with open(os.path.join(dump_dir, "args_data.json")) as args_file:
        return json.load(args_file)


def run_kernel_runner(dump_dir, output_name):
    result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "-d", dump_dir, "-o", output_name], capture_output=True,
                            text=True, check=True)
    print("TritonKernelRunner stderr:", result.stderr)
    assert any(name.startswith("cpp_outs_") and name.endswith(".pt") for name in os.listdir(dump_dir)), \
        f"No TritonKernelRunner output tensor file found in dump dir: {dump_dir}"


@requires_runner
def test_argument_parsing():
    with report_subprocess_errors("test_argument_parsing"):
        result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "--help"], capture_output=True, text=True, check=True)
    assert "USAGE" in result.stdout, "Help message not displayed correctly"


@requires_runner
def test_invalid_argument():
    result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, "--invalid-arg"], capture_output=True, text=True)
    assert result.returncode != 0, "Invalid argument should result in an error"
    assert "unknown command" in result.stderr.lower(), "Error message not displayed for invalid argument"


@requires_runner
def test_args_json_gen():
    with report_subprocess_errors("test_args_json_gen"):
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            run_kernel_script("01-vector-add.py", "add_kernel", cache_root)
            dump_dir = find_dump_dir(cache_root, "add_kernel.spv")
            assert dump_dir is not None, f"args_data.json for add_kernel.spv not found under cache root: {cache_root}"
            run_kernel_runner(dump_dir, "tensor_2")


@requires_runner
@pytest.mark.parametrize("spirv_test_dir", SPIRV_CLI_ARGS.keys())
def test_spirv_execution(spirv_test_dir):
    if not os.path.exists(spirv_test_dir):
        pytest.skip(f"Test SPIR-V directory {spirv_test_dir} not found")
    cli_args = SPIRV_CLI_ARGS[spirv_test_dir]
    result = subprocess.run([TRITON_KERNEL_RUNNER_PATH, *cli_args], capture_output=True, text=True, check=True,
                            cwd=spirv_test_dir)
    print("TritonKernelRunner output:", result.stdout)


@requires_runner
def test_dumped_build_flags_reflect_auto_grf_retry():
    """Repro dumps must capture the GRF mode Triton actually ran with, including driver.c's
    runtime auto-large-GRF retry -- otherwise replay builds an un-escalated binary (issue #5529)."""
    with report_subprocess_errors("test_dumped_build_flags_reflect_auto_grf_retry"):
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            run_kernel_script("01-auto-grf.py", "auto_grf", cache_root)
            dump_dir = find_dump_dir(cache_root, "auto_grf_kernel.spv")
            assert dump_dir is not None, \
                f"args_data.json for auto_grf_kernel.spv not found under cache root: {cache_root}"

            build_flags = load_args_data(dump_dir)["build_flags"]
            assert LARGE_GRF_FLAG in build_flags, (f"dumped build_flags do not reflect the auto-large-GRF retry "
                                                   f"(expected {LARGE_GRF_FLAG!r} in {build_flags!r})")

            run_kernel_runner(dump_dir, "tensor_0")


@requires_runner
def test_native_binary_dump_and_execution():
    """A native-binary (AOT/zebin) dump must record `is_native_binary` and replay with
    `ZE_MODULE_FORMAT_NATIVE`, not `_IL_SPIRV` -- the native-binary leg of issue #5529."""
    with report_subprocess_errors("test_native_binary_dump_and_execution"):
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            run_kernel_script("01-native-binary.py", "native_binary", cache_root)
            dump_dir = find_dump_dir(cache_root, "native_binary_kernel.zebin")
            assert dump_dir is not None, \
                f"args_data.json for native_binary_kernel.zebin not found under cache root: {cache_root}"

            args_data = load_args_data(dump_dir)
            assert args_data.get("is_native_binary") is True, \
                f"dump does not record the kernel as a native binary: {args_data!r}"
            assert os.path.exists(os.path.join(dump_dir, "native_binary_kernel.zebin")), \
                f"dumped zebin file not found in {dump_dir}"

            run_kernel_runner(dump_dir, "tensor_2")


@requires_runner
def test_native_binary_dump_reflects_auto_grf_retry():
    """A native binary that spills must dump/replay at its escalated GRF mode -- the native-binary
    analogue of `test_dumped_build_flags_reflect_auto_grf_retry`, exercising make_zebin's own retry."""
    with report_subprocess_errors("test_native_binary_dump_reflects_auto_grf_retry"):
        with tempfile.TemporaryDirectory(prefix="triton_kernel_runner_cache_") as cache_root:
            run_kernel_script("01-native-binary-auto-grf.py", "native_binary_auto_grf", cache_root)
            dump_dir = find_dump_dir(cache_root, "native_binary_auto_grf_kernel.zebin")
            assert dump_dir is not None, (f"args_data.json for native_binary_auto_grf_kernel.zebin not found "
                                          f"under cache root: {cache_root}")

            args_data = load_args_data(dump_dir)
            assert args_data.get("is_native_binary") is True, \
                f"dump does not record the kernel as a native binary: {args_data!r}"

            build_flags = args_data["build_flags"]
            assert LARGE_GRF_FLAG in build_flags, (
                f"dumped build_flags do not reflect make_zebin's auto-large-GRF retry "
                f"(expected {LARGE_GRF_FLAG!r} in {build_flags!r})")

            run_kernel_runner(dump_dir, "tensor_0")
