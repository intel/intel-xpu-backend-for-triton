import argparse
import os
import re
import sys
import subprocess
from pathlib import Path
import json

import triton_utils

_FA_VARIANT_TO_MODE = {
    "default": "fa_only",
    "fp8_only": "fp8_only",
    "skip_fp8": "skip_fp8",
}

_FA_MODE_ENV = {
    "fa_only": {"HEAD_DIM": "64"},
    "fp8_only": {},
    "skip_fp8": {},
}


def _tutorial06_run_mode(test_name: str) -> str:
    match = re.search(r"test_06_fused_attention\[([^\]]+)\]", test_name)
    if match:
        variant = match.group(1)
        if variant not in _FA_VARIANT_TO_MODE:
            raise ValueError(f"Unknown FA variant '{variant}' in '{test_name}'. "
                             f"Expected one of: {', '.join(_FA_VARIANT_TO_MODE)}")
        return _FA_VARIANT_TO_MODE[variant]
    if "test_06_fused_attention" in test_name:
        return "fa_only"
    return "skip"


def make_select_option(test_name: str) -> list[str]:
    if not test_name:
        return []

    def dedupe_module_dir(nodeid: str) -> str:
        # triton_utils generates nodeid-like paths from pytest.Item.nodeid,
        # in suites it can emit paths with a duplicated module component.
        # turns ".../name/name.py::test" into ".../name.py::test"
        return re.sub(r"(?<=/)([^/]+)/\1(?=\.py::)", r"\1", nodeid)

    select_test_name = dedupe_module_dir(test_name)
    print(f"Select - rewrite: {test_name} to {select_test_name}")

    select_file = os.path.join(os.getcwd(), "select.txt")
    with open(select_file, "w", encoding="utf-8") as f:
        f.write(select_test_name)

    return [
        "--select-from-file",
        str(select_file),
    ]


def main():  # pylint: disable=too-many-locals

    # Future enhancements:
    # - Add more sophisticated check on test results presence - check whether test results for the provided test are present
    # - Option to run individual tests instead of testsuite
    # - Option to force re-run tests
    # - Test name filtering by short name without path
    # - Test name filtering by test variant for run and reporting
    # - Skiplist selection based on platform
    # - Bring back benchmarks support
    # - Option to ignore testsuite exclude list

    script_dir = Path(__file__).parent
    json_path = script_dir / "test_triton_config.json"
    with json_path.open("r", encoding="utf-8") as f:
        test_triton_config = json.load(f)

    testsuites = test_triton_config["include"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=testsuites.keys(),
        help="The test suite to run",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        required=True,
        help="The specific test name to run (optional)",
    )
    parser.add_argument(
        "--home-dir",
        "--h",
        type=str,
        default="",
        required=True,
        help="Home dir path",
    )

    args = parser.parse_args()

    test_name = args.test
    testsuite = args.suite

    reports_dir = os.path.join(args.home_dir, "reports")
    skiplist_dir = os.path.join(os.getcwd(), "scripts/skiplist/default")

    print(f"Suite: {testsuite}")
    print(f"Test Name: {test_name}")

    common_opts = [
        "--skip-pip-install",
        "--skip-pytorch-install",
        "--ignore-errors",
    ]
    reports_opt = [
        "--reports-dir",
        reports_dir,
    ]
    skip_opt = [
        "--skip-list",
        skiplist_dir,
    ]
    test_opt = make_select_option(test_name)

    base_cmd = [
        "bash",
        "-v",
        "-x",
        "scripts/test-triton.sh",
    ]
    run_opt = [f"--{testsuites[testsuite]}"]
    full_cmd = base_cmd + run_opt + common_opts + reports_opt + skip_opt + test_opt

    result_hw_details = subprocess.run(["bash", "-c", "source ./scripts/capture-hw-details.sh"], capture_output=True,
                                       text=True, check=False)
    # Update environment with variables from sourced capture-hw-details.sh
    for line in result_hw_details.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            print(f"Setting env var: {key}={value}")
            os.environ[key] = value
    if result_hw_details.stderr:
        print(result_hw_details.stderr, file=sys.stderr)

    if os.path.isdir(reports_dir):
        print("[WARNING] Triton xpu tests reports have been already run. Test results will be reused")
    else:
        if testsuite in ["flash-attention", "softmax", "gemm", "triton-benchmarks"]:
            raise NotImplementedError("Benchmarks suite is not supported by this script.")

        if testsuite == "tutorials":
            mode = _tutorial06_run_mode(test_name)
            full_cmd += ["--tutorial06-run-mode", mode]
            env = {**os.environ, **_FA_MODE_ENV.get(mode, {})}
        else:
            env = None

        subprocess.run(full_cmd, check=False, stdout=sys.stdout, stderr=sys.stderr, env=env)

    config = triton_utils.Config(
        reports=reports_dir,
        merge_test_results=True,
        suite=testsuite,
        testname_filter=test_name,
        error_on_failures=True,
    )
    summary, ex_code = triton_utils.PassRateActionRunner(config)()
    print(summary)
    sys.exit(ex_code)


if __name__ == "__main__":
    main()
