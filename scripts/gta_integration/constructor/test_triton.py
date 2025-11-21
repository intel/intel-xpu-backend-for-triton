import argparse
import os
import sys
import subprocess
from pathlib import Path
import json

import triton_utils


def main():  # pylint: disable=too-many-locals

    # Future enhancements:
    # - Add more sophisticated check on test results presence - check whether test results for the provided test are present
    # - Option to run individual tests instead of testsuite
    # - Option to force re-run tests
    # - Test name filtering by short name without path
    # - Test name filtering by test variant for run and reporting
    # - Skiplist selection based on platform
    # - Bring back tutorials support
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
    test_opt = []
    # if test_name:
    #     select_file = os.path.join(triton_proj, "select.txt")
    #     with open(select_file, "w") as f:
    #         f.write(test_name)
    #     test_opt = [
    #         "--select-from-file",
    #         select_file,
    #     ]

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
        if testsuite == "tutorials":
            raise NotImplementedError("Tutorials suite is not supported by this script.")
        if testsuite in ["flash-attention", "softmax", "gemm", "triton-benchmarks"]:
            raise NotImplementedError("Benchmarks suite is not supported by this script.")
        subprocess.run(full_cmd, check=False, stdout=sys.stdout, stderr=sys.stderr)

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
