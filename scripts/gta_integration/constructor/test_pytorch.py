import os
import re
import shutil
import subprocess
import sys

import triton_utils


def main():
    test_names = sys.argv[1:]

    print(f"Test Names: {test_names}")

    reports_dir = os.path.join(os.getcwd(), "test-reports")

    shutil.rmtree(reports_dir, ignore_errors=True)

    subprocess.run([
        "python",
        "test/run_test.py",
        "--save-xml",
        reports_dir,
        "--keep-going",
        "--include",
    ] + test_names, check=False, stdout=sys.stdout, stderr=sys.stderr)

    config = triton_utils.Config(
        reports=reports_dir,
        merge_test_results=True,
        tests_with_multiple_testsuites=True,
        error_on_failures=True,
        include_subdir_patterns=[re.compile(r"^.*$")],
    )

    # pylint: disable=duplicate-code
    # Duplicated entry-point pattern is intentional, this file is distributed
    # independently when packed for CI and cannot share code with test_triton.py.
    summary, ex_code = triton_utils.PassRateActionRunner(config)()
    print(summary)
    sys.exit(ex_code)
    # pylint: enable=duplicate-code


if __name__ == "__main__":
    main()
