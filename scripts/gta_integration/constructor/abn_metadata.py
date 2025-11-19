import argparse
from pathlib import Path

import json

import triton_utils


def normalize_key(raw_key: str) -> str:
    return raw_key.replace("::", "__").replace("/", "_").replace(".", "___")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        "--r",
        type=str,
        required=False,
        default="reports-nightly",
        help="Path to the reports folder",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    json_path = script_dir / "test_triton_config.json"
    with json_path.open("r", encoding="utf-8") as f:
        test_triton_config = json.load(f)

    ignore_testsuites = list(test_triton_config["exclude"].keys())

    config = triton_utils.Config(
        status_filter=["passed"],
        ignore_testsuite_filter=ignore_testsuites,
        _download_dir=args.reports,
        reports=args.reports,
        tests_with_multiple_testsuites=True,
        merge_test_results=True,
        file_name="abn_metadata.json",
        latest_nightly_gh_run=True,
        _report_grouping_level="testsuite",
    )
    triton_utils.DownloadReportsActionRunner(config)()
    report = triton_utils.PassRateActionRunner(config)
    print(report.summary)
    print(report.summary_detailed)
    tests = list(report.tests)
    json_data = {}
    # Remove this after the option to run tutorials as pytest tests will be added
    tests.append(triton_utils.Test(testsuite="tutorials", testname="all_tutorials"))
    for test in tests:
        json_data[normalize_key(f"{test.testsuite}__{test.testname}")] = {
            "api": "L0",
            "commandLine": f"run_triton_tests.sh --suite {test.testsuite} --test {test.testname}",
            "validReturnCodes": [0],
            "passRatesFunctionalCheck": True,
            "passRatesRegex": {
                "blockCount": "",
                "errorCount": "",
                "failCount": 'failed": (\\d+)',
                "passCount": 'passed": (\\d+)',
                "skipCount": "",
                "totalCount": "",
            },
        }
    with open(config.file_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
