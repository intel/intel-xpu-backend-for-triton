# triton_utils

CLI utility to aid Intel Triton backend developers with test results analysis. Built-in support for sharded runs.

## Quick Start

```bash
# Install
pip install -e .

# Analyze pass rate from local reports
triton-utils pass_rate --reports ./test-reports

# Download reports from CI and analyze
triton-utils download_reports --download-dir ./reports --gh-run-id 18661995769
triton-utils pass_rate --reports ./reports
```

## Requirements

- Python 3.10+
- `gh` CLI (GitHub CLI) - required for downloading reports from CI

## Installation

### User mode
```bash
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton/scripts/triton_utils
pip install .
```

### Dev mode
```bash
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton/scripts/triton_utils
pip install -e .
```

## Usage

### 1. Download reports from CI

Convenient way to get a local copy of test reports from a CI workflow run.

#### Basic usage

```bash
triton-utils download_reports --download-dir <download_dir_path> --gh-run-id <gh_run_id>
```

#### Example output

```
workflow: 'Build and test'('.github/workflows/build-test.yml')
id: 12345678
repo: 'intel/intel-xpu-backend-for-triton'
branch: 'main'

Artifacts:
Name: test-reports-linux, size: 1234567 bytes
...
```

#### Notes

- Default repo is `intel/intel-xpu-backend-for-triton`. To change use `--repo <repo_org/repo_name>`.
- Default branch is `main`. To change use `--branch <branch_name>`.

For the full list of options run:
```bash
triton-utils download_reports --help
```

### 2. Pass rate analysis

Pass rate analysis with advanced filtering and data merging options.

#### Basic usage

```bash
triton-utils pass_rate --reports <path_to_reports_dir>
```

#### Example output

```json
{
 "passed": 9500,
 "failed": 50,
 "skipped": 200,
 "xfailed": 136,
 "fixme": 0,
 "time": 339.75,
 "total": 9886,
 "pass_rate": 96.1,
 "pass_rate_without_xfailed": 97.44
}
```

#### Notes

##### Reports directory layouts

CLI supports two reports directory layouts:
- **Single test report**: All test report files are located in the root `<reports_dir>` directory.
- **Multiple test report sets**: Reports directory contains multiple subdirs, each containing test report files which may overlap by filename and may contain multiple test results for the same test. For CI workflow runs, each subdir represents an individual GHA artifact.

CLI allows excluding/including certain subdirs from analysis using regexp patterns:
```bash
--include-subdir-pattern "<regexp>"  # default: ^test-report(?!.*lts$).*
--exclude-subdir-pattern "<regexp>"  # default: ^(?!)$ (matches nothing)
```

##### Tests present in multiple test suites

CLI will error out by default if a test has multiple results for different test suites (common scenario: tests that run on both CPU and GPU). To fix:

```bash
# Allow multiple testsuites
--tests-with-multiple-testsuites

# Or exclude one of the test suites (can be added multiple times)
--ignore-testsuite <test_suite_name>
```

##### Status filtering

Filter test results by status (can be added multiple times):
```bash
--status <status_name>
```

Default: all statuses (`passed`, `skipped`, `failed`, `xfailed`).

##### Merge multiple test results

To merge multiple test results for the same test and keep only the best outcome:
```bash
--merge-test-results
```

For the full list of options run:
```bash
triton-utils pass_rate --help
```

### 3. Test report analysis

Detailed test report analysis with grouping options.

#### Basic usage

```bash
triton-utils tests_stats --reports <path_to_reports_dir>
```

This mode supports all filtering and merging options from `pass_rate` mode, plus:

```bash
--report-grouping-level <grouping_level>  # Options: report, testsuite, test (default: test)
```

#### Additional options

```bash
# List test variants in skip list format
--list-test-instances

# Show failure reasons summary
--list-failure-reasons

# Pretty print as table
--pretty-print
```

### 4. Export test results

Export test results for analysis in Excel/pandas.

#### Basic usage

```bash
triton-utils export_to --format csv --file-name <csv_file_name> --reports <reports_dir> --tests-with-multiple-testsuites
```

This mode supports all filtering and merging options from `pass_rate` mode.

### 5. Compare test reports

Compare test results between two test runs.

#### Basic usage

```bash
triton-utils compare --r <reports_folder_1> --r2 <reports_folder_2> --tests-with-multiple-testsuites --level testsuite
```

## Troubleshooting

### `gh auth` not configured

If you see authentication errors when downloading reports:
```bash
gh auth login
gh auth status  # verify authentication
```

### No test reports found

Ensure the reports directory contains `.xml` files in JUnit format. Check:
```bash
ls -la <reports_dir>/**/*.xml
```

### Tests with multiple testsuites error

If you see "Test contains test cases from multiple testsuites", either:
1. Add `--tests-with-multiple-testsuites` flag
2. Exclude one testsuite with `--ignore-testsuite <name>`
