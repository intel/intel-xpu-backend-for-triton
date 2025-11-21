
# Design and implementation goals

CLI utility to aid regular Intel Triton backend developer needs in test results analysis.

Non goals:
- TBD

# Installation

### User mode
```
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton/scripts/triton_utils
pip install .
```
### Dev mode
```
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton/scripts/triton_utils
pip install -e .
```

# Usage scenarios / modes:

## 1. Reports dowload from the CI run

Convienientt way to get the local copy of the test reports from from the CI workflow run.

### Basic usage

```
triton-utils download_reports --download-dir <download_dir_path> --gh-run-id <gh_run_id>
```
### Notes

#### Defaults

- Default repo is `intel/intel-xpu-backend-for-triton`. To change the default use `--repo <repo_org/repo_name>` flag.

- Default branch - `main`. To change the default use `--branch <repo_option>`.

#### Advanced options
To get the the full list of the supported options run:
```
triton_utils download_reports --help
```

## 2. Pass rate analysis

Pass rate analysis with the advanced filtering and data merging options.

### Basic usage

```
triton-utils pass_rate --reports <path_to_reports_dir>
```

### Notes

#### Reports directory layouts

CLI supports two reports dir layouts:
  - Single test report. All test report files are located in the root `<reports_dir>` directory.
  - Multiple test reports sets. Reports dir contains multiple subdirs, each of it contains test report files which may overlap by file name and may contain multiple test results for the same test. For test reports from CI workflow runs each subdir represents individual GHA artifact with the test reports.

CLI allows to exclude / include certain subdirs from the analysis scope be either providing exclusion regexp (default is `^(?!)$`)
```
--include-subdir-pattern "<regexp>"
```
or inclusion regexp (default is `^test-report(?!.*lts$).*")`
```
--exclude-subdir-pattern "<regexp>"
```

#### Tests present in multiple test suites

CLI will error out by default if there is at least one test which have multiple test results for different test suites, common scenario - tests which can be run on CPU and GPU (interpreter and language test suites). To fix the issue either allow it by adding the flag
```
--tests-with-multiple-testsuites
```
or exclude one of the tests suites from the analysis
```
--ignore-testsuite <test_suite_name>
```
The second flag can be added multiple times.

#### Status filtering

To filter test results by status add the flag:
```
--status <status_name>
```

This flag can be added multiple times.

Default status filter value is all statuses: `passed`, `skipped`, `failed`, `xfailed`.

#### Merge multiple test results for the same test in the same testsuite

To merge multiple test results for the same test in the same testsuite and take into the consideration only one best outcome add flag:
```
--merge-test-results
```

#### Advanced options
To get the the full list of the supported options run:
```
triton_utils pass_rate --help
```

## 3. Interactive test report analysis

Interactive test report analysis with features based on the actual historical use cases and needs.

### Basic usage

```
triton-utils tests_stats --reports <path_to_reports_dir>
```

This CLI mode supports all filtering and results merging options applicable to `pass_rate` mode and allows to run the analysis on a more granular levels by providing the flag:
```
--report-grouping-level <grouping_level>
```
Default value for this flag is `test`.

#### List test variants (instances) in the skip list format

To list test variants (instances) in the skip list format add flag:
```
--list-test-instances
```
#### List test variants (instances) in the skip list format

To get the summary the failure reasonslist add flag:
```
--list-failure-reasons
```

## 4. Test results export for the advanced analytics

Test results export for Advanced analytics and analysis in Excel / pandas.

### Basic usage

This CLI mode supports all filtering and results merging options applicable to `pass_rate` mode

```
triton-utils export_to --format csv --file-name <csv_file_name> --reports <reports_dir> --tests-with-multiple-testsuites
```


## 4. Test definitions export to the external CI systmes

Test definitions exports to the external CI systems based on the actual tesing data
