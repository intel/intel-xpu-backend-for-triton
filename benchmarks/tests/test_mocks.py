from typing import List, Optional

import re
import io

import pytest

import pandas as pd

from triton_kernels_benchmark.benchmark_utils import BenchmarkCategory, BenchmarkConfigs

ALL_CATEGORIES = {cat.value for cat in BenchmarkCategory}

SOFTMAX_PERFORMANCE_CSV = """
N,Triton-GB/s,XeTLA-GB/s,Triton-GB/s-min,XeTLA-GB/s-min,Triton-GB/s-max,XeTLA-GB/s-max,Triton-TFlops,XeTLA-TFlops,Triton-TFlops-min,XeTLA-TFlops-min,Triton-TFlops-max,XeTLA-TFlops-max,Triton-CV,XeTLA-CV,datetime,run_counter
256.000000,473.397771,568.333815,90.083848,514.007860,494.611303,582.542232,0.473398,0.568334,0.090084,0.514008,0.494611,0.582542,0.019154,0.018093,2025-05-05 21:45:29.943213,1
1024.000000,683.111432,541.549931,672.164101,537.731297,689.852609,548.992673,0.683111,0.541550,0.672164,0.537731,0.689853,0.548993,0.006031,0.004731,2025-05-05 21:45:29.943213,1
2048.000000,677.320009,726.915809,672.164101,708.497308,683.111380,825.650389,0.677320,0.726916,0.672164,0.708497,0.683111,0.825650,0.003426,0.018620,2025-05-05 21:45:29.943213,1
4096.000000,627.302921,477.032066,616.809404,474.468764,641.330889,488.846612,0.627303,0.477032,0.616809,0.474469,0.641331,0.488847,0.008189,0.003735,2025-05-05 21:45:29.943213,1
8192.000000,679.033333,611.916382,665.234595,604.802311,762.600731,637.916958,0.679033,0.611916,0.665235,0.604802,0.762601,0.637917,0.016350,0.008740,2025-05-05 21:45:29.943213,1
16384.000000,712.219329,677.833087,703.447226,661.562161,760.871449,688.437266,0.712219,0.677833,0.703447,0.661562,0.760871,0.688437,0.009147,0.009317,2025-05-05 21:45:29.943213,1
32768.000000,733.450281,729.424324,727.861837,726.286411,756.582488,737.136026,0.733450,0.729424,0.727862,0.726286,0.756582,0.737136,0.003869,0.002001,2025-05-05 21:45:29.943213,1
"""

PERFORMANCE_CSVS = {
    "softmax": SOFTMAX_PERFORMANCE_CSV,
}


@pytest.fixture(autouse=True)
def capture_hw_details_env(monkeypatch):
    monkeypatch.setenv("GPU_DEVICE", "Intel(R) Data Center GPU Max 1100")
    yield


@pytest.mark.parametrize("command", ["run"])
@pytest.mark.parametrize(
    "benchmark",
    [
        ["softmax"],
        ["softmax", "--shape-pattern", "[256]"],
        ["softmax", "--shape-pattern", "[*]"],
    ],
)
@pytest.mark.parametrize("provider", ["triton", None])
@pytest.mark.parametrize("n_runs", [1, 2, 3])
@pytest.mark.parametrize("show_details", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("reports", [False, True])
def test_benchmark_run_monkeypatched(
    command: str,
    benchmark: List[str],
    provider: Optional[str],
    n_runs: int,
    show_details: bool,
    json_output: bool,
    reports: bool,
    capsys,
    tmp_path,
):
    args = [command] + benchmark
    if provider:
        args.extend(["--provider", provider])
    if n_runs > 1:
        args.extend(["--n_runs", str(n_runs)])
    if show_details:
        args.extend(["--show-details"])
    if json_output:
        args.extend(["--json"])
    if reports:
        args.extend(["--reports", str(tmp_path)])

    configs = BenchmarkConfigs.from_args(args)
    for config in configs.configs:
        config.res_df_list = [pd.read_csv(io.StringIO(PERFORMANCE_CSVS[config.key]))] * n_runs
    configs.run()

    captured_output = capsys.readouterr().out
    output_lines = captured_output.splitlines()
    if provider and not json_output:
        assert "Selected providers: {'triton': 'Triton'}" in output_lines
    # Check if the prettified result table have CV column, example - "metric     GB/s   GB/s TFlops TFlops     CV    CV"
    if show_details and not json_output:
        assert not show_details or re.search(r"^metric.* CV", captured_output, flags=re.MULTILINE)
