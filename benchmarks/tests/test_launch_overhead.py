import io
import json

import pandas as pd

from benchmark_helpers import make_cfg, template_for
from triton_kernels_benchmark import launch_overhead_benchmark
from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser
from triton_kernels_benchmark.benchmark_utils import BenchmarkConfigs

# Raw results in the format written by Mark.run.
LAUNCH_OVERHEAD_CSV = """
arg_type,num_args,level,triton-time_us,triton-time_us-min,triton-time_us-max,triton-ns_per_arg,triton-ns_per_arg-min,triton-ns_per_arg-max,triton-cpu_mhz,triton-cpu_mhz-min,triton-cpu_mhz-max,triton-CV,datetime,run_counter
none,0,jit,8.10,8.02,8.31,,,,3700.0,,,0.008,2026-10-02 10:00:00.000000,1
ptr,4,jit,13.20,13.05,13.52,1275.0,,,3700.0,,,0.006,2026-10-02 10:00:00.000000,1
td,1,launcher,5.90,5.81,6.02,2310.0,,,3699.0,,,0.005,2026-10-02 10:00:00.000000,1
"""
FIRST_LAUNCH_CSV = """
cache,phase,triton-time_ms,triton-time_ms-min,triton-time_ms-max,triton-CV,datetime,run_counter
cold,total,9050.0,8990.0,9310.0,0.015,2026-10-02 10:00:00.000000,1
cold,driver,7950.0,7900.0,8200.0,0.017,2026-10-02 10:00:00.000000,1
warm,total,960.0,930.0,990.0,0.020,2026-10-02 10:00:00.000000,1
"""


def _run_with_results(key: str, csv: str, tmp_path, monkeypatch) -> pd.DataFrame:
    """Runs the config `key` on recorded raw results and returns its long report."""
    monkeypatch.setenv("GPU_DEVICE", "Intel(R) Data Center GPU Max 1100")
    configs = BenchmarkConfigs.from_args(["run", key, "--reports", str(tmp_path), "--show-details"])
    results = pd.read_csv(io.StringIO(csv))
    for config in configs.configs:
        config.res_df_list = [results]
        results.to_csv(tmp_path / f"{config.plot_name}.csv", index=False)
    configs.run()
    report = pd.read_csv(tmp_path / f"{template_for(key).report_name}-report.csv")
    assert set(report["benchmark_group"]) == {"overhead"}
    assert set(report["compiler"]) == {"triton"}
    return report


def test_empty_kernel_signature():
    kernel = launch_overhead_benchmark._empty_kernel("launch_overhead_test", 3)  # pylint: disable=protected-access
    assert kernel.arg_names == ["a0", "a1", "a2"]


def test_shape_pattern_selects_sweep_points():
    shapes = [str(shape) for shape in make_cfg(template_for("launch-overhead")).supported_shapes]
    expected = ["[ptr-4-jit]", "[ptr-16-jit]", "[ptr-64-jit]"]
    assert ShapePatternParser("[ptr-*-jit]").filter_by_pattern(shapes) == expected


def test_shape_pattern_selects_first_launch_phases():
    shapes = [str(shape) for shape in make_cfg(template_for("launch-overhead-cold")).supported_shapes]
    assert len(shapes) == 10
    assert ShapePatternParser("[*-total]").filter_by_pattern(shapes) == ["[cold-total]", "[warm-total]"]


def test_sweep_summary_and_long_report(capsys, tmp_path, monkeypatch):
    report = _run_with_results("launch-overhead", LAUNCH_OVERHEAD_CSV, tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert "time_us" in output and "ns_per_arg" in output
    assert set(report["benchmark"]) == {"launch-overhead-sweep"}
    assert sorted(report["value_name"].unique()) == ["cpu_mhz", "ns_per_arg", "time_us"]
    # The empty kernel has no per-argument cost.
    assert len(report[report["value_name"] == "ns_per_arg"]) == 2
    params = [json.loads(p) for p in report["params"]]
    assert {"arg_type": "ptr", "num_args": 4, "level": "jit", "cpu": launch_overhead_benchmark.cpu_model()} in params


def test_first_launch_summary_and_long_report(capsys, tmp_path, monkeypatch):
    report = _run_with_results("launch-overhead-cold", FIRST_LAUNCH_CSV, tmp_path, monkeypatch)
    assert "time_ms" in capsys.readouterr().out
    assert set(report["benchmark"]) == {"launch-overhead-cold"}
    assert set(report["value_name"]) == {"time_ms"}
    params = [json.loads(p) for p in report["params"]]
    assert {"cache": "warm", "phase": "total", "cpu": launch_overhead_benchmark.cpu_model()} in params


def test_shared_measurement_measures_again_in_the_next_run():
    calls = []
    measurements = launch_overhead_benchmark._SharedMeasurement(  # pylint: disable=protected-access
        lambda arg_type: calls.append(arg_type) or {"run": len(calls)})
    assert measurements.get(("ptr", ), "jit") == {"run": 1}
    assert measurements.get(("ptr", ), "launcher") == {"run": 1}
    assert measurements.get(("ptr", ), "jit") == {"run": 2}
    assert calls == ["ptr", "ptr"]
