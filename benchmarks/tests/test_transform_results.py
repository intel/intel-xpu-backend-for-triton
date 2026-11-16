import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transform_results

# Raw results in the format written by Mark.run for a two-provider (load vs descriptor) benchmark.
DESCRIPTOR_CSV = """
N,load-GB/s,load-GB/s-min,load-GB/s-max,descriptor-GB/s,descriptor-GB/s-min,descriptor-GB/s-max,load-CV,descriptor-CV,datetime,run_counter
1024,100.5,99.0,101.0,110.5,109.0,111.0,0.01,0.02,2026-10-02 10:00:00.000000,1
4096,200.5,199.0,201.0,210.5,209.0,211.0,0.01,0.02,2026-10-02 10:00:00.000000,1
"""
# Three providers and a TFlops metric.
ROW_LIMIT_CSV = """
G,N,small_groups,load_masked-TFlops,load_unmasked-TFlops,descriptor-TFlops,datetime,run_counter
8,512,0,1.5,2.5,3.5,2026-10-02 10:00:00.000000,1
"""


def _write(tmp_path, text, name="raw.csv"):
    path = tmp_path / name
    path.write_text(text.lstrip())
    return str(path)


def _host_env(monkeypatch):
    monkeypatch.setenv("GPU_DEVICE", "Intel(R) Graphics [0xe20b]")
    monkeypatch.setenv("AGAMA_VERSION", "1234")


def test_infer_param_cols_skips_metrics_and_bookkeeping(tmp_path):
    assert transform_results.infer_param_cols(_write(tmp_path, DESCRIPTOR_CSV)) == ["N"]
    assert transform_results.infer_param_cols(_write(tmp_path, ROW_LIMIT_CSV)) == ["G", "N", "small_groups"]


def test_descriptor_report_has_one_row_per_provider_and_shape(tmp_path, monkeypatch):
    _host_env(monkeypatch)
    path = _write(tmp_path, DESCRIPTOR_CSV)
    df = transform_results.parse_csv(path, "pr-7247", "descriptor", "copy-1d", transform_results.infer_param_cols(path))
    assert len(df) == 4
    assert set(df["compiler"]) == {"load", "descriptor"}
    assert set(df["value_name"]) == {"gbps"}
    assert set(df["benchmark_group"]) == {"descriptor"}
    assert set(df["benchmark"]) == {"copy-1d"}
    assert set(df["tag"]) == {"pr-7247"}
    assert set(df["agama_version"]) == {"1234"}
    descriptor = df[df["compiler"] == "descriptor"].sort_values("value")
    assert list(descriptor["value"]) == [110.5, 210.5]
    assert [json.loads(p) for p in descriptor["params"]] == [{"N": 1024}, {"N": 4096}]


def test_row_limit_report_keeps_three_providers(tmp_path, monkeypatch):
    _host_env(monkeypatch)
    path = _write(tmp_path, ROW_LIMIT_CSV)
    df = transform_results.parse_csv(path, "ci", "descriptor", "grouped-matmul-row-limit",
                                     transform_results.infer_param_cols(path))
    assert dict(zip(df["compiler"], df["value"])) == {"load_masked": 1.5, "load_unmasked": 2.5, "descriptor": 3.5}
    assert set(df["value_name"]) == {"tflops"}


def test_existing_compilers_still_match(tmp_path, monkeypatch):
    _host_env(monkeypatch)
    path = _write(
        tmp_path, """
input_type,triton-time_us,pytorch-time_us,datetime,run_counter
fp32,5.0,6.0,2026-10-02 10:00:00.000000,1
""")
    df = transform_results.parse_csv(path, "", "overhead", "launch-overhead", ["input_type"])
    assert dict(zip(df["compiler"], df["value"])) == {"triton": 5.0, "pytorch": 6.0}
