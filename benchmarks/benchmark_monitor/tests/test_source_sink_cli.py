import csv
import io
import json
import socket
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

from benchmark_monitor import SeriesKey, Status, Thresholds, Verdict, load_catalog
from benchmark_monitor.__main__ import main
from benchmark_monitor.report import Report
from benchmark_monitor.sink import JsonSink, MarkdownSink, _by_case
from benchmark_monitor.source import ArtifactSource, CsvReportSource

HOST = {"gpu_device": "Intel(R) Arc(TM) B580 Graphics", "torch_version": "2.9", "agama_version": ""}
D1, D2 = date(2026, 11, 1), date(2026, 11, 20)


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def wide(**kw):
    return {
        "run_uuid": "r1", "datetime": "2026-11-10T03:00:00", "benchmark": "softmax", "compiler": "triton", "tflops":
        "5.0", "hbm_gbs": "100", "params": '{"N": 1, "M": 2}', "tag": "client", **HOST, **kw
    }


def long(**kw):
    return {
        "run_uuid": "r2", "ts": "2026-11-10T03:00:00", "benchmark_group": "vllm", "benchmark": "moe", "compiler":
        "triton", "value_name": "tflops", "value": "7.5", "params": '{"b": 1}', "tag": "ci", **HOST, **kw
    }


@pytest.fixture(name="cat")
def _cat():
    return load_catalog()


def scope(cat, name):
    return cat.scopes[name]


def test_wide_melts_metrics_and_attrs(tmp_path, cat):
    p = write(tmp_path / "a-report.csv", [wide()])
    obs = list(CsvReportSource([p], cat).observations(scope(cat, "nightly-b580-micro"), D1, D2))
    assert {o.key.metric: o.value for o in obs} == {"tflops": 5.0, "gbps": 100.0}
    k = obs[0].key
    assert (k.source, k.platform, k.tag, k.params) == ("micro", "b580", "client", '{"M":2,"N":1}')
    assert obs[0].attrs == {"gpu_device": HOST["gpu_device"], "torch_version": "2.9"}
    assert obs[0].observed_at == datetime(2026, 11, 10, 3)


def test_long_format_and_group_filter(tmp_path, cat):
    p = write(tmp_path / "b-report.csv", [long(), long(benchmark_group="other")])
    sc = scope(cat, "nightly-b580-vllm")
    assert len(list(CsvReportSource([p], cat).observations(sc, D1, D2))) == 2
    narrow = type(sc)(name="n", source="third_party", platform="b580", tags=("ci", ), groups=("vllm", ))
    obs = list(CsvReportSource([p], cat).observations(narrow, D1, D2))
    assert [o.key.benchmark for o in obs] == ["moe"] and obs[0].key.source == "third_party"


def test_skips_and_filters(tmp_path, cat):
    rows = [
        wide(tflops="0", hbm_gbs=""),  # no positive value
        wide(datetime=""),
        wide(datetime="0000-00-00 00:00:00"),
        wide(datetime="2026-12-30T00:00:00"),  # outside window
        wide(tag="pr-1"),  # tag
        wide(gpu_device="Intel Max 1550"),  # platform
        wide(hbm_gbs="nan"),  # keeps tflops only
    ]
    p = write(tmp_path / "c-report.csv", rows)
    obs = list(CsvReportSource([p], cat).observations(scope(cat, "nightly-b580-micro"), D1, D2))
    assert [o.key.metric for o in obs] == ["tflops"]


def test_source_kind_mismatch_and_artifact_scope(tmp_path, cat):
    p = write(tmp_path / "d-report.csv", [long()])
    assert not list(CsvReportSource([p], cat).observations(scope(cat, "nightly-b580-micro"), D1, D2))
    art = type(scope(cat, "nightly-b580-vllm"))(name="a", source="artifact", platform="b580", tags=("ci", ))
    assert len(list(CsvReportSource([p], cat).observations(art, D1, D2))) == 1


def test_artifact_source_recurses(tmp_path, cat):
    write(tmp_path / "x" / "y" / "m-report.csv", [wide()])
    write(tmp_path / "z" / "ignored.csv", [wide()])
    obs = list(ArtifactSource(tmp_path, cat).observations(scope(cat, "nightly-b580-micro"), D1, D2))
    assert len(obs) == 2


def _report():
    key = SeriesKey("micro", "b580", "client", "softmax", "triton", "tflops", "{}")
    th = Thresholds(5.0, 3.0, 8, 20, 15.0, 0.05, True)
    return Report(date(2026, 11, 16), (Verdict(key, D1, Status.REGRESSION, 12.34, -4.5, 10.0, 0.2, 9,
                                               th), Verdict(key, D1, Status.OK, None, None, None, None, 9, th)),
                  warnings=("careful", ))


def test_sinks(tmp_path):
    out = io.StringIO()
    MarkdownSink(out).emit(_report())
    text = out.getvalue()
    assert text.index("softmax") < text.index("## Counts")
    assert "| 12.3 | -4.50 | 10 | 9 |" in text and "- regression: 1" in text and "- ok: 1" in text and "careful" in text
    path = tmp_path / "r.json"
    JsonSink(path).emit(_report())
    data = json.loads(path.read_text())
    assert data["as_of"] == "2026-11-16" and data["counts"] == {"regression": 1, "ok": 1}
    assert data["verdicts"][0]["status"] == "regression"


def test_cli_end_to_end_with_real_gate(tmp_path):
    for day in range(1, 13):  # 11 flat nights, then a 20% drop on the 12th
        value = "8.0" if day == 12 else "10.0"
        write(tmp_path / "art" / f"n{day}-report.csv",
              [wide(run_uuid=f"r{day}", datetime=f"2026-11-{day:02d}T03:00:00", tflops=value, hbm_gbs="")])
    out = tmp_path / "out.json"
    argv = [
        "--catalog",
        str(Path(__file__).parent.parent / "benchmark_monitor/catalog/default.yaml"), "--artifacts",
        str(tmp_path / "art"), "--scope", "nightly-b580-micro", "--as-of", "2026-11-12", "--format", "json", "--out",
        str(out)
    ]
    assert main(argv) == 0
    data = json.loads(out.read_text())
    assert data["as_of"] == "2026-11-12" and data["counts"] == {"regression": 1}
    assert data["verdicts"][0]["key"]["benchmark"] == "softmax" and data["verdicts"][0]["n_history"] == 11


def test_cli_errors_exit_2(tmp_path, capsys):
    base = ["--artifacts", str(tmp_path), "--scope", "nightly-b580-micro"]
    assert main(base + ["--catalog", str(tmp_path / "missing.yaml")]) == 2  # unreadable catalog
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nbogus: 1\n")
    assert main(base + ["--catalog", str(bad)]) == 2  # invalid catalog
    assert main(["--artifacts", str(tmp_path / "nope"), "--scope", "nightly-b580-micro"]) == 2  # not a directory
    err = capsys.readouterr().err
    assert err.count("error:") == 2 and "not a directory" in err


def test_cli_default_catalog_and_undeclared_metric_warning(tmp_path, capsys):
    write(tmp_path / "art" / "v-report.csv", [long(value_name="furlongs", tag="ci"), long(value_name="cpu_mhz")])
    argv = ["--artifacts", str(tmp_path / "art"), "--scope", "nightly-b580-vllm", "--as-of", "2026-11-12"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("furlongs") == 1 and "cpu_mhz" not in out  # one warning; declared-ungated is silent


def test_artifact_scope_keeps_long_and_wide_series_apart(tmp_path, cat):
    write(tmp_path / "w-report.csv", [wide()])
    write(tmp_path / "l-report.csv", [long(benchmark="softmax", value_name="tflops")])
    art = type(scope(cat, "nightly-b580-vllm"))(name="a", source="artifact", platform="b580", tags=("ci", "client"))
    obs = list(ArtifactSource(tmp_path, cat).observations(art, D1, D2))
    assert {o.key.source for o in obs} == {"micro", "third_party"}
    assert len({o.key for o in obs if o.key.metric == "tflops"}) == 2


def test_non_finite_values_are_dropped(tmp_path, cat):
    p = write(tmp_path / "f-report.csv", [wide(tflops="inf", hbm_gbs="1e999"), wide(tflops="5", hbm_gbs="-inf")])
    obs = list(CsvReportSource([p], cat).observations(scope(cat, "nightly-b580-micro"), D1, D2))
    assert [(o.key.metric, o.value) for o in obs] == [("tflops", 5.0)]


def test_cli_unknown_scope(tmp_path, capsys):
    assert main([
        "--catalog",
        str(Path(__file__).parent.parent / "benchmark_monitor/catalog/default.yaml"), "--artifacts",
        str(tmp_path), "--scope", "nope"
    ]) == 2
    assert "unknown scope" in capsys.readouterr().err


def test_import_without_network():
    code = ("import socket\n"
            "def boom(*a, **k): raise RuntimeError('network')\n"
            "socket.socket = boom\n"
            "import benchmark_monitor, benchmark_monitor.source, benchmark_monitor.sink, benchmark_monitor.__main__\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)


def test_socket_blocked_in_process(tmp_path, cat, monkeypatch):

    def boom(*_a, **_k):
        raise RuntimeError("network")

    monkeypatch.setattr(socket, "socket", boom)
    p = write(tmp_path / "e-report.csv", [wide()])
    assert list(CsvReportSource([p], cat).observations(scope(cat, "nightly-b580-micro"), D1, D2))


def _suppression_report(n_new=12):
    from benchmark_monitor.suppress import Confirmed, StormDigest, SuppressionResult  # pylint: disable=import-outside-toplevel
    th = Thresholds(5.0, 3.0, 8, 20, 15.0, 0.05, True)

    def reg(i):
        key = SeriesKey("micro", "b580", "client", f"gemm-{i}", "triton", "tflops", "{}")
        return Verdict(key, D1, Status.REGRESSION, 10.0 + i, -5.0, 10.0, 0.1, 9, th)

    new = tuple(reg(i) for i in range(n_new))
    sup = SuppressionResult(new=new, confirmed=(Confirmed(new[0].key, new[0], 2, D1), ), sustained=(reg(40), ),
                            muted=(reg(41), ), storms=(StormDigest("flash-attn", 11, 9.5, 20.0), ),
                            storm_nights=frozenset({D1}), suspect={"micro": "partial run"})
    return Report(date(2026, 11, 16), new, sup, ("careful", ))


def test_sinks_render_suppression_sections_short():
    out = io.StringIO()
    MarkdownSink(out).emit(_suppression_report())
    text = out.getvalue()
    for title in ("## New (12)", "## Confirmed (1)", "## Sustained (1)", "## Muted (flip-flop) (1)", "## Storms (1)",
                  "## Warnings"):
        assert title in text
    assert "- ... and 2 more" in text and "flash-attn: 11 new" in text and "2 nights since 2026-11-01" in text


def test_json_sink_serialises_suppression(tmp_path):
    path = tmp_path / "s.json"
    JsonSink(path).emit(_suppression_report(2))
    data = json.loads(path.read_text())
    assert data["suppression"]["storm_nights"] == ["2026-11-01"] and len(data["suppression"]["new"]) == 2
    assert data["suppression"]["confirmed"][0]["streak"] == 2


def test_cli_rejects_nights_below_one(tmp_path, capsys):
    assert main(["--artifacts", str(tmp_path), "--scope", "nightly-b580-micro", "--nights", "0"]) == 2
    assert "--nights" in capsys.readouterr().err


def _metric_report():
    from benchmark_monitor.suppress import Confirmed, SuppressionResult  # pylint: disable=import-outside-toplevel
    th = Thresholds(5.0, 3.0, 8, 20, 15.0, 0.05, True)

    def reg(bench, metric, drop, params="{}"):
        key = SeriesKey("micro", "b580", "client", bench, "triton", metric, params)
        return Verdict(key, D1, Status.REGRESSION, drop, -5.0, 10.0, 0.1, 9, th)

    new = (reg("b-attn", "gbps", 9.0), reg("a-gemm", "tflops", 12.0), reg("b-attn", "tflops", 9.1),
           reg("a-gemm", "gbps", 12.0), reg("c-add", "gbps", 9.1, '{"n":1}'), reg("c-add", "gbps", 9.1, '{"n":0}'))
    sup = SuppressionResult(new=new, confirmed=tuple(Confirmed(v.key, v, 2, D1) for v in new[:3]), sustained=(),
                            muted=(), storms=(), storm_nights=frozenset(), suspect={})
    return Report(date(2026, 11, 16), new, sup, ())


def test_sink_lists_each_case_once_with_all_its_metrics():
    out = io.StringIO()
    MarkdownSink(out).emit(_metric_report())
    text = out.getvalue()
    assert "4 regressing cases tonight (6 series); 4 new." in text
    assert "## New (4, 6 series)" in text
    assert "- a-gemm / b580 `{}`: gbps -12.0%, tflops -12.0%" in text  # tie on drop: metric name order
    assert "- b-attn / b580 `{}`: tflops -9.1%, gbps -9.0%" in text
    assert text.count("a-gemm / b580") == 2  # New and Confirmed only
    assert "## Confirmed (2, 3 series)" in text
    assert "b-attn / b580 `{}`: gbps 2 nights since 2026-11-01, tflops 2 nights since 2026-11-01" in text


def test_case_order_is_deterministic():
    vs = list(_metric_report().suppression.new)
    first = [label for label, _ in _by_case(vs)]
    assert first == [label for label, _ in _by_case(reversed(vs))]
    # drop desc, then benchmark, then params
    assert first == ["a-gemm / b580 `{}`", "b-attn / b580 `{}`", "c-add / b580 `{\"n\":0}`", "c-add / b580 `{\"n\":1}`"]
