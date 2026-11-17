import csv
import io
import json
import socket
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from benchmark_monitor import SeriesKey, Status, Thresholds, Verdict, load_catalog
from benchmark_monitor.__main__ import main
from benchmark_monitor.report import Report
from benchmark_monitor.sink import JsonSink, MarkdownSink
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


def test_markdown_table_identifies_series_and_escapes_cells():
    key = SeriesKey("third_party", "b580", "ci|x", "moe", "triton-td", "tflops", '{"label":"a|b`c\nd"}')
    th = Thresholds(5.0, 3.0, 8, 20, 15.0, 0.05, True)
    rep = Report(date(2026, 11, 16), (Verdict(key, D1, Status.REGRESSION, 12.34, -4.5, 10.0, 0.2, 9, th), ))
    text = MarkdownSink(io.StringIO()).render(rep)
    header, _, row = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert "compiler" in header and "source" in header and "tag" in header
    assert "triton-td" in row and "third_party" in row
    cells = [c for c in row.replace("\\|", "\0").split("|")[1:-1]]
    assert len(cells) == header.count("|") - 1  # escaped pipe did not add a column
    assert "`" not in row and "\n" not in row and "a\\|b" in row


def test_history_start_default_is_three_windows_and_configurable():
    from benchmark_monitor.__main__ import history_start
    assert history_start(date(2026, 11, 30), None, 20) == date(2026, 10, 1)
    assert history_start(date(2026, 11, 30), 5, 20) == date(2026, 11, 25)
    assert history_start(date(2026, 11, 30), 0, 20) == date.min


def test_cli_gappy_series_keeps_enough_points_for_the_window(tmp_path):
    # One run every 3rd night: the last 20 *calendar* days hold only ~6 points (< min_history 8).
    base = date(2026, 11, 1)
    offsets = list(range(0, 33, 3)) + [33]  # 11 flat runs, then a 20% drop
    for off in offsets:
        night = base + timedelta(days=off)
        write(tmp_path / "art" / f"n{off}-report.csv",
              [wide(run_uuid=f"r{off}", datetime=f"{night}T03:00:00", tflops="8.0" if off == 33 else "10.0",
                    hbm_gbs="")])
    out = tmp_path / "out.json"
    argv = ["--artifacts", str(tmp_path / "art"), "--scope", "nightly-b580-micro", "--as-of", str(base + timedelta(days=33)),
            "--format", "json", "--out", str(out)]
    assert main(argv) == 0
    data = json.loads(out.read_text())
    assert data["counts"] == {"regression": 1} and data["verdicts"][0]["n_history"] == 11
    assert main(argv + ["--history-days", "20", "--out", str(out)]) == 0  # the old behaviour: masked
    assert json.loads(out.read_text())["counts"] == {"insufficient_history": 1}
