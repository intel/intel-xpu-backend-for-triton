import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import triton_benchmarks_pins as pins


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    for name in ("AUTOTUNE_PINS_MODE", "AUTOTUNE_PINS_DIR", "AUTOTUNE_PINS_RECORDS", "AUTOTUNE_PINS_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pins, "_SESSION", None)
    yield
    monkeypatch.setattr(pins, "_SESSION", None)


@pytest.mark.parametrize("name, expected", [
    ("Intel(R) Arc(TM) B580 Graphics", "b580"),
    ("Intel(R) Arc(TM) Pro B60 Graphics", "pro-b60"),
    ("Intel(R) Data Center GPU Max 1100", "max-1100"),
    ("Intel(R) Data Center GPU Max 1550", "max-1550"),
])
def test_device_id(name, expected):
    assert pins.device_id(name) == expected


def test_pin_file_round_trip(tmp_path):
    pin_file = pins.PinFile("gemm", "b580", triton={("k", "(1, 2)"): {"BLOCK_M": 64, "num_warps": 8}},
                            inductor={("flex_attention", '{"provider": "triton"}', "in"): "TritonTemplateCaller: x"})
    path = pins.pin_path(tmp_path, "b580", "gemm")
    pin_file.save(path)
    loaded = pins.PinFile.load(path)
    assert loaded == pin_file
    assert loaded.version == pin_file.version
    other = pins.PinFile("gemm", "b580", triton={("k", "(1, 2)"): {"BLOCK_M": 128, "num_warps": 8}})
    assert other.version != pin_file.version


def _triton_record(key, timings, bench="gemm"):
    return {
        "kind": "triton", "benchmark": bench, "device": "b580", "kernel": "mod.kernel", "key": key, "best": None,
        "timings": [[{"BLOCK_M": block_m}, time] for block_m, time in timings.items()]
    }


def _write_run(directory: Path, entries):
    directory.mkdir(parents=True)
    with open(directory / "gemm.jsonl", "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return directory


def test_update_policy(tmp_path):
    pins_dir = tmp_path / "pins"
    pins.PinFile(
        "gemm", "b580", triton={
            ("mod.kernel", "kept"): {"BLOCK_M": 32},
            ("mod.kernel", "changed"): {"BLOCK_M": 32},
            ("mod.kernel", "replaced"): {"BLOCK_M": 256},
            ("mod.kernel", "dropped"): {"BLOCK_M": 32},
            ("mod.kernel", "noisy"): {"BLOCK_M": 32},
        }).save(pins.pin_path(pins_dir, "b580", "gemm"))
    older = _write_run(tmp_path / "run1", [_triton_record("noisy", {32: 1.2, 64: 0.85})])
    newer = _write_run(tmp_path / "run2", [
        _triton_record("new", {32: 1.0, 64: 0.5}),
        _triton_record("kept", {32: 1.0, 64: 0.98}),
        _triton_record("changed", {32: 1.0, 64: 0.9}),
        _triton_record("replaced", {32: 1.0, 64: 0.9}),
        _triton_record("noisy", {32: 0.9, 64: 0.85}),
    ])
    out = tmp_path / "out"
    summaries = pins.update_pins([older, newer], pins_dir, out, min_gain=0.03)
    assert len(summaries) == 1
    summary = summaries[0]
    result = pins.PinFile.load(pins.pin_path(out, "b580", "gemm")).triton
    assert result[("mod.kernel", "new")] == {"BLOCK_M": 64}
    assert result[("mod.kernel", "kept")] == {"BLOCK_M": 32}  # 2% gain is below min_gain
    assert result[("mod.kernel", "changed")] == {"BLOCK_M": 64}
    assert result[("mod.kernel", "replaced")] == {"BLOCK_M": 64}
    # Median 1.05 vs 0.85 is a 19% gain, but the pin itself moved 29% between the runs.
    assert result[("mod.kernel", "noisy")] == {"BLOCK_M": 32}
    assert ("mod.kernel", "dropped") not in result
    assert (summary.new, summary.kept, len(summary.changed), summary.replaced, summary.dropped) == (1, 2, 1, 1, 1)


def test_update_keeps_first_record_of_a_run(tmp_path):
    run = _write_run(tmp_path / "run",
                     [_triton_record("k", {32: 1.0, 64: 2.0}),
                      _triton_record("k", {32: 2.0, 64: 1.0})])
    pins.update_pins([run], tmp_path / "pins", tmp_path / "pins", min_gain=0.03)
    assert pins.PinFile.load(pins.pin_path(tmp_path / "pins", "b580",
                                           "gemm")).triton[("mod.kernel", "k")] == {"BLOCK_M": 32}


def test_update_tie_keeps_autotuner_choice(tmp_path):
    entry = _triton_record("k", {32: 1.0, 64: 1.0})
    entry["best"] = {"BLOCK_M": 64}
    run = _write_run(tmp_path / "run", [entry])
    pins.update_pins([run], tmp_path / "pins", tmp_path / "pins", min_gain=0.03)
    assert pins.PinFile.load(pins.pin_path(tmp_path / "pins", "b580",
                                           "gemm")).triton[("mod.kernel", "k")] == {"BLOCK_M": 64}


def test_update_skips_config_failing_in_newest_run(tmp_path):
    older = _write_run(tmp_path / "run1", [_triton_record("k", {32: 0.5, 64: 1.0})])
    newer = _write_run(tmp_path / "run2", [_triton_record("k", {32: None, 64: 1.0})])
    pins.update_pins([older, newer], tmp_path / "pins", tmp_path / "pins", min_gain=0.03)
    assert pins.PinFile.load(pins.pin_path(tmp_path / "pins", "b580",
                                           "gemm")).triton[("mod.kernel", "k")] == {"BLOCK_M": 64}


def _git(cwd, *args):
    env = {
        **os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com", "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com"
    }
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_show(tmp_path, capsys):
    pins_dir = tmp_path / "autotune_pins"
    path = pins.pin_path(pins_dir, "b580", "gemm")
    old = pins.PinFile("gemm", "b580", triton={("mod.kernel", "(1024, 1024)"): {"BLOCK_M": 64, "num_warps": 8}})
    old.save(path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "Add gemm pins")
    new = pins.PinFile("gemm", "b580", triton={("mod.kernel", "(1024, 1024)"): {"BLOCK_M": 128, "num_warps": 8}})
    new.save(path)
    _git(tmp_path, "commit", "-qam", "Move gemm pin")

    pins.main(["show", f"pinned@{old.version}", "--pins", str(pins_dir)])
    out = capsys.readouterr().out
    assert "autotune_pins/b580/gemm.json, added in" in out and "Add gemm pins" in out
    assert "working tree" not in out
    assert "mod.kernel\n  (1024, 1024): BLOCK_M=64, num_warps=8\n" in out

    pins.main(["show", new.version, "--pins", str(pins_dir)])
    out = capsys.readouterr().out
    assert "(working tree)" in out and "Move gemm pin" in out and "BLOCK_M=128" in out

    with pytest.raises(SystemExit):
        pins.main(["show", "000000000000", "--pins", str(pins_dir)])


def test_inactive_without_mode():
    with pins.benchmark("gemm"):
        assert pins.report_value() is None
    pins.raise_for_misses()


def test_invalid_mode(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "pin")
    with pytest.raises(pins.AutotunePinsError):
        pins.report_value()


def test_pinned_benchmark_without_pin_file(monkeypatch, tmp_path):
    pytest.importorskip("triton")
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "pinned")
    monkeypatch.setenv("AUTOTUNE_PINS_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOTUNE_PINS_DEVICE", "test-gpu")
    with pins.benchmark("no-autotuning"):
        assert pins.report_value().startswith("pinned@")
    pins.raise_for_misses()


@pytest.fixture(name="triton_kernel")
def fixture_triton_kernel():
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("no XPU")

    def make():

        @triton.autotune(configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})], key=["n"])
        @triton.jit
        def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=offsets < n) + 1, mask=offsets < n)

        def run(n):
            x = torch.zeros(n, device="xpu")
            y = torch.empty_like(x)
            add_one[lambda meta: (triton.cdiv(n, meta["BLOCK"]), )](x, y, n)
            assert torch.all(y == 1)

        return add_one, run

    return make


def test_triton_record_then_pin(monkeypatch, tmp_path, triton_kernel):
    monkeypatch.setenv("AUTOTUNE_PINS_DIR", str(tmp_path / "pins"))
    monkeypatch.setenv("AUTOTUNE_PINS_RECORDS", str(tmp_path / "records"))
    monkeypatch.setenv("AUTOTUNE_PINS_DEVICE", "test-gpu")

    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "record")
    kernel, run = triton_kernel()
    with pins.benchmark("add-one"):
        run(4096)
        assert pins.report_value() == "full"
    [entry] = [json.loads(line) for line in (tmp_path / "records" / "add-one.jsonl").read_text().splitlines()]
    assert entry["kernel"].endswith("add_one") and len(entry["timings"]) == 2
    best = entry["best"]
    pins.update_pins([tmp_path / "records"], tmp_path / "pins", tmp_path / "pins", min_gain=0.03)

    monkeypatch.setattr(pins, "_SESSION", None)
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "pinned")
    kernel, run = triton_kernel()
    with pins.benchmark("add-one"):
        run(4096)
        assert pins.report_value().startswith("pinned@")
    assert pins.triton_config_dict(kernel.best_config) == best
    assert not hasattr(kernel, "configs_timings")  # nothing was benchmarked
    pins.raise_for_misses()

    kernel, run = triton_kernel()
    with pins.benchmark("add-one"):
        run(8192)  # another autotune key, which has no pin
    assert kernel.configs_timings  # autotuned, so the rest of the run is still measured
    with pytest.raises(pins.AutotunePinsError, match="no pin for"):
        pins.raise_for_misses()


def test_benchmarks_sharing_a_kernel(monkeypatch, tmp_path, triton_kernel):
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "record")
    monkeypatch.setenv("AUTOTUNE_PINS_RECORDS", str(tmp_path))
    monkeypatch.setenv("AUTOTUNE_PINS_DEVICE", "test-gpu")
    _, run = triton_kernel()
    for key in ("first", "second"):  # the second one must not reuse the decision of the first
        with pins.benchmark(key):
            run(4096)
        assert len((tmp_path / f"{key}.jsonl").read_text().splitlines()) == 1


def test_pruned_to_one_config_needs_no_pin(monkeypatch, tmp_path):
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("no XPU")
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "pinned")
    monkeypatch.setenv("AUTOTUNE_PINS_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOTUNE_PINS_DEVICE", "test-gpu")

    @triton.autotune(configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})], key=["n"],
                     prune_configs_by={"early_config_prune": lambda configs, nargs, **kwargs: configs[:1]})
    @triton.jit
    def copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=offsets < n), mask=offsets < n)

    x = torch.ones(4096, device="xpu")
    y = torch.empty_like(x)
    with pins.benchmark("pruned-to-one"):
        copy[lambda meta: (triton.cdiv(4096, meta["BLOCK"]), )](x, y, 4096)
    assert torch.all(y == 1)
    pins.raise_for_misses()


def test_pin_rejected_by_pruning(monkeypatch, tmp_path):
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("no XPU")
    monkeypatch.setenv("AUTOTUNE_PINS_DIR", str(tmp_path / "pins"))
    monkeypatch.setenv("AUTOTUNE_PINS_RECORDS", str(tmp_path / "records"))
    monkeypatch.setenv("AUTOTUNE_PINS_DEVICE", "test-gpu")
    rejected = []

    def prune(configs, _nargs, **_kwargs):
        return [c for c in configs if c.kwargs["BLOCK"] not in rejected]

    @triton.autotune(configs=[triton.Config({"BLOCK": b}) for b in (64, 128, 256)], key=["n"],
                     prune_configs_by={"early_config_prune": prune})
    @triton.jit
    def copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=offsets < n), mask=offsets < n)

    def run():
        x = torch.ones(4096, device="xpu")
        y = torch.empty_like(x)
        copy[lambda meta: (triton.cdiv(4096, meta["BLOCK"]), )](x, y, 4096)
        assert torch.all(y == 1)

    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "record")
    with pins.benchmark("copy"):
        run()
    pins.update_pins([tmp_path / "records"], tmp_path / "pins", tmp_path / "pins", min_gain=0.03)

    monkeypatch.setattr(pins, "_SESSION", None)
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "pinned")
    rejected.append(copy.best_config.kwargs["BLOCK"])  # the pruning now rejects the pinned config
    with pins.benchmark("copy"):
        run()
    assert copy.best_config.kwargs["BLOCK"] not in rejected
    with pytest.raises(pins.AutotunePinsError, match="pinned config is not among the configs"):
        pins.raise_for_misses()


def test_inductor_record_then_pin(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("no XPU")
    from torch._inductor import config as inductor_config  # pylint: disable=import-outside-toplevel
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setattr(inductor_config, "force_disable_caches", False)
    monkeypatch.setenv("AUTOTUNE_PINS_DIR", str(tmp_path / "pins"))
    monkeypatch.setenv("AUTOTUNE_PINS_RECORDS", str(tmp_path / "records"))
    monkeypatch.setenv("AUTOTUNE_PINS_DEVICE", "test-gpu")

    def run():
        torch._dynamo.reset()  # pylint: disable=protected-access
        a, b = (torch.randn(512, 512, device="xpu", dtype=torch.float16) for _ in range(2))
        matmul = torch.compile(lambda x, y: x @ y, options={"max_autotune": True})
        with pins.benchmark_point({"M": 512}, "triton"):
            torch.testing.assert_close(matmul(a, b), a @ b, atol=1e-2, rtol=1e-2)

    run()  # warms the graph cache, which would skip the autotuning below
    monkeypatch.setattr(pins, "_SESSION", None)
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "record")
    with pins.benchmark("matmul"):
        run()
    entries = [json.loads(line) for line in (tmp_path / "records" / "matmul.jsonl").read_text().splitlines()]
    assert entries and all(e["kind"] == "inductor" and len(e["timings"]) > 1 for e in entries)
    pins.update_pins([tmp_path / "records"], tmp_path / "pins", tmp_path / "pins", min_gain=0.03)

    monkeypatch.setattr(pins, "_SESSION", None)
    monkeypatch.setenv("AUTOTUNE_PINS_MODE", "pinned")
    with pins.benchmark("matmul"):
        run()
    assert pins._SESSION.hits == len(entries)  # pylint: disable=protected-access
    pins.raise_for_misses()
