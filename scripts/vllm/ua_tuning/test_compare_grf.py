# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the default-versus-256 experiment and failure reporting."""

import copy
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import compare_grf as comparison
import pytest


def option(mode, tile, samples, status="ok"):
    return {
        "id": f"{mode}-{tile}", "config": {"grf_mode": None if mode == "default" else mode, "tile_size": tile},
        "status": status, "samples_ms": samples
    }


def test_default_omits_override_and_preserves_grid():
    grid = [{"grf_mode": mode, "tile_size": tile} for tile in (16, 32) for mode in ("default", "256")]
    tuner = SimpleNamespace(candidate_configs=lambda key: grid)
    candidates = comparison.mode_candidates(tuner, None)
    assert candidates == [{**config, "grf_mode": None if config["grf_mode"] == "default" else "256"} for config in grid]
    assert grid[0]["grf_mode"] == "default"


def test_independent_winners_and_resource_failures():
    options = [
        option("default", 16, [3, 1, 2]),
        option("default", 32, [4, 5, 6]),
        option("256", 16, [4, 3, 2]),
        option("256", 32, [1, 1.2, .8]),
        option("default", 64, [.01], "resource_error"),
        option("256", 128, [.001], "correctness_error")
    ]
    winners = comparison.select_winners(options)
    assert winners["default"]["config"]["tile_size"] == 16
    assert winners["256"]["config"]["tile_size"] == 32
    with pytest.raises(RuntimeError, match="No correct, launchable 256-GRF candidate"):
        comparison.select_winners(options[:2])


@pytest.mark.parametrize("left,right,winner", [(0.8, 1, "default"), (1, 0.8, "256"), (0.98, 1, "tie"), (1, 0.98, "tie")])
def test_practical_ties(left, right, winner):
    assert comparison.summarize({"default": [left] * comparison.PAIRS, "256": [right] * comparison.PAIRS})["winner"] == winner


def test_inconsistent_pairs_do_not_claim_a_win():
    result = comparison.summarize({"default": [1] * comparison.PAIRS, "256": [.8, .8, .8, .8, 1.1, 1.1]})
    assert result["median_ms"]["256"] < result["median_ms"]["default"]
    assert result["winner"] == "tie"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_samples_cannot_hide_behind_a_valid_median(value):
    samples = {mode: [1] * comparison.PAIRS for mode in comparison.MODES}
    samples["default"][0] = value
    with pytest.raises(RuntimeError, match="Invalid winner timing"):
        comparison.summarize(samples)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("identical", [False, True])
def test_frozen_winners_alternate_and_identical_binaries_tie(tmp_path, reverse, identical):
    runner = object.__new__(comparison.Runner)
    runner.inputs, runner.device = {}, "xpu:0"
    order = []
    current = []

    def override(config):
        current[:] = [comparison.requested_mode(config)]
        return nullcontext()

    runner.tuner = SimpleNamespace(
        runtime=SimpleNamespace(AttentionConfig=lambda **config: config, override_config=override),
        unified_attention=lambda **kw: None)

    def prepare(candidate, inputs, device, binary_dir=None):
        mode = comparison.requested_mode(candidate["config"])
        candidate.update(status="ok", native={"attention": {"sha256": "same" if identical else mode}})

    def timer(fn, device):
        fn()
        order.extend(current)
        return 1 if current[0] == "default" else 2

    runner.prepare, runner.time = prepare, timer
    result = runner.compare({mode: option(mode, 16, [])["config"] for mode in comparison.MODES}, tmp_path, reverse=reverse)
    expected = []
    for pair in range(comparison.PAIRS):
        expected.extend(comparison.MODES if (pair + reverse) % 2 == 0 else reversed(comparison.MODES))
    assert order == expected
    assert result["winner"] == ("tie" if identical else "default")
    assert result["same_launch_settings"]
    assert all(len(samples) == comparison.PAIRS for samples in result["samples_ms"].values())


@pytest.mark.parametrize("mode,actual,valid", [("default", 128, True), ("default", 256, True), ("256", 256, True),
                                              ("256", 128, False)])
def test_requested_label_does_not_replace_native_mode(monkeypatch, mode, actual, valid):
    runner = object.__new__(comparison.Runner)
    runner.compiled = {}
    runner.check = lambda: None
    runner.atol, runner.rtol = .025, .01

    def prepare(candidate, inputs, device):
        candidate["status"] = "ok"
        runner.compiled["kernel_unified_attention"] = object()

    runtime = SimpleNamespace(AttentionConfig=lambda **config: SimpleNamespace(
        launch_kwargs=lambda backend: {} if config["grf_mode"] is None else {"grf_mode": config["grf_mode"]}))
    runner.tuner = SimpleNamespace(prepare_option=prepare, runtime=runtime)
    monkeypatch.setattr(comparison, "native_metadata", lambda code: {"kernels": [{"grf_count": actual}]})
    inputs = {"out": SimpleNamespace(fill_=lambda value: None)}
    candidate = option(mode, 16, [])
    if valid:
        runner.prepare(candidate, inputs, "xpu:0")
        assert candidate["native"]["kernel_unified_attention"]["kernels"][0]["grf_count"] == actual
        assert ("grf_mode" in candidate["launch_kwargs"]) == (mode == "256")
    else:
        with pytest.raises(RuntimeError, match="Requested 256 GRFs"):
            runner.prepare(candidate, inputs, "xpu:0")


def test_fixed_settings_controls_and_unavailable_counterpart():
    options = [option("default", 16, [1]), option("default", 32, [2]), option("256", 16, [2]), option("256", 32, [.8])]
    winners = comparison.select_winners(options)
    groups, missing = comparison.comparison_groups(winners, options)
    assert len(groups) == 3
    assert not missing
    for name, configs in groups.items():
        if name.startswith("matched_"):
            assert comparison.launch_settings(configs["default"]) == comparison.launch_settings(configs["256"])
    options[2]["status"] = "resource_error"
    groups, missing = comparison.comparison_groups(winners, options)
    assert "matched_default_winner" not in groups
    assert missing["matched_default_winner"]["256"] == "resource_error"


@pytest.mark.parametrize("winner", comparison.MODES)
def test_both_directions_need_fresh_confirmation(winner):
    initial = {"native": {"identity": "frozen"}, "winner": winner}
    confirmation = {"native": {"identity": "frozen"}, "winner": "tie"}
    assert comparison.confirm_result(copy.deepcopy(initial), confirmation)["winner"] == f"unconfirmed_{winner}"
    assert comparison.confirm_result(copy.deepcopy(initial), initial)["winner"] == winner
    with pytest.raises(RuntimeError, match="native binaries differ"):
        comparison.confirm_result(initial, {"native": {"identity": "changed"}, "winner": winner})


def test_search_excludes_timing_failures_and_keeps_measuring(tmp_path):
    class ResourceError(Exception):
        pass

    runner = object.__new__(comparison.Runner)
    runner.inputs, runner.device, runner.resource_error = {}, "xpu:0", ResourceError
    current = []

    def override(config):
        current[:] = [config["tile_size"]]
        return nullcontext()

    runner.tuner = SimpleNamespace(
        runtime=SimpleNamespace(AttentionConfig=lambda **config: config, override_config=override),
        unified_attention=lambda **kw: None, synchronize=lambda device: None)

    def timer(fn, device):
        if current[0] == 16:
            raise AssertionError("Incorrect output after graph replay")
        if current[0] == 32:
            raise ResourceError("Resource limit")
        return 1

    runner.time = timer
    options = [option("default", 16, []), option("256", 32, []), option("default", 64, [])]
    runner.search(options, tmp_path, seed=20, rounds=2)
    assert [candidate["status"] for candidate in options] == ["correctness_error", "resource_error", "ok"]
    assert options[2]["samples_ms"] == [1, 1]
    assert json.loads((tmp_path / "search.json").read_text()) == options


def test_empty_manifest_does_not_need_a_gpu(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    monkeypatch.setattr(comparison, "Runner", lambda device: pytest.fail("Empty manifest instantiated the GPU runner"))
    comparison.run(SimpleNamespace(output=tmp_path / "reports", manifest=manifest, replay=None))
    summary = json.loads((tmp_path / "reports/summary.json").read_text())
    assert summary["requested_workloads"] == summary["processed_workloads"] == 0


def test_workload_errors_and_correctness_exclusions_preserve_coverage(tmp_path, monkeypatch):
    @dataclass
    class Key:
        value: int = 1

    class FakeRunner:
        def __init__(self, device):
            self.device = device
            self.torch = SimpleNamespace(inference_mode=nullcontext)
            self.tuner = SimpleNamespace(
                candidate_configs=lambda key: [{"grf_mode": mode, "tile_size": tile}
                                              for tile in (16, 32) for mode in comparison.MODES],
                config_id=lambda config: json.dumps(config),
                runtime=SimpleNamespace(make_attention_key=lambda **inputs: Key(), AttentionConfig=lambda **kw: kw,
                                        validate_config=lambda *args: True))

        def allocate(self, case, seed):
            self.case, self.inputs = case, {}
            if case["id"] == "error":
                raise RuntimeError("Synthetic workload failure")

        def prepare(self, candidate, inputs, device):
            status = "ok"
            if self.case["id"] == "unavailable" and comparison.requested_mode(candidate["config"]) == "default":
                status = "resource_error"
            if self.case["id"] == "complete" and candidate["config"]["tile_size"] == 32:
                status = "correctness_error"
            candidate["status"] = status

        def search(self, options, directory, seed, rounds):
            for candidate in options:
                if candidate["status"] == "ok":
                    candidate["samples_ms"] = [1 if comparison.requested_mode(candidate["config"]) == "default" else .8]

        def compare(self, configs, directory):
            return {"winner": "256", "native": {"default": "d", "256": "f"}, "identical_binaries": False,
                    "median_ms": {"default": 1, "256": .8}, "latency_reduction_fraction_256": .2}

        def release(self):
            pass

    def replay(command, check):
        frozen = Path(command[command.index("--replay") + 1])
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "confirmation.json").write_text((frozen.parent / "initial.json").read_text())

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": name} for name in ("unavailable", "error", "complete")]))
    monkeypatch.setattr(comparison, "Runner", FakeRunner)
    monkeypatch.setattr(comparison.subprocess, "run", replay)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    args = SimpleNamespace(output=tmp_path / "reports", manifest=manifest, replay=None, device="xpu:0", seed=20, rounds=3)
    with pytest.raises(RuntimeError, match="finished with errors"):
        comparison.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["processed_workloads"] == summary["requested_workloads"] == 3
    assert summary["statuses"] == {"unavailable": 1, "error": 1, "completed": 1}
    assert summary["counts"] == {"256": 1}
    assert summary["correctness_exclusions"] == 2
    records = json.loads((args.output / "workloads.json").read_text())
    assert records[0]["missing_modes"] == ["default"]
    assert records[2]["candidate_statuses"]["default"]["correctness_error"] == 1


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float8_e4m3fn"])
@pytest.mark.parametrize("window,expected", [(None, [13 / 3, 4, 2]), (2, [4, 2, 2])])
def test_paged_reference_causal_window_and_fp8(dtype_name, window, expected):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    case = {"seq_lens": [[2, 4], [1, 2]], "block_size": 2, "q_heads": 4, "k_heads": 2, "sliding_window": window}
    values = torch.tensor([1, 3, 5, 7]).reshape(2, 2, 1, 1).expand(2, 2, 2, 16).to(dtype)
    inputs = {
        "q": torch.zeros(3, 4, 16).to(dtype), "k": torch.zeros_like(values), "v": values, "out": torch.empty(3, 4, 16),
        "block_table": torch.tensor([[1, 0], [0, 0]]), "softmax_scale": .25, "softcap": 0
    }
    actual = comparison.reference(torch, case, inputs)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype).reshape(3, 1, 1).expand_as(actual))
