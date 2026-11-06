# SPDX-License-Identifier: Apache-2.0
"""CPU checks for independent selection and the frozen-winner comparison."""

from contextlib import nullcontext
from types import SimpleNamespace

import compare_grf as comparison
import pytest


def option(mode, tile, samples, status="ok"):
    return {
        "id": f"{mode}-{tile}", "config": {"grf_mode": mode, "tile_size": tile}, "status": status, "samples_ms": samples
    }


def test_independent_winners_and_resource_failures():
    options = [
        option("128", 16, [3, 1, 2]),
        option("128", 32, [4, 5, 6]),
        option("256", 16, [4, 3, 2]),
        option("256", 32, [1, 1.2, .8]),
        option("128", 64, [.01], "resource_error")
    ]
    winners = comparison.select_winners(options)
    assert winners["128"]["config"]["tile_size"] == 16
    assert winners["256"]["config"]["tile_size"] == 32
    with pytest.raises(RuntimeError, match="No correct, launchable 256-GRF candidate"):
        comparison.select_winners(options[:2])


@pytest.mark.parametrize("left,right,winner", [(0.8, 1, "128"), (1, 0.8, "256"), (0.98, 1, "tie"), (1, 0.98, "tie")])
def test_practical_ties(left, right, winner):
    assert comparison.summarize({"128": [left] * 5, "256": [right] * 5})["winner"] == winner


def test_frozen_winners_alternate(tmp_path):
    runner = object.__new__(comparison.Runner)
    runner.inputs, runner.device = {}, "xpu:0"
    order = []
    current = []

    def override(config):
        current[:] = [config["grf_mode"]]
        return nullcontext()

    runner.tuner = SimpleNamespace(
        runtime=SimpleNamespace(AttentionConfig=lambda **config: config, override_config=override),
        unified_attention=lambda **kw: None)

    def prepare(candidate, inputs, device, binary_dir=None):
        candidate.update(status="ok", native={"verified": candidate["config"]["grf_mode"]})

    def timer(fn, device):
        fn()
        order.extend(current)
        return 1 if current[0] == "128" else 2

    runner.prepare, runner.time = prepare, timer
    result = runner.compare({mode: {"grf_mode": mode} for mode in comparison.MODES}, tmp_path)
    assert order == ["128", "256", "256", "128", "128", "256", "256", "128", "128", "256"]
    assert result["winner"] == "128"
    assert all(len(samples) == comparison.PAIRS for samples in result["samples_ms"].values())


def test_requested_label_does_not_replace_native_mode(monkeypatch):
    runner = object.__new__(comparison.Runner)
    runner.compiled = {}
    runner.check = lambda: None
    runner.atol, runner.rtol = .025, .01

    def prepare(candidate, inputs, device):
        candidate["status"] = "ok"
        runner.compiled["kernel_unified_attention"] = object()

    runner.tuner = SimpleNamespace(prepare_option=prepare)
    monkeypatch.setattr(comparison, "native_metadata", lambda code: {"kernels": [{"grf_count": 256}]})
    inputs = {"out": SimpleNamespace(fill_=lambda value: None)}
    with pytest.raises(RuntimeError, match="Requested 128 GRFs"):
        runner.prepare(option("128", 16, []), inputs, "xpu:0")


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
