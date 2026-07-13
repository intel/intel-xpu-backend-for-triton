from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import List, Tuple

import pytest

from benchmark_helpers import make_cfg
from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser
from triton_kernels_benchmark.benchmark_testing import (
    BenchmarkCategory,
    MarkArgs,
)
from triton_kernels_benchmark.benchmark_utils import BenchmarkConfigs
from triton_kernels_benchmark.configs.benchmark_config_templates import CONFIGS

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None


def _collect_cases() -> List[Tuple[str, str, str]]:
    cases: List[Tuple[str, str, str]] = []
    for template in CONFIGS:
        # describe_metadata_only templates (vLLM) resolve get_benchmark -> import vllm
        # when probed; skip them unless vllm is installed.
        if template.describe_metadata_only and not _VLLM_AVAILABLE:
            continue
        probe = make_cfg(template)
        for shape in probe.supported_shapes:
            shape_str = str(shape)
            for provider_key in probe.supported_providers:
                cases.append((template.key, shape_str, provider_key))
    return cases


CASES = _collect_cases()
ALL_CATEGORIES = {cat.value for cat in BenchmarkCategory}


@pytest.mark.parametrize(("config_key", "shape", "provider"), CASES)
def test_benchmarks(
    config_key: str,
    shape: str,
    provider: str,
    benchmark_mark_args: MarkArgs,
    benchmark_reports_dir: Path | None,
    benchmark_describe_only: bool,
) -> None:
    configs = BenchmarkConfigs._get_configs(  # pylint: disable=protected-access
        configs_filter={config_key},
        categories_filter=ALL_CATEGORIES,
        providers_filter=[provider],
        shape_pattern=ShapePatternParser(shape),
    )
    assert len(configs) == 1, f"expected exactly one config for {config_key!r}, got {len(configs)}"
    cfg = configs[0]

    if benchmark_describe_only:
        assert cfg.selected_shapes, f"shape {shape} did not resolve for {config_key}"
        assert cfg.selected_providers, f"provider {provider} not supported by {config_key}"
        # Mirror the `triton-benchmarks describe` output (benchmark_utils.py:72).
        print(str(cfg))
        return

    args = benchmark_mark_args
    if benchmark_reports_dir is not None:
        # Mark.run writes a fixed "{plot_name}.csv"; each xdist worker gets its own
        # subdir to avoid clobbering peers. pytest_sessionfinish (conftest.py) merges
        # these into the consolidated reports the CLI produces.
        per_test_dir = benchmark_reports_dir / "_parts" / f"{config_key}__{shape}__{provider}"
        per_test_dir.mkdir(parents=True, exist_ok=True)
        args = dataclasses.replace(args, reports=str(per_test_dir))

    cfg.run(args)
