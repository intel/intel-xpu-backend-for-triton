from __future__ import annotations

from dataclasses import fields

from triton_kernels_benchmark.benchmark_testing import BenchmarkConfigRunResult
from triton_kernels_benchmark.configs.benchmark_config_templates import CONFIGS


def make_cfg(template, providers: list[str] | None = None) -> BenchmarkConfigRunResult:
    template_kwargs = {f.name: getattr(template, f.name) for f in fields(template) if f.name != "providers_filter"}
    return BenchmarkConfigRunResult(**template_kwargs, providers_filter=providers, shape_pattern=None)


def template_for(config_key: str):
    for template in CONFIGS:
        if template.key == config_key:
            return template
    raise KeyError(f"No benchmark template with key {config_key!r}")
