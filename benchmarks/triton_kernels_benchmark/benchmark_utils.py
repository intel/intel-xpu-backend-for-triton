from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union, List, Dict
from dataclasses import dataclass, field, replace
from enum import Enum

import argparse
import time

from triton_kernels_benchmark.benchmark_testing import Benchmark, MarkArgs, Mark

from triton_kernels_benchmark import fused_softmax
from triton_kernels_benchmark import gemm_benchmark
from triton_kernels_benchmark import flash_attention_benchmark


class BenchmarkCategory(Enum):
    CORE = "core"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"


@dataclass
class BenchmarkSummary:
    key: str
    name: str
    variant_fields: List[str]
    variants: List[Union[int, str, List[str]]]
    supported_providers: List[str]
    selected_providers: List[str]

    def __str__(self) -> str:
        variants_repr = []
        for variant in self.variants:
            labels_and_values = [
                f"{label}:{value}"
                for label, value in zip(self.variant_fields, variant if isinstance(variant, list) else [variant])
            ]
            variants_repr.append(f"{self.key}[" + "-".join(labels_and_values) + "]")
        summary = [
            f"Name: {self.name}",
            "Variants:",
            *variants_repr,
        ]
        return "\n".join(summary)


@dataclass
class BenchmarkConfig:
    key: str
    get_benchmark: Callable[..., Mark]
    category: BenchmarkCategory
    providers_filter: Optional[list[str]] = None
    config_opts: Dict[str, Union[str, bool, List[str]]] = field(default_factory=dict)
    benchmark_summary: BenchmarkSummary = field(init=False)

    def _get_benchmark(self, apply_providers_filter=True) -> Mark:
        run_opts = self.config_opts
        if self.providers_filter and apply_providers_filter:
            run_opts = run_opts | {"providers_filter": self.providers_filter}
        mark = self.get_benchmark(**run_opts)
        if not isinstance(mark.benchmarks, Benchmark):
            raise NotImplementedError(
                "Benchmark config list is not supported, exactly one benchmark config is expected.")
        return mark

    def __post_init__(self):
        benchmark: Benchmark = self._get_benchmark().benchmarks
        full_benchmark: Benchmark = self._get_benchmark(False).benchmarks
        self.benchmark_summary = BenchmarkSummary(
            key=self.key,
            name=benchmark.plot_name,
            variant_fields=benchmark.x_names,
            variants=benchmark.x_vals,
            supported_providers=full_benchmark.line_vals,
            selected_providers=benchmark.line_vals,
        )

    def __str__(self) -> str:
        str_repr = [
            f"Config: {self.key}",
            f"Config category: {self.category.value}",
            f"Run options: {self.config_opts}",
            f"Supported providers {self.benchmark_summary.supported_providers}",
            f"Selected providers: {self.benchmark_summary.selected_providers}",
            str(self.benchmark_summary),
        ]
        return "\n".join(str_repr)

    def run(self, dry_run: bool, args: MarkArgs):
        print(str(self))
        if dry_run:
            return
        # FIXME: add run summary - total timings, etc
        print(f"Running {self.key}")
        self._get_benchmark().run(show_plots=False, print_data=True, mark_args=args)


@dataclass
class BenchmarkConfigs:
    configs: List[BenchmarkConfig]
    dry_run: bool
    args: MarkArgs

    _benchmark_config_templates: ClassVar[List[BenchmarkConfig]] = [
        # To filter the providers list you can run console entry point with multiple `--provider <supported_provider>` options
        BenchmarkConfig(
            key="fused_softmax",
            get_benchmark=fused_softmax.get_benchmark,
            config_opts={},
            category=BenchmarkCategory.CORE,
        ),
        BenchmarkConfig(
            key="gemm",
            get_benchmark=gemm_benchmark.get_benchmark,
            config_opts={},
            category=BenchmarkCategory.CORE,
        ),
        BenchmarkConfig(
            key="gemm_bt",
            get_benchmark=gemm_benchmark.get_benchmark,
            config_opts={"transpose_b": True},
            category=BenchmarkCategory.CORE,
        ),
        BenchmarkConfig(
            key="flash_attention",
            get_benchmark=flash_attention_benchmark.get_benchmark,
            config_opts={"fa_kernel_mode": "fwd"},
            category=BenchmarkCategory.CORE,
        ),
    ]

    @classmethod
    def _get_configs(cls, suite: str, categories_filter: List[str]) -> List[BenchmarkConfig]:
        selected_config = None
        all_configs = cls._benchmark_config_templates
        config_keys = []
        for config in all_configs:
            key = config.key
            if key == "all":
                raise AssertionError("`all` can't be a config name key")
            if key in config_keys:
                raise ValueError(f"Duplicate config name key - {key}")
            config_keys.append(key)
            if key == suite:
                selected_config = config
        if selected_config and selected_config.category.value not in categories_filter:
            raise AssertionError(
                f"Selected config {selected_config.key} category {selected_config.category} is not in the categories filter"
            )
        if selected_config:
            return [selected_config]
        if suite == "all":
            return [config for config in all_configs if config.category.value in categories_filter]
        raise NotImplementedError(
            f"Config {suite} is unknown. Choose `all` or one of the following: " + ", ".join(config_keys), )

    def run(self):
        start = time.perf_counter()
        for config in self.configs:
            config.run(args=self.args, dry_run=self.dry_run)
        end = time.perf_counter()
        print(f"Total run time: {round(end-start, 3)}")

    @classmethod
    def _parse_args(cls, remaining_args) -> argparse.Namespace:

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "benchmark_suite",
            help="Name of the benchmark config. Run `all` `--dry-run` to get the list of the supported configs",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List all known configs, do not run the benchmarks",
        )
        parser.add_argument(
            "--provider", action="append",
            help=("Run benchmark with a provider. This option can be executed multiple times. "
                  "If not set all providers will be benchmarked in a given context - "
                  "hw capabilities, provider support for a specific set of benchmark config params, etc"))
        categories = [cat.value for cat in BenchmarkCategory]
        parser.add_argument(
            "--bench-category", action="append", choices=categories, default=categories, help=str(
                "Apply the action to benchmark suites in the specified category. Must be one of: " +
                ", ".join(categories), ))
        args = parser.parse_args(remaining_args)

        return args

    @classmethod
    def from_args(cls) -> BenchmarkConfigs:
        base_args, remaining_args = MarkArgs.parse_common_args()
        additional_args = cls._parse_args(remaining_args)
        template_configs = cls._get_configs(
            additional_args.benchmark_suite,
            additional_args.bench_category,
        )
        return BenchmarkConfigs(
            configs=[replace(template, providers_filter=additional_args.provider) for template in template_configs],
            args=MarkArgs(**vars(base_args)), dry_run=additional_args.dry_run)


def main():
    BenchmarkConfigs.from_args().run()


if __name__ == "__main__":
    main()
