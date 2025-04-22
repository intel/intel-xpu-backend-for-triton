from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union, List, Dict
from dataclasses import dataclass, field, replace
from enum import Enum

import argparse
import time

from triton_kernels_benchmark.benchmark_testing import Benchmark, MarkArgs, Mark

from triton_kernels_benchmark import fused_softmax, gemm_benchmark, flash_attention_benchmark


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
    config_summary: BenchmarkSummary = field(init=False)

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
        self.config_summary = BenchmarkSummary(
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
            f"Supported providers {self.config_summary.supported_providers}",
            f"Selected providers: {self.config_summary.selected_providers}",
            str(self.config_summary),
        ]
        return "\n".join(str_repr)

    def run(self, collect_only: bool, args: MarkArgs):
        print(str(self))
        if collect_only:
            return
        # FIXME: add run summary - total timings, etc
        print(f"Running {self.key}")
        self._get_benchmark().run(show_plots=False, print_data=True, mark_args=args)


@dataclass
class BenchmarkConfigs:
    configs: List[BenchmarkConfig]
    collect_only: bool
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
                raise AssertionError("`all` can't be a config name key.")
            if key in config_keys:
                raise ValueError(f"Duplicate config name key - {key}.")
            config_keys.append(key)
            if key == suite:
                selected_config = config
        if selected_config and selected_config.category.value not in categories_filter:
            raise AssertionError(
                f"Selected config {selected_config.key} category {selected_config.category} is not in the categories filter."
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
            config.run(args=self.args, collect_only=self.collect_only)
        end = time.perf_counter()
        print(f"Total run time including overhead: {round(end-start, 3)} seconds.")

    @classmethod
    def _parse_args(cls, remaining_args) -> argparse.Namespace:

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "config_filter",
            help="Name of the benchmark config. Run `all --co` to get the list of the supported configs.",
        )
        parser.add_argument(
            "--co",
            action="store_true",
            help="List all known configs, do not run the benchmarks.",
        )
        parser.add_argument(
            "--provider", action="append",
            help=("Run benchmark with a provider. This option can be executed multiple times. "
                  "If not set all providers will be benchmarked in a given context - "
                  "hw capabilities, provider support for a specific set of benchmark config params, etc."))
        categories = [cat.value for cat in BenchmarkCategory]
        categories_str = ", ".join(categories)
        parser.add_argument(
            "--category",
            action="append",
            choices=categories,
            default=categories,
            help=str(
                f"Apply the action to bechmark suites in the specified category. Must be one of: {categories_str}.", ),
        )
        args = parser.parse_args(remaining_args)

        return args

    @classmethod
    def _from_args(
        cls,
        args: MarkArgs,
        configs_filter: str,
        categories_filter: List[str],
        providers_filter: List[str],
        collect_only: bool,
    ) -> BenchmarkConfigs:
        template_configs = cls._get_configs(
            configs_filter,
            categories_filter,
        )
        return BenchmarkConfigs(
            configs=[replace(template, providers_filter=providers_filter) for template in template_configs],
            args=args,
            collect_only=collect_only,
        )

    @classmethod
    def from_args(cls) -> BenchmarkConfigs:
        base_args, remaining_args = MarkArgs.parse_common_args()
        additional_args = cls._parse_args(remaining_args)
        args = MarkArgs(**vars(base_args))
        return cls._from_args(
            args=args,
            configs_filter=additional_args.config_filter,
            categories_filter=additional_args.category,
            providers_filter=additional_args.provider,
            collect_only=additional_args.co,
        )


def main():
    BenchmarkConfigs.from_args().run()


if __name__ == "__main__":
    main()
