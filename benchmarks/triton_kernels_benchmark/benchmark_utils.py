from __future__ import annotations

from typing import List, Set
from dataclasses import dataclass, replace

import argparse
import time

from triton_kernels_benchmark.benchmark_testing import BenchmarkConfig, BenchmarkCategory, MarkArgs
from triton_kernels_benchmark.benchmark_config_templates import CONFIGS


@dataclass
class BenchmarkConfigs:
    configs: List[BenchmarkConfig]
    collect_only: bool
    select_all: bool
    args: MarkArgs
    tag: str = ""

    @classmethod
    def _get_configs(
        cls,
        config_filter: Set[str],
        select_all: bool,
        categories_filter: Set[str],
    ) -> List[BenchmarkConfig]:
        known_configs = {config.key: config for config in CONFIGS}
        selected_config_keys = known_configs.keys() if select_all else config_filter
        unknown_configs = selected_config_keys - known_configs.keys()
        if unknown_configs:
            raise NotImplementedError(f"Unknown configs are provided {unknown_configs}")
        # selected_categories =
        selected_configs = [
            known_configs[key]
            for key in selected_config_keys
            if {category.value
                for category in known_configs[key].categories}.issubset(categories_filter)
        ]
        if not selected_configs:
            raise AssertionError(f"No configs are selected from {config_filter} category {categories_filter}")
        return selected_configs

    def run(self):
        start = time.perf_counter()
        for config in self.configs:
            config.run(args=self.args, collect_only=self.collect_only, tag=self.tag)
        end = time.perf_counter()
        print(f"Total run time including overhead: {round(end-start, 3)} seconds.")

    @classmethod
    def _parse_args(cls, remaining_args) -> argparse.Namespace:

        parser = argparse.ArgumentParser()
        configs_group = parser.add_mutually_exclusive_group(required=True)
        all_configs = [config.key for config in CONFIGS]
        configs_group.add_argument(
            "--config",
            choices=all_configs,
            default=[],
            help=("List of the benchmark config to include."
                  f"Known configs: {all_configs}"
                  "Run `all --co` to get the supported configs details)."),
        )
        configs_group.add_argument(
            "--all",
            action="store_true",
            help="Include all known configs.",
        )
        parser.add_argument(
            "--co",
            action="store_true",
            help="List all known configs, do not run the benchmarks.",
        )
        parser.add_argument(
            "--provider",
            action="append",
            help=("Run benchmark with a provider. This option can be executed multiple times. "
                  "If not set all providers will be benchmarked in a given context - "
                  "hw capabilities, provider support for a specific set of benchmark config params, etc."),
        )
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
        parser.add_argument(
            "--tag",
            type=str,
            default="",
            required=False,
            help="How to tag results in the report.",
        )
        args = parser.parse_args(remaining_args)

        if not args.all and not args.config:
            raise ValueError("No configs are provided")

        return args

    @classmethod
    def _from_args(cls, args: MarkArgs, configs_filter: Set[str], categories_filter: Set[str],
                   providers_filter: List[str], collect_only: bool, select_all: bool, tag: str) -> BenchmarkConfigs:
        template_configs = cls._get_configs(
            config_filter=configs_filter,
            categories_filter=categories_filter,
            select_all=select_all,
        )
        return BenchmarkConfigs(
            configs=[replace(template, providers_filter=providers_filter) for template in template_configs],
            args=args,
            collect_only=collect_only,
            select_all=select_all,
            tag=tag,
        )

    @classmethod
    def from_args(cls) -> BenchmarkConfigs:
        base_args, remaining_args = MarkArgs.parse_common_args()
        additional_args = cls._parse_args(remaining_args)
        args = MarkArgs(**vars(base_args))
        return cls._from_args(
            args=args,
            configs_filter=({additional_args.config}
                            if isinstance(additional_args.config, str) else set(additional_args.config)),
            select_all=additional_args.all,
            categories_filter=set(additional_args.category),
            providers_filter=additional_args.provider,
            collect_only=additional_args.co,
            tag=additional_args.tag,
        )


def main():
    BenchmarkConfigs.from_args().run()


if __name__ == "__main__":
    main()
