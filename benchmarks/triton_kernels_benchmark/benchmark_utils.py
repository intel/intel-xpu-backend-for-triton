from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import asdict, dataclass, fields

import argparse
import time
import datetime

import os
import socket

import pandas as pd

from triton_kernels_benchmark.benchmark_testing import (
    BenchmarkConfig,
    BenchmarkConfigRunResult,
    BenchmarkCategory,
    MarkArgs,
)
from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser
from triton_kernels_benchmark.benchmark_config_templates import CONFIGS


@dataclass
class BenchmarkConfigs(MarkArgs):
    # Workaround for the default / non-default arguments. In Python 3.10 can be replaced by @dataclass(kw_only=True)
    configs: Optional[List[BenchmarkConfigRunResult]] = None
    collect_only: bool = False
    json_output: bool = False
    detailed_output: bool = False
    junit_report: bool = False
    tag: str = ""

    def __post_init__(self):
        if not self.configs:
            raise ValueError("configs value must be provided")

    def run(self):

        def _junit_report(run_results: List[BenchmarkConfigRunResult]) -> str:
            report_header = (
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
                "<testsuites>\n"
                f"  <testsuite name=\"triton-benchmarks\" errors=\"0\" failures=\"0\" skipped=\"0\" tests=\"{len(run_results)}\" "
                f"time=\"1\" timestamp=\"{datetime.datetime.now().astimezone().isoformat()}\" hostname=\"{socket.gethostname()}\">\n"
            )
            test_items = []
            for run_result in run_results:
                test_items.append(
                    f"<testcase classname=\"\" name=\"{run_result.key}\" time=\"{run_result.run_time}\" />\n")
            report_footer = ("  </testsuite>\n"
                             "</testsuites>\n")
            return report_header + "".join(test_items) + report_footer

        def _df_or_json(df_to_print: pd.DataFrame):
            if self.json_output:
                return (df_to_print.copy().reset_index().to_json(orient="records", lines=True))
            return df_to_print

        def _json_print(obj: Union[str, pd.DataFrame]):
            if self.json_output and isinstance(obj, pd.DataFrame):
                print(obj.copy().reset_index().to_json(orient="records", lines=True))
            elif not self.json_output:
                print(obj)

        run_results = []
        start_t = time.perf_counter()
        run_results = []
        for config in self.configs:
            _json_print(str(config))
            if self.collect_only:
                continue
            _json_print(f"Running {config.key}")
            run_result = config.run(self)
            # Pretty print run results table
            pd.set_option("display.float_format", "{:.2f}".format)  # pylint: disable=consider-using-f-string
            run_summaries_df_list = (run_result.run_summary_detailed_df_list
                                     if self.detailed_output else run_result.run_summary_df_list)
            _json_print(f"Total number of runs: {len(run_summaries_df_list)}")
            for idx, summary_df in enumerate(run_summaries_df_list):
                _json_print(f"Run: {idx}")
                _json_print(summary_df)
            _json_print(f"Run time: {round(run_result.run_time, 3)} seconds.")
            run_results.append(run_result)
            if self.reports:
                run_result.build_report(reports_folder=self.reports, tag=self.tag)
        if self.junit_report and self.reports:
            with open(os.path.join(self.reports, "triton-benchmarks.xml"), "w", encoding="utf-8") as junit_rep_file:
                junit_rep_file.write(_junit_report(run_results))
        end_t = time.perf_counter()
        _json_print(f"Total run time including overhead: {round(end_t-start_t, 3)} seconds.")

    @classmethod
    def _get_all_configs(cls) -> Dict[str, BenchmarkConfig]:
        return {config.key: config for config in CONFIGS}

    @classmethod
    def _get_template_configs(
        cls,
        config_filter: Set[str],
        categories_filter: Set[str],
    ) -> List[BenchmarkConfig]:
        known_configs = cls._get_all_configs()
        selected_config_keys = config_filter
        unknown_configs = selected_config_keys - known_configs.keys()
        if unknown_configs:
            raise NotImplementedError(f"Unknown configs are provided {unknown_configs}")
        selected_configs = [
            known_configs[key]
            for key in selected_config_keys
            if {category.value
                for category in known_configs[key].categories}.issubset(categories_filter)
        ]
        if not selected_configs:
            raise AssertionError(f"No configs are selected from {config_filter} category {categories_filter}")
        return selected_configs

    @classmethod
    def _get_configs(
        cls,
        configs_filter: Set[str],
        categories_filter: Set[str],
        providers_filter: List[str],
        shape_pattern: Optional[ShapePatternParser],
    ) -> List[BenchmarkConfigRunResult]:
        template_configs = cls._get_template_configs(
            config_filter=configs_filter,
            categories_filter=categories_filter,
        )
        return [
            BenchmarkConfigRunResult(
                **{key: value
                   for key, value in asdict(template_config).items()
                   if key != "providers_filter"},
                providers_filter=providers_filter,
                shape_pattern=shape_pattern,
            )
            for template_config in template_configs
        ]

    @classmethod
    def _from_args(
        cls,
        configs_filter: Set[str],
        categories_filter: Set[str],
        providers_filter: List[str],
        shape_pattern: Optional[ShapePatternParser],
        **kwargs: Any,
    ) -> BenchmarkConfigs:
        configs = cls._get_configs(
            configs_filter=configs_filter,
            categories_filter=categories_filter,
            providers_filter=providers_filter,
            shape_pattern=shape_pattern,
        )
        cls_init_fields = {obj_field.name for obj_field in fields(cls) if obj_field.init}
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in cls_init_fields}
        return cls(
            configs=configs,
            **filtered_kwargs,
        )

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]]) -> argparse.Namespace:

        categories = [cat.value for cat in BenchmarkCategory]

        class _CustomHelpFormatter(argparse.HelpFormatter):

            def __init__(self, prog):
                super().__init__(prog, max_help_position=40)
                self._action_max_length = 25

        def _add_reporting_opts(run_parser: argparse.ArgumentParser):
            reporting_group = run_parser.add_argument_group("Reporting options")
            reporting_group.add_argument(
                "--reports", type=str, default="", help=
                ("Path to the folder to save the reports,"
                 "if provided csv perfromance report for each bencmark ('<benchmark_config>.xml') will be generated and saved."
                 ))
            reporting_group.add_argument(
                "--junit-report",
                action="store_true",
                help="Save run results report('triton_benchmark.xml') in junit format in the reports folder.",
            )
            reporting_group.add_argument(
                "--tag",
                type=str,
                default="",
                required=False,
                help="How to tag results in the perfromance reports.",
            )

        def _add_run_opts(cmd_with_run_opts_parser: argparse.ArgumentParser):
            run_opts_group = cmd_with_run_opts_parser.add_argument_group("Run options")
            run_opts_group.add_argument(
                "--n_runs",
                type=int,
                default=1,
                help=
                ("Number of re-runs for this benchmark. The default is one."
                 "Each run includes n warmup executions for each selected provider/shape and k measured executions for each selected provider/shape."
                 ),
            )

        def _add_filter_opts(cmd_with_filters_parser: argparse.ArgumentParser):
            filters_group = cmd_with_filters_parser.add_argument_group("Filter options")
            filters_group.add_argument(
                "--provider",
                action="append",
                dest="providers_filter",
                metavar="PROVIDER",
                help=("Run benchmark with the selected provider only. This option can be executed multiple times. "
                      "If not set all providers will be benchmarked in a given context - "
                      "hw capabilities, provider support for a specific set of benchmark config params, etc."),
            )
            filters_group.add_argument(
                "--shape-pattern",
                dest="shape_pattern",
                metavar="SHAPE_PATTERN",
                default=None,
                type=str,
                help=("Limit benchmark run to a certain shape or shape pattern"),
            )

        def _add_output_opts(cmd_with_outputs_parser: argparse.ArgumentParser):
            output_group = cmd_with_outputs_parser.add_argument_group("Output options")
            output_group.add_argument(
                "--show-details",
                action="store_true",
                dest="detailed_output",
                help=("Show detailed outputs.\n"
                      "For config summary - info on providers, categories, shape fields. "
                      "Default shows only config[<shape>]\n"
                      "For the detailed run results - memory throughtput and coefficient of variation between runs."
                      "Default shows only compute performance\n"),
            )
            output_group.add_argument(
                "--json",
                action="store_true",
                dest="json_output",
                help="Show outputs in json format",
            )

        def _add_additional_filters_for_all(all_parser: argparse.ArgumentParser):
            filters_group_for_all = all_parser.add_argument_group("Filter options")
            filters_group_for_all.add_argument(
                "--config",
                choices=cls._get_all_configs().keys(),
                action="append",
                dest="configs_filter",
                metavar="CONFIG",
                default=[],
                help=
                (f"Filter ALL configs to the provided list and run COMMAND. Known configs are {cls._get_all_configs().keys()}"
                 "Run `describe ALL` to get the supported configs details (shapes, providers, etc)."),
            )
            categories_str = ", ".join(categories)
            filters_group_for_all.add_argument(
                "--category",
                action="append",
                dest="categories_filter",
                choices=categories,
                default=categories,
                metavar="CATEGORY",
                help=str(
                    "Filter ALL configs to the ones in the provided category list and run COMMAND."
                    f"Known categories are {categories_str}.", ),
            )

        def _add_argument_groups(
            command_or_benchmark_parser: argparse.ArgumentParser,
            run_command: bool = False,
        ):
            if run_command:
                _add_run_opts(command_or_benchmark_parser)
            _add_output_opts(command_or_benchmark_parser)
            _add_filter_opts(command_or_benchmark_parser)
            if run_command:
                _add_reporting_opts(command_or_benchmark_parser)

        def _add_configs_as_subcommands(
            subparser: argparse.ArgumentParser,
            run_command: bool = False,
        ):
            configs_subparser = subparser.add_subparsers(
                title="Subcommands",
                required=True,
                metavar="BENCHMARK",
                dest="benchmark",
                help="Select BENCHMARK to run",
            )
            all_parser = configs_subparser.add_parser(
                "ALL",
                help="Select all benchmarks",
                formatter_class=_CustomHelpFormatter,
            )
            _add_argument_groups(all_parser, run_command=run_command)
            _add_additional_filters_for_all(all_parser)
            for key, config in cls._get_all_configs().items():
                _add_argument_groups(
                    configs_subparser.add_parser(
                        key,
                        # FIXME: Add supported config providers to the help message
                        help=config.description,
                        formatter_class=_CustomHelpFormatter,
                    ),
                    run_command=run_command,
                )

        parser = argparse.ArgumentParser(
            conflict_handler="resolve",
            formatter_class=_CustomHelpFormatter,
        )
        subparsers = parser.add_subparsers(
            title="Commands",
            required=True,
            metavar="COMMAND",
            dest="command",
            help="Run triton-benchmarks COMMAND -h to get help on the COMMAND specific supported options",
        )
        run_subparser = subparsers.add_parser(
            "run",
            conflict_handler="resolve",
            formatter_class=_CustomHelpFormatter,
            help="Run selected BENCHMARK(s).",
        )
        # Same standard benchmark argument groups are added to command and subcommand to improve UX
        _add_argument_groups(run_subparser, True)
        _add_configs_as_subcommands(run_subparser, True)
        describe_subparser = subparsers.add_parser(
            "describe",
            conflict_handler="resolve",
            formatter_class=_CustomHelpFormatter,
            help="Describe BENCHMARK(s) configuration - supported providers, specific shapes to run, etc.",
        )
        # Same standard benchmark argument groups are added to command and subcommand to improve UX
        _add_argument_groups(describe_subparser)
        _add_configs_as_subcommands(describe_subparser)
        args = parser.parse_args(argv)

        args.collect_only = args.command == "describe"
        args.categories_filter = args.categories_filter if hasattr(args, "categories_filter") else categories
        args.reports = args.reports if hasattr(args, "reports") else ""
        args.junit_report = args.junit_report if hasattr(args, "junit_report") else False

        if args.shape_pattern:
            args.shape_pattern = ShapePatternParser(args.shape_pattern)

        if args.benchmark == "ALL":
            args.configs_filter = cls._get_all_configs().keys()
        elif args.benchmark in cls._get_all_configs().keys():
            args.configs_filter = set([args.benchmark])
        else:
            raise NotImplementedError(f"Unknown config {args.benchmark}")

        if not args.configs_filter:
            raise ValueError("No configs are provided")

        if not args.reports and args.junit_report:
            raise ValueError("To generate junit report provide the reports folder location via --reports option")

        args.configs_filter = ({args.configs_filter}
                               if isinstance(args.configs_filter, str) else set(args.configs_filter))

        return args

    @classmethod
    def from_args(cls, argv: Optional[List[str]] = None) -> BenchmarkConfigs:
        args = cls._parse_args(argv)
        return cls._from_args(**vars(args), )


def main():
    BenchmarkConfigs.from_args().run()


if __name__ == "__main__":
    main()
