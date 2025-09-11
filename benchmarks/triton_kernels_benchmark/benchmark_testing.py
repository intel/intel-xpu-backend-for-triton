from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Optional, List, Tuple, Union, Set
from collections.abc import Iterable
from enum import Enum
from dataclasses import dataclass, field
import itertools

import argparse
import datetime
import os
import time

import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from triton.testing import assert_close as triton_assert_close, Benchmark, do_bench as triton_do_bench

from triton_kernels_benchmark import build_report
from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser

BENCHMARKING_METHOD = os.getenv("BENCHMARKING_METHOD", "UPSTREAM_PYTORCH_PROFILER")
BENCHMARKING_CONFIG = {
    "verify": os.getenv("VERIFY", "1") == "1",
    "do_prewarmup": os.getenv("PREWARMUP", "1") == "1",
}


def disable_verification():
    BENCHMARKING_CONFIG["verify"] = False


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()


def do_prewarmup(fn, min_seconds=5):
    """Looks like some functions require pre-warmup with minimum time to do the compilation.
    It has to be done once."""
    if not BENCHMARKING_CONFIG["do_prewarmup"]:
        return

    start = time.time()
    while time.time() - start < min_seconds:
        fn()
        synchronize()
    BENCHMARKING_CONFIG["do_prewarmup"] = False


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if times.numel() > 2:
            # exclude max and min times
            times = torch.sort(times).values[1:-1]
        # add coefficient of the variance.
        std = torch.std(times)
        mean = torch.mean(times)
        cv = std / mean
        ret.extend([mean.tolist(), cv.tolist()])
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def do_bench_elapsed_time(fn, n_warmup=25, n_repeat=100, grad_to_none=None, quantiles=None, return_mode="mean",
                          device="xpu"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param n_warmup: Number of repetitions for warmup
    :type n_warmup: int
    :param n_repeat: Number of repetitions to collect measurements
    :type n_repeat: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    """
    assert return_mode in ["min", "max", "mean", "median"]

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=device)

    # Estimate the runtime of the function
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # The cache is also maintained in `triton_do_bench` function,
    # there is no need to duplicate the amount of memory used.
    del cache

    # compute warmup and repeat times
    warmup_time = n_warmup * estimate_ms
    rep_time = n_repeat * estimate_ms

    times = triton_do_bench(fn, warmup=warmup_time, rep=rep_time, grad_to_none=grad_to_none, return_mode="all")
    times = torch.tensor(times, dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_upstream_pytorch_profiler(fn, n_warmup=25, n_repeat=100, grad_to_none=None, quantiles=None,
                                       return_mode="mean", device="xpu", sync_submitting=True, benchmark_label=None):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param n_warmup: Number of repetitions for warmup
    :type n_warmup: int
    :param n_repeat: Number of repetitions to collect measurements
    :type n_repeat: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    """

    assert return_mode in ["min", "max", "mean", "median"]

    fn()
    synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=device)

    # Warm-up
    for _ in range(n_warmup):
        fn()
        # To be consistent with the benchmark measurements
        if sync_submitting:
            synchronize()

    # Benchmark
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            # we clear the L2 cache before each run
            cache.zero_()
            if sync_submitting:
                synchronize()
            # record time of `fn`
            with record_function("__profile_kernel_of_func"):
                fn()
        # Record clocks
        synchronize()

    profiling_func_filter = filter(
        lambda x: x.name.startswith("__profile_kernel_of_func" if benchmark_label is None else benchmark_label),
        prof.events())
    functions = list(profiling_func_filter)

    def extract_kernels(funcs):
        kernels = []
        kernels += list(itertools.chain.from_iterable(map(lambda func: extract_kernels(func.cpu_children), funcs)))
        kernels += list(itertools.chain.from_iterable([func.kernels for func in funcs]))
        return kernels

    kernels = [extract_kernels(func.cpu_children) for func in functions]
    # For example, for backward FA, kernels can be empty for one of the threads.
    # Keep in mind that `backward` function is launched in another thread and
    # requires the use of `record_function` function additionally in its thread
    # for correct registration of kernels.
    # For details: https://github.com/pytorch/pytorch/issues/144778
    kernels = [kernel for kernel in kernels if kernel != []]
    relax_profiling_data_check = os.getenv("TRITON_RELAX_PROFILING_CHECK", "0") == "1"
    if not (len(kernels) >= n_repeat - 1 if relax_profiling_data_check else len(kernels) == n_repeat):
        raise AssertionError(
            f"the profiling number not match; {n_repeat=}, {kernels=}, "
            f"top functions by xpu_time:\n {prof.key_averages(group_by_stack_n=5).table(sort_by='xpu_time')}",
            "You may try to relax profiling check by setting env variable TRITON_RELAX_PROFILING_CHECK=1",
        )
    # Make the time to the milliseconds.
    times = torch.tensor([sum((k.duration for k in ks)) * 1e-3 for ks in kernels], dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)


if BENCHMARKING_METHOD == "ELAPSED_TIME":
    do_bench = do_bench_elapsed_time
elif BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    do_bench = do_bench_upstream_pytorch_profiler
else:
    raise NotImplementedError(f"BENCHMARKING_METHOD: {BENCHMARKING_METHOD} isn't implemented")


def assert_close(x_fn, y_fn, atol=None, rtol=None, err_msg=""):
    if BENCHMARKING_CONFIG["verify"]:
        triton_assert_close(x_fn(), y_fn(), atol, rtol, err_msg)


def filter_providers(
    supported_providers: dict[str, str],
    providers_filter: Optional[list[str]],
) -> dict[str, str]:
    if providers_filter:
        if missing_keys := providers_filter - supported_providers.keys():
            raise AssertionError(f"Unsupported providers are provided in filter {missing_keys}")
        providers = {name: label for name, label in supported_providers.items() if name in providers_filter}
        if not providers:
            raise AssertionError(f"No providers are selected from {supported_providers} for {providers_filter} filter.")
        return providers
    return supported_providers


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


@dataclass
class MarkArgs:
    reports: str = ""
    n_runs: int = 1

    @classmethod
    def _get_parser(cls) -> argparse.ArgumentParser:
        """Parses arguments via CLI, allows save_path overloading to `reports`."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--reports",
            type=str,
            default="",
            help="Path to directory to save reports.",
        )
        parser.add_argument(
            "--n_runs",
            type=int,
            default=1,
            help="Number of runs for this benchmark. The default is one.",
        )
        return parser

    @classmethod
    def from_args(cls) -> "MarkArgs":
        args = cls._get_parser().parse_args()
        return MarkArgs(args.reports, args.n_runs)


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    # pylint: disable=too-many-branches
    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False, run_counter=0,
             save_precision=6, **kwargs):
        y_vals = []
        for label in bench.ylabel:
            y_mean = [f"{x}-{label}" for x in bench.line_names]
            y_min = [f"{x}-{label}-min" for x in bench.line_names]
            y_max = [f"{x}-{label}-max" for x in bench.line_names]
            y_vals += y_mean + y_min + y_max
        y_vals += [f"{x}-CV" for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_vals)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_vals = {}
            for label in itertools.chain(bench.ylabel, ["CV"]):
                row_vals[label] = ([], [], [])
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwargs)
                for i, label in enumerate(itertools.chain(bench.ylabel, ["CV"])):
                    try:
                        y_mean, y_min, y_max = ret[i]
                    except TypeError:
                        y_mean, y_min, y_max = ret[i], None, None
                    row_vals[label][0].append(y_mean)
                    row_vals[label][1].append(y_min)
                    row_vals[label][2].append(y_max)
            rows = []
            for label in bench.ylabel:
                if len(row_vals[label][0]) > 0:
                    rows += row_vals[label][0]
                if len(row_vals[label][1]) > 0:
                    rows += row_vals[label][1]
                if len(row_vals[label][2]) > 0:
                    rows += row_vals[label][2]
            rows += row_vals["CV"][0]
            df.loc[len(df)] = list(x) + rows

        filename = f"{bench.plot_name}_{run_counter}"
        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for label in bench.ylabel:
                for i, y in enumerate(bench.line_names):
                    y = f"{y}-{label}"
                    y_min, y_max = df[y + "-min"], df[y + "-max"]
                    col = bench.styles[i][0] if bench.styles else None
                    sty = bench.styles[i][1] if bench.styles else None
                    ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                    if not y_min.isnull().all() and not y_max.isnull().all():
                        y_min = y_min.astype(float)
                        y_max = y_max.astype(float)
                        ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{filename}.png"))
        # df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df["Diff"] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ":")
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{filename}.csv"), float_format=f"%.{save_precision}f", index=False)
        return df

    def run(self, show_plots=False, print_data=False, return_df=False, save_precision=6, mark_args=None, **kwargs):
        args = MarkArgs().from_args() if mark_args is None else mark_args

        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []

        if args.reports:
            # Create directory if it doesn't exist
            os.makedirs(args.reports, exist_ok=True)

        for bench in benchmarks:
            benchmark_dfs = []
            for run_counter in range(args.n_runs):
                df = self._run(bench, args.reports, show_plots, print_data, run_counter=run_counter, **kwargs)
                df["datetime"] = datetime.datetime.now()
                df["run_counter"] = run_counter + 1
                benchmark_dfs.append(df)

            if args.reports:
                merged_df = pd.concat(benchmark_dfs, axis=0)
                merged_df.to_csv(os.path.join(args.reports, f"{bench.plot_name}.csv"),
                                 float_format=f"%.{save_precision}f", index=False)
            result_dfs.extend(benchmark_dfs)

        if return_df:
            if len(result_dfs) == 1:
                return result_dfs[0]
            return result_dfs

        return None

    def run_constrained(
        self,
        shapes: List[ShapeValue],
        show_plots=False,
        print_data=False,
        save_precision=6,
        mark_args=None,
        **kwargs,
    ) -> pd.DataFrame:
        if not isinstance(self.benchmarks, Benchmark):
            raise NotImplementedError("Only single benchmark is supported")
        x_vals = self.benchmarks.x_vals
        self.benchmarks.x_vals = [shape.dims for shape in shapes]
        res_df = self.run(
            show_plots=show_plots,
            print_data=print_data,
            save_precision=save_precision,
            return_df=True,
            mark_args=mark_args,
            **kwargs,
        )
        self.benchmarks.x_vals = x_vals
        return [res_df] if isinstance(res_df, pd.DataFrame) else res_df


class BenchmarkCategory(Enum):
    SOFTMAX = "softmax"
    GEMM = "gemm"
    FLASH_ATTENTION = "flash_attention"
    PREFIX_SUMS = "prefix_sums"
    CORE = "core"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"


DimValue = Union[int, str, bool]
DimValues = Union[DimValue, List[DimValue], Tuple[DimValue, ...]]


@dataclass
class ShapeValue:
    dims: DimValues
    matcher: Optional[ShapePatternParser]

    @property
    def matches_pattern(self) -> bool:
        return self.matcher(str(self)) if self.matcher else True

    def to_list(self) -> List[Union[str, int, bool]]:
        values = self.dims
        return values if isinstance(values, list) else list(values) if isinstance(values, tuple) else [values]

    def _to_iterable(self) -> Iterable[Union[str, int, bool]]:
        return self.dims if isinstance(self.dims, Iterable) else [self.dims]

    def __str__(self):
        return "[" + "-".join([str(value) for value in self._to_iterable()]) + "]"

    @classmethod
    def from_vals(
        cls,
        x_vals: List[DimValues],
        pattern_matcher: Optional[ShapePatternParser],
    ) -> List[ShapeValue]:
        return [ShapeValue(x_val, pattern_matcher) for x_val in x_vals]


@dataclass
class _BenchmarkSummary(ABC):
    shape_pattern: Optional[ShapePatternParser] = field(default=None, init=True)

    memory_metric: ClassVar[str] = "GB/s"
    compute_metric: ClassVar[str] = "TFlops"

    @property
    @abstractmethod
    def _benchmark(self) -> Benchmark:
        pass

    @property
    @abstractmethod
    def _full_benchmark(self) -> Benchmark:
        pass

    @property
    def plot_name(self) -> str:
        return self._benchmark.plot_name

    @property
    def supported_shapes(self) -> List[ShapeValue]:
        return ShapeValue.from_vals(self._benchmark.x_vals, self.shape_pattern)

    @property
    def shape_dimensions(self):
        return self._benchmark.x_names

    @property
    def selected_shapes(self) -> List[ShapeValue]:
        return [shape for shape in self.supported_shapes if shape.matches_pattern]

    @property
    def shapes(self) -> List[str]:
        return [str(shape) for shape in self.supported_shapes if shape.matches_pattern]

    @property
    def supported_providers(self) -> Dict[str, str]:
        return dict(zip(self._full_benchmark.line_vals, self._full_benchmark.line_names))

    @property
    def selected_providers(self) -> Dict[str, str]:
        return dict(zip(self._benchmark.line_vals, self._benchmark.line_names))

    @property
    def primary_metric(self) -> str:
        known_metrics = [self.memory_metric, self.compute_metric]
        for metric in self.perf_metrics:
            if metric not in known_metrics:
                raise NotImplementedError(f"Unsupported {metric} metric. Known metrics are {known_metrics}")
        return self.compute_metric

    @property
    def perf_metrics(self) -> str:
        return self._benchmark.ylabel

    @property
    def reference_provider(self) -> Optional[str]:
        all_providers = list(self.selected_providers.keys())
        non_triton_providers = [provider for provider in all_providers if not provider.startswith("triton")]
        if "xetla" in non_triton_providers:
            return "xetla"
        if "onednn" in non_triton_providers:
            return "onednn"
        if len(all_providers) < 2:
            return None
        if len(non_triton_providers) > 0:
            return non_triton_providers[0]
        return all_providers[1]


@dataclass
class BenchmarkRunResult(_BenchmarkSummary, ABC):
    res_df_list: Optional[List[pd.DataFrame]] = field(default=None, init=False)
    run_time: Optional[float] = field(default=None, init=False)

    @property
    def run_summary_detailed_df_list(self) -> List[pd.DataFrame]:
        return [self._run_summary_df(res_df.copy()) for res_df in self.res_df_list]

    @property
    def run_summary_df_list(self) -> List[pd.DataFrame]:
        return [self._run_summary_df(res_df.copy(), True) for res_df in self.res_df_list]

    def _run_summary_df(self, res_df: pd.DataFrame, primary_metric_only: bool = False) -> pd.DataFrame:

        def _keep_column(column: str) -> bool:
            return (any(column.startswith(p + "-") for p in providers) and not column.endswith(("-min", "-max"))
                    and (not primary_metric_only or column.endswith(self.primary_metric)))

        shape_fields = self.shape_dimensions
        shape_name = "[" + "-".join(shape_fields) + "]"
        shape_fields_w_floats = [
            float_col for float_col in res_df.select_dtypes(include="float").columns if float_col in shape_fields
        ]
        for shape_field_w_floats in shape_fields_w_floats:
            res_df[shape_field_w_floats] = res_df[shape_field_w_floats].astype(int)
        res_df[shape_name] = "[" + res_df[shape_fields].astype(str).agg("-".join, axis=1) + "]"
        res_df.drop(
            columns=shape_fields + ["run_counter", "datetime"] +
            [c for c in res_df.columns if c.endswith(("-min", "-max", "-CV"))], )
        providers = self.selected_providers.values()
        metric_cols = [column for column in res_df.columns if _keep_column(column)]
        column_tuples = [tuple(col.split("-", 1)) for col in metric_cols]
        metrics_df = res_df[metric_cols].copy()
        metrics_df.index = res_df[shape_name]
        metrics_df.index.name = shape_name
        metrics_df.columns = pd.MultiIndex.from_tuples(column_tuples, names=["provider", "metric"])
        geomeans_df = pd.DataFrame(
            [metrics_df.select_dtypes(include="number").apply(scipy.stats.gmean)],
            index=["GeoMean"],
        )
        summary_df = pd.concat([metrics_df, geomeans_df])

        ref_provider = self.reference_provider if self.reference_provider else ""
        metric = self.primary_metric
        ref_provider_name = next((provider for provider in providers if provider.lower() == ref_provider.lower()), None)
        if ref_provider and primary_metric_only:
            non_ref_providers = [provider for provider in providers if provider.lower() != ref_provider.lower()]
            for non_ref_provider in non_ref_providers:
                summary_df[(non_ref_provider, "%diff")] = (
                    (100 *
                     (summary_df[(non_ref_provider, metric)] / summary_df[(ref_provider_name, metric)])).round(1).map(
                         "{:.1f}%".format)  # pylint: disable=C0209
                )
        return summary_df


@dataclass
class BenchmarkConfig:
    key: str
    get_benchmark: Callable[..., Mark]
    categories: Set[BenchmarkCategory]
    description: str = ""
    providers_filter: Optional[list[str]] = None
    run_opts: Dict[str, Union[str, bool, List[str]]] = field(default_factory=dict)

    def _get_benchmark(self, apply_providers_filter=True) -> Mark:
        run_opts = self.run_opts
        if self.providers_filter and apply_providers_filter:
            run_opts = run_opts | {"providers_filter": self.providers_filter}
        mark = self.get_benchmark(**run_opts)
        if not isinstance(mark.benchmarks, Benchmark):
            raise NotImplementedError(
                "Benchmark config list is not supported, exactly one benchmark config is expected.")
        return mark


@dataclass
class BenchmarkConfigRunResult(BenchmarkRunResult, BenchmarkConfig):

    @property
    def shapes_repr(self) -> List[str]:
        return [f"{self.key}{shape}" for shape in self.shapes]

    @property
    def _benchmark(self) -> Benchmark:
        return self._get_benchmark().benchmarks

    @property
    def _full_benchmark(self) -> Benchmark:
        return self._get_benchmark(False).benchmarks

    def __str__(self) -> str:

        def _shapes_repr(shapes: List[ShapeValue]):
            return ", ".join([self.key + str(shape) for shape in shapes])

        str_repr = [
            f"Config: {self.key}",
            f"Config categories: {[category.value for category in self.categories]}",
            f"Run options: {self.run_opts}",
            f"Shape dimensions: {self.shape_dimensions}",
            f"Shapes pattern: {str(self.shape_pattern) if str(self.shape_pattern) else 'Not set'}",
            f"Supported shapes: {_shapes_repr(self.supported_shapes)}",
            f"Selected shapes: {_shapes_repr(self.supported_shapes)}",
            f"Supported providers: {self.supported_providers}",
            f"Selected providers: {self.selected_providers}",
        ]
        return "\n".join(str_repr)

    def run(self, args: MarkArgs) -> BenchmarkConfigRunResult:
        start_time = time.perf_counter()
        # FIXME: Eliminate mark_args argument
        # This is useful for unit testing, composite becnhmark configs and results caching
        self.res_df_list = (self.get_benchmark().run_constrained(shapes=self.selected_shapes, mark_args=args)
                            if self.res_df_list is None else self.res_df_list)
        self.run_time = time.perf_counter() - start_time
        return self

    def build_report(self, reports_folder: str, tag: str = ""):
        res_df = pd.concat(self.res_df_list, axis=0, ignore_index=True)
        shape_cols_for_report_builder = [
            column for column in res_df.select_dtypes(include=["number", "bool"]).columns
            if column in self.shape_dimensions
        ]
        for provider_key, provider_label in self.selected_providers.items():
            report_args = build_report.PassedArgs(
                source=f"{reports_folder}/{self.plot_name}.csv",
                target=f"{reports_folder}/{self.key}-{provider_key}-report.csv",
                param_cols=",".join(shape_cols_for_report_builder),
                benchmark=self.key,
                compiler=str(provider_key),
                tflops_col=f"{provider_label}-{self.compute_metric}",
                hbm_col=f"{provider_label}-{self.memory_metric}",
                tag=tag,
                mask=False,
            )
            build_report.build_report(report_args, res_df)
