import argparse
import itertools
import os

from triton.testing import assert_close as triton_assert_close, Benchmark

BENCHMARKING_METHOD = os.getenv("BENCHMARKING_METHOD", "UPSTREAM_PYTORCH_PROFILER")
VERIFY = os.getenv("VERIFY", "1") == "1"


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()


def _summarize_statistics(times, quantiles, return_mode):
    import torch
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
    import torch
    import triton
    from triton.testing import do_bench as triton_do_bench

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
    # FIXME: to avoid negative timings before DLE 2025.1;
    # this workaround doesn't work for BMG.
    triton.runtime.driver.active.utils.wait()
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
                                       return_mode="mean", device="xpu", sync_submitting=True):
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
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

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

    profiling_func_filter = filter(lambda x: x.name.startswith("__profile_kernel_of_func"), prof.events())
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
    # FIXME: relaxation for new agama release
    assert len(kernels) == n_repeat - 1, (
        f"the profiling number not match; {n_repeat=}, {kernels=}, \n" +
        f"top functions by xpu_time:\n {prof.key_averages(group_by_stack_n=5).table(sort_by='xpu_time')}")
    # Make the time to the milliseconds.
    times = torch.tensor([sum([k.duration for k in ks]) * 1e-3 for ks in kernels], dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)


if BENCHMARKING_METHOD == "ELAPSED_TIME":
    do_bench = do_bench_elapsed_time
elif BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    do_bench = do_bench_upstream_pytorch_profiler
else:
    raise NotImplementedError(f"BENCHMARKING_METHOD: {BENCHMARKING_METHOD} isn't implemented")


def assert_close(x_fn, y_fn, atol=None, rtol=None, err_msg=""):
    if VERIFY:
        triton_assert_close(x_fn(), y_fn(), atol, rtol, err_msg)


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    # pylint: disable=too-many-branches
    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwrags):
        import matplotlib.pyplot as plt
        import pandas as pd
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
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
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
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        # df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df["Diff"] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ":")
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path="", return_df=False, **kwargs):
        save_path = save_path_from_args(save_path)
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []

        for bench in benchmarks:
            result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))

        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "results.html"), "w", encoding="utf-8") as html:
                html.write("<html><body>\n")
                for bench in benchmarks:
                    html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
                html.write("</body></html>\n")

        if return_df:
            return result_dfs[0] if has_single_bench else result_dfs

        return None


def save_path_from_args(save_path: str):
    """Returns a save path that is specified as an argument or via --reports comman line option."""
    if save_path:
        return save_path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        type=str,
        default="",
        help="directory to save reports",
    )
    args = parser.parse_args()
    return args.reports
