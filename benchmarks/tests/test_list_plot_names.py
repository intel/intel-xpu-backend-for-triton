import re
import sys
from pathlib import Path

MICRO = Path(__file__).resolve().parents[1] / "micro_benchmarks"
sys.path.insert(0, str(MICRO))
import list_plot_names  # noqa: E402  # pylint: disable=wrong-import-position

DESCRIPTOR_LOAD = MICRO / "core_ops" / "descriptor_load.py"


def test_plot_names_match_run_benchmarks():
    names = list_plot_names.plot_names(str(DESCRIPTOR_LOAD))
    src = DESCRIPTOR_LOAD.read_text()
    # benchmark_<fn>.run(...) calls in run_benchmarks() must correspond 1:1 to the plot names.
    run_calls = re.findall(r"^    benchmark_(\w+)\.run\(", src, re.M)
    assert len(names) == len(run_calls) > 0
    assert [n.replace("-", "_") for n in names] == run_calls
