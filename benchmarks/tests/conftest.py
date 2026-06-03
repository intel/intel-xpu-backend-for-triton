from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from benchmark_helpers import make_cfg, template_for
from triton_kernels_benchmark.benchmark_testing import (
    BenchmarkCategory,
    MarkArgs,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("triton-benchmarks")
    group.addoption(
        "--reports",
        default="",
        help="Directory to write benchmark report artifacts.",
    )
    group.addoption(
        "--tag",
        default="",
        help="Tag propagated to build_report.",
    )
    group.addoption(
        "--n-runs",
        type=int,
        default=1,
        dest="n_runs",
        help="Number of re-runs per test case.",
    )
    group.addoption(
        "--benchmark-brief",
        action="store_true",
        dest="benchmark_brief",
        help="Print only mean values without min, max, CV.",
    )
    group.addoption(
        "--benchmark-eff",
        action="store_true",
        dest="benchmark_eff",
        help="Print HW utilization using gpu_info.json.",
    )
    group.addoption(
        "--describe-only",
        action="store_true",
        dest="describe_only",
        help="Describe each (benchmark, shape, provider) without executing the kernel.",
    )


@pytest.fixture(scope="session")
def benchmark_reports_dir(request: pytest.FixtureRequest) -> Path | None:
    value = request.config.getoption("--reports")
    return Path(value) if value else None


@pytest.fixture(scope="session")
def benchmark_describe_only(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("describe_only"))


@pytest.fixture(scope="session")
def benchmark_mark_args(request: pytest.FixtureRequest, benchmark_reports_dir: Path | None,  # pylint: disable=redefined-outer-name
                        ) -> MarkArgs:
    return MarkArgs(
        reports=str(benchmark_reports_dir) if benchmark_reports_dir else "",
        n_runs=request.config.getoption("n_runs"),
        brief=request.config.getoption("benchmark_brief"),
        eff=request.config.getoption("benchmark_eff"),
    )


_PARTS_SUBDIR = "_parts"


def _iter_case_dirs(parts_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (config_key, case_dir) for each per-test subdir under _parts/."""
    if not parts_dir.is_dir():
        return
    for case_dir in sorted(parts_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        # Directory names follow "<config>__<shape>__<provider>".
        config_key = case_dir.name.split("__", 1)[0]
        yield config_key, case_dir


def _aggregate_reports(reports: Path, tag: str) -> None:
    parts_dir = reports / _PARTS_SUBDIR
    if not parts_dir.is_dir():
        return

    # Group per-test subdirs by config_key.
    dirs_by_config: dict[str, list[Path]] = {}
    for config_key, case_dir in _iter_case_dirs(parts_dir):
        dirs_by_config.setdefault(config_key, []).append(case_dir)

    for config_key, case_dirs in dirs_by_config.items():
        providers = list({d.name.rsplit("__", 1)[-1] for d in case_dirs})
        cfg = make_cfg(template_for(config_key), providers=providers)
        merged_name = f"{cfg.plot_name}.csv"
        frames: list[pd.DataFrame] = []
        for case_dir in case_dirs:
            csv_path = case_dir / merged_name
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            continue
        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined.to_csv(reports / merged_name, index=False)
        cfg.res_df_list = [combined]
        cfg.build_report(reports_folder=str(reports), tag=tag)


def pytest_sessionstart(session: pytest.Session) -> None:
    if hasattr(session.config, "workerinput"):
        return
    reports = session.config.getoption("--reports")
    if not reports:
        return
    parts_dir = Path(reports) / _PARTS_SUBDIR
    if parts_dir.is_dir():
        shutil.rmtree(parts_dir)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # pylint: disable=unused-argument
    # xdist workers have a `workerinput` attribute on their config; only the
    # controller (or a non-xdist run) should aggregate once all tests are done.
    if hasattr(session.config, "workerinput"):
        return
    reports = session.config.getoption("--reports")
    if not reports:
        return
    tag = session.config.getoption("--tag")
    _aggregate_reports(Path(reports), tag)


# Expose BenchmarkCategory so tests can reference the full category set symbolically.
ALL_CATEGORIES = {cat.value for cat in BenchmarkCategory}
