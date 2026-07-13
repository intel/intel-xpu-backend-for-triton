#!/usr/bin/env python3
"""Compare triton-td GB/s across the three 3D producer/reducer solutions.

Sister to ``plot_triton_td_comparison.py``. Defaults to comparing the
baseline, metadata-buffer, and sentinel runs, annotated with the actual
``TILE_SIZE``, ``BLOCK_Q``, ``num_warps``, and ``num_stages`` used by each run.
Each run is a ``<root>/<name>/<dtype>/`` directory holding:

    - ``unified-attention-performance-td_0.csv``     (perf, triton-td-GB/s)
    - ``unified-attention-autotune-decisions.csv``   (actual config values)

Use ``--baseline`` to choose the run used for ratio summaries. If omitted, the
first run in ``--runs`` is the baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONFIG_COLS = [
    "q_heads",
    "k_heads",
    "head_size",
    "seq_lens",
    "sliding_window",
    "soft_cap",
    "num_blocks",
    "block_size",
]

# Default solution runs. Keep baseline blue / sentinel green consistent with
# plot_triton_td_comparison.py; metadata-buffer gets a distinct red.
DEFAULT_RUNS: list[tuple[str, str]] = [
    ("baseline", "#4C78A8"),
    ("buffer", "#E45756"),
    ("sentinel", "#54A24B"),
]

# Fallback palette for any --runs entry that omits an explicit color.
PALETTE = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#B279A2", "#9D755D"]

PARAMS = [
    ("TILE_SIZE", "TS"),
    ("BLOCK_Q", "BQ"),
    ("num_warps", "NW"),
    ("num_stages", "NS"),
]

SEQ_LEN_LABELS = {
    "[(1, 257)]": "single_257",
    "[(1, 255), (1, 256), (1, 257), (1, 511), (1, 512), (1, 513), (1, 1023), (1, 1024)]": "boundary_8",
    "[(1, 1513), (1, 4100), (1, 530), (1, 123), (1, 4803), (1, 434), (1, 3015), (1, 34)]": "random_8",
}


def short_seq_lens(s: str) -> str:
    if s in SEQ_LEN_LABELS:
        return SEQ_LEN_LABELS[s]
    n = s.count("(1,")
    return f"pow2x{n}"


def parse_runs(spec: str | None) -> list[tuple[str, str]]:
    """Parse ``--runs`` as ``name`` or ``name:#hex`` comma-separated entries."""
    if not spec:
        return DEFAULT_RUNS
    runs: list[tuple[str, str]] = []
    for i, raw in enumerate(part.strip() for part in spec.split(",")):
        if not raw:
            continue
        name, _, color = raw.partition(":")
        color = color or PALETTE[i % len(PALETTE)]
        runs.append((name, color))
    if not runs:
        return DEFAULT_RUNS
    return runs


def config_value(config: object, key: str) -> float:
    if pd.isna(config):
        return np.nan
    for part in str(config).split(";"):
        name, _, value = part.partition("=")
        if name == key and value:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return np.nan


def normalize_decision_columns(autotune: pd.DataFrame) -> pd.DataFrame:
    for param, _label in PARAMS:
        actual_col = f"actual_{param}"
        selected_col = f"selected_{param}"
        if actual_col not in autotune.columns:
            if "selected_config" in autotune.columns:
                autotune[actual_col] = autotune["selected_config"].map(lambda config: config_value(config, param))
            elif selected_col in autotune.columns:
                autotune[actual_col] = autotune[selected_col]
            else:
                autotune[actual_col] = np.nan
    return autotune


def filter_decisions_for_perf(autotune: pd.DataFrame, perf_csv: Path) -> pd.DataFrame:
    if "-td" not in perf_csv.name:
        return autotune
    filtered = autotune
    if "td_patched" in filtered.columns:
        filtered = filtered[filtered["td_patched"].astype(str) == "1"]
    if "provider" in filtered.columns and filtered["provider"].astype(str).str.contains("td").any():
        filtered = filtered[filtered["provider"].astype(str).str.contains("td")]
    return filtered


def load_run(perf_csv: Path, autotune_csv: Path) -> pd.DataFrame:
    perf = pd.read_csv(perf_csv)
    perf = perf[CONFIG_COLS + ["triton-td-GB/s"]].copy()
    # The autotune decision artifact is optional: a run that does not record a
    # TILE_SIZE/BLOCK_Q simply shows "?" in the annotation rather than failing.
    if autotune_csv.exists():
        autotune = pd.read_csv(autotune_csv)
        autotune = filter_decisions_for_perf(normalize_decision_columns(autotune), perf_csv)
        value_cols = [f"actual_{param}" for param, _label in PARAMS]
        if set(value_cols).issubset(autotune.columns):
            autotune = autotune[CONFIG_COLS + value_cols].drop_duplicates(CONFIG_COLS)
            perf = perf.merge(autotune, on=CONFIG_COLS, how="left")
    for param, _label in PARAMS:
        actual_col = f"actual_{param}"
        if actual_col not in perf.columns:
            perf[actual_col] = np.nan
    return perf


def config_label(row: pd.Series) -> str:
    sw = "" if pd.isna(row["sliding_window"]) else f" sw={int(row['sliding_window'])}"
    sc = "" if pd.isna(row["soft_cap"]) else f" sc={row['soft_cap']:g}"
    return (
        f"q{int(row['q_heads'])}/k{int(row['k_heads'])} "
        f"h{int(row['head_size'])} "
        f"bs{int(row['block_size'])} "
        f"{short_seq_lens(row['seq_lens'])}"
        f"{sw}{sc}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <root>/triton_td_solution_comparison-<dtype>.png)",
    )
    parser.add_argument("--dtype", default="bf16", help="Subdirectory under each run (bf16/fp8)")
    parser.add_argument(
        "--runs",
        default=None,
        help="Comma list of run names (optionally name:#hex). "
        "First entry is the ratio baseline. Default: baseline,buffer,sentinel",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Run label to use as the ratio baseline. Default: first run in --runs.",
    )
    parser.add_argument(
        "--perf-name",
        default="unified-attention-performance-td_0.csv",
        help="Per-run perf CSV filename",
    )
    parser.add_argument(
        "--autotune-name",
        default="unified-attention-autotune-decisions.csv",
        help="Per-run autotune-decision CSV filename",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    baseline_name = args.baseline or runs[0][0]
    out = args.out or args.root / f"triton_td_solution_comparison-{args.dtype}.png"

    merged = None
    available: list[tuple[str, str]] = []
    for name, color in runs:
        perf_csv = args.root / name / args.dtype / args.perf_name
        if not perf_csv.exists():
            print(f"skipping '{name}': missing {perf_csv}")
            continue
        run = load_run(perf_csv, args.root / name / args.dtype / args.autotune_name)
        run = run.rename(
            columns={
                "triton-td-GB/s": f"triton-td-GB/s_{name}",
                **{f"actual_{param}": f"actual_{param}_{name}" for param, _label in PARAMS},
            }
        )
        merged = run if merged is None else merged.merge(run, on=CONFIG_COLS, how="outer")
        available.append((name, color))

    if merged is None or not available:
        raise SystemExit(f"No run perf CSVs found under {args.root} for dtype={args.dtype}")
    if args.baseline and not any(name == baseline_name for name, _color in available):
        raise SystemExit(f"Baseline run '{baseline_name}' is not available in plotted runs")

    merged["label"] = merged.apply(config_label, axis=1)
    merged = merged.sort_values(CONFIG_COLS).reset_index(drop=True)

    n = len(merged)
    y = np.arange(n)
    n_runs = len(available)
    bar_h = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(14, max(8, 0.32 * n)))

    def format_param_value(value: object) -> str:
        return "?" if pd.isna(value) else f"{int(value)}"

    def add_annotation(bar, fields: list[tuple[str, object]]) -> None:
        width = bar.get_width()
        if pd.isna(width):
            return
        text = " ".join(f"{label}={format_param_value(value)}" for label, value in fields)
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"  {text}",
            va="center",
            ha="left",
            fontsize=4,
        )

    def annotate(bars: list, run_name: str) -> None:
        value_columns = [merged[f"actual_{param}_{run_name}"] for param, _label in PARAMS]
        for row_idx, bar in enumerate(bars):
            fields = [
                (label, values.iloc[row_idx]) for (_param, label), values in zip(PARAMS, value_columns)
            ]
            add_annotation(bar, fields)

    def max_annotation_chars() -> int:
        max_chars = 0
        for name, _color in available:
            for _, row in merged.iterrows():
                max_chars = max(
                    max_chars,
                    sum(
                        len(label) + 1 + len(format_param_value(row[f"actual_{param}_{name}"])) + 1
                        for param, label in PARAMS
                    ),
                )
        return max_chars

    # Bars are stacked vertically per config, centered on each y tick.
    for i, (name, color) in enumerate(available):
        offset = (i - (n_runs - 1) / 2) * bar_h
        bars = ax.barh(
            y + offset,
            merged[f"triton-td-GB/s_{name}"],
            height=bar_h,
            label=name,
            color=color,
        )
        annotate(bars, name)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("triton-td GB/s")
    ax.set_title(
        f"Unified Attention 3D triton-td GB/s: {' vs '.join(name for name, _ in available)} ({args.dtype})\n"
        f"baseline: {baseline_name}; annotation: actual TS/BQ/NW/NS"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    xmax = max(merged[f"triton-td-GB/s_{name}"].max(skipna=True) for name, _ in available)
    ax.set_xlim(0, xmax * (1.05 + 0.011 * max_annotation_chars()))

    # Summary: each non-baseline run compared against the first run over shared configs.
    summary_lines = []
    baseline_col = f"triton-td-GB/s_{baseline_name}"
    have_baseline = baseline_col in merged.columns
    for name, _ in available:
        if name == baseline_name or not have_baseline:
            continue
        paired = merged.dropna(subset=[baseline_col, f"triton-td-GB/s_{name}"])
        if paired.empty:
            continue
        ratio = paired[f"triton-td-GB/s_{name}"] / paired[baseline_col]
        mean_pct = (ratio.mean() - 1.0) * 100
        median_pct = (ratio.median() - 1.0) * 100
        geomean_pct = (np.exp(np.log(ratio).mean()) - 1.0) * 100
        summary_lines.append(
            f"{name} vs {baseline_name} over {len(paired)} configs — "
            f"mean: {mean_pct:+.2f}%   median: {median_pct:+.2f}%   geomean: {geomean_pct:+.2f}%"
        )
    summary = "\n".join(summary_lines)
    if summary:
        fig.text(0.5, 0.01, summary, ha="center", fontsize=10)

    fig.tight_layout(rect=(0, 0.01 + 0.018 * max(len(summary_lines), 1), 1, 1))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    if summary:
        print(summary)
    elif len(available) == 1:
        print(f"single run '{available[0][0]}'; no ratio summary")


if __name__ == "__main__":
    main()
