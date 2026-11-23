"""The Report handed to sinks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from benchmark_monitor.model import Verdict
from benchmark_monitor.suppress import SuppressionResult


@dataclass(frozen=True)
class Report:
    """Verdicts for one evaluation night, the suppression view and free-text warnings.

    ``suppression`` is the pipeline's alert-suppression result (see ``suppress``); ``None`` means the
    report lists raw gate verdicts only.
    """

    as_of: date
    verdicts: tuple[Verdict, ...] = ()
    suppression: SuppressionResult | None = None
    warnings: tuple[str, ...] = ()
