"""The Report handed to sinks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from benchmark_monitor.model import Verdict


@dataclass(frozen=True)
class Report:
    """Verdicts for one evaluation night, an optional suppression view and free-text warnings.

    ``suppression`` is the pipeline's alert-suppression result when there is one (see ``suppress``);
    ``None`` means the report lists raw gate verdicts only.
    """

    as_of: date
    verdicts: tuple[Verdict, ...] = ()
    suppression: Any = None
    warnings: tuple[str, ...] = ()
