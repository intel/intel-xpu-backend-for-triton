"""Layer 2b history storage — abstracts where/how processed benchmark history is persisted.

Provides a backend interface (``HistoryBackend``) and concrete implementations
for JSON files, Parquet files, and a future database backend.  All backends
operate on the domain types defined in :mod:`benchmark_monitor.model` (Layer 0).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

from benchmark_monitor.model import BenchmarkEntry, BenchmarkHistory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY_ENTRIES = 200
"""Maximum number of entries retained when saving history (oldest pruned first)."""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _dedup_and_prune(entries: list[BenchmarkEntry], gpu: str) -> list[BenchmarkEntry]:
    """Deduplicate entries by *run_id* (keep last) and prune to MAX_HISTORY_ENTRIES.

    Mirrors the logic previously in ``convert_results.convert()`` (lines 218-238).
    """
    seen_run_ids: dict[str, int] = {}
    deduped: list[BenchmarkEntry] = []
    for entry in entries:
        if entry.run_id in seen_run_ids:
            # Replace the earlier occurrence with the newer one.
            idx = seen_run_ids[entry.run_id]
            deduped[idx] = entry
        else:
            seen_run_ids[entry.run_id] = len(deduped)
            deduped.append(entry)

    if len(deduped) > MAX_HISTORY_ENTRIES:
        logger.info(
            "Pruning %s history from %d to %d entries.",
            gpu,
            len(deduped),
            MAX_HISTORY_ENTRIES,
        )
        deduped = deduped[-MAX_HISTORY_ENTRIES:]

    return deduped


def _entry_to_dict(entry: BenchmarkEntry) -> dict:
    """Serialize a BenchmarkEntry to a plain dict suitable for JSON."""
    return asdict(entry)


def _dict_to_entry(d: dict) -> BenchmarkEntry:
    """Deserialize a plain dict (from JSON) into a BenchmarkEntry."""
    return BenchmarkEntry(
        run_id=d.get("run_id", ""),
        datetime=d.get("datetime", ""),
        tag=d.get("tag", ""),
        commit_sha=d.get("commit_sha", ""),
        agama_version=d.get("agama_version", ""),
        libigc1_version=d.get("libigc1_version", ""),
        results=d.get("results", {}),
    )


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class HistoryBackend(ABC):
    """Abstract interface for benchmark history persistence."""

    @abstractmethod
    def load(self, gpu: str) -> BenchmarkHistory:
        """Load the full history for *gpu*.

        Returns an empty ``BenchmarkHistory`` if no data exists yet.
        """

    @abstractmethod
    def save(self, gpu: str, history: BenchmarkHistory) -> None:
        """Persist *history* for *gpu*, applying deduplication and pruning."""

    @abstractmethod
    def list_gpus(self) -> list[str]:
        """Return the list of GPU identifiers that have stored history."""


# ---------------------------------------------------------------------------
# JSON file backend
# ---------------------------------------------------------------------------


class JsonFileBackend(HistoryBackend):
    """Store history as ``<base_dir>/<gpu>/history.json`` files.

    Port of the I/O logic from ``convert_results.read_history()`` and
    ``convert_results.write_history()``.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _history_path(self, gpu: str) -> Path:
        return self.base_dir / gpu / "history.json"

    def load(self, gpu: str) -> BenchmarkHistory:
        path = self._history_path(gpu)
        if not path.exists():
            logger.info("History file %s does not exist; starting fresh.", path)
            return BenchmarkHistory(gpu=gpu)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.warning("History file %s is not a JSON array; starting fresh.", path)
            return BenchmarkHistory(gpu=gpu)

        entries = [_dict_to_entry(d) for d in data]
        return BenchmarkHistory(gpu=gpu, entries=entries)

    def save(self, gpu: str, history: BenchmarkHistory) -> None:
        entries = _dedup_and_prune(history.entries, gpu)

        path = self._history_path(gpu)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialized = [_entry_to_dict(e) for e in entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
            f.write("\n")

        logger.info("Wrote %d entries to %s", len(entries), path)

    def list_gpus(self) -> list[str]:
        if not self.base_dir.is_dir():
            return []
        return sorted(d.name for d in self.base_dir.iterdir() if d.is_dir() and (d / "history.json").exists())


# ---------------------------------------------------------------------------
# Parquet backend
# ---------------------------------------------------------------------------


class ParquetBackend(HistoryBackend):
    """Store history as ``<base_dir>/<gpu>/history.parquet`` files.

    Each ``BenchmarkEntry`` is flattened into a single row with columns:
    ``run_id``, ``datetime``, ``tag``, ``commit_sha``, ``agama_version``,
    ``libigc1_version``, and ``results_json`` (a JSON string of the results dict).

    Requires ``pandas`` and ``pyarrow``.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _history_path(self, gpu: str) -> Path:
        return self.base_dir / gpu / "history.parquet"

    def load(self, gpu: str) -> BenchmarkHistory:
        import pandas as pd  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        path = self._history_path(gpu)
        if not path.exists():
            logger.info("Parquet file %s does not exist; starting fresh.", path)
            return BenchmarkHistory(gpu=gpu)

        df = pd.read_parquet(path)
        entries: list[BenchmarkEntry] = []
        for _, row in df.iterrows():
            results: dict[str, dict[str, float]] = {}
            results_raw = row.get("results_json", "{}")
            if results_raw:
                results = json.loads(results_raw)

            entries.append(
                BenchmarkEntry(
                    run_id=str(row.get("run_id", "")),
                    datetime=str(row.get("datetime", "")),
                    tag=str(row.get("tag", "")),
                    commit_sha=str(row.get("commit_sha", "")),
                    agama_version=str(row.get("agama_version", "")),
                    libigc1_version=str(row.get("libigc1_version", "")),
                    results=results,
                ))

        return BenchmarkHistory(gpu=gpu, entries=entries)

    def save(self, gpu: str, history: BenchmarkHistory) -> None:
        import pandas as pd  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        entries = _dedup_and_prune(history.entries, gpu)

        rows: list[dict] = []
        for entry in entries:
            rows.append({
                "run_id": entry.run_id,
                "datetime": entry.datetime,
                "tag": entry.tag,
                "commit_sha": entry.commit_sha,
                "agama_version": entry.agama_version,
                "libigc1_version": entry.libigc1_version,
                "results_json": json.dumps(entry.results),
            })

        path = self._history_path(gpu)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False, engine="pyarrow")

        logger.info("Wrote %d entries to %s", len(entries), path)

    def list_gpus(self) -> list[str]:
        if not self.base_dir.is_dir():
            return []
        return sorted(d.name for d in self.base_dir.iterdir() if d.is_dir() and (d / "history.parquet").exists())


# ---------------------------------------------------------------------------
# Database backend (stub)
# ---------------------------------------------------------------------------


class DatabaseBackend(HistoryBackend):
    """Placeholder for a future database-backed history store."""

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string

    def load(self, gpu: str) -> BenchmarkHistory:
        raise NotImplementedError("Database backend planned for future release")

    def save(self, gpu: str, history: BenchmarkHistory) -> None:
        raise NotImplementedError("Database backend planned for future release")

    def list_gpus(self) -> list[str]:
        raise NotImplementedError("Database backend planned for future release")
