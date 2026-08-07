"""Tests for benchmark_monitor.storage — history backend implementations."""
# pylint: disable=too-few-public-methods,duplicate-code

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_monitor.model import BenchmarkEntry, BenchmarkHistory
from benchmark_monitor.storage import (
    DatabaseBackend,
    JsonFileBackend,
    ParquetBackend,
    _dedup_and_prune,
    MAX_HISTORY_ENTRIES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(run_id: str, tflops: float = 10.0, tag: str = "ci") -> BenchmarkEntry:
    return BenchmarkEntry(
        run_id=run_id,
        datetime="2025-01-01T00:00:00",
        tag=tag,
        commit_sha="abc123",
        agama_version="1.0",
        libigc1_version="2.0",
        results={"gemm/triton/{M:1024}": {"tflops": tflops}},
    )


# ===================================================================
# JsonFileBackend
# ===================================================================


class TestJsonFileBackendLoad:

    def test_load_missing_file(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path)
        hist = backend.load("pvc")
        assert hist.gpu == "pvc"
        assert not hist.entries

    def test_load_non_list_json(self, tmp_path: Path):
        """A valid JSON file that is not a list should return empty history."""
        gpu_dir = tmp_path / "pvc"
        gpu_dir.mkdir()
        (gpu_dir / "history.json").write_text('{"key": "value"}')
        backend = JsonFileBackend(tmp_path)
        hist = backend.load("pvc")
        assert not hist.entries

    def test_load_empty_list(self, tmp_path: Path):
        gpu_dir = tmp_path / "pvc"
        gpu_dir.mkdir()
        (gpu_dir / "history.json").write_text("[]")
        backend = JsonFileBackend(tmp_path)
        hist = backend.load("pvc")
        assert not hist.entries


class TestJsonFileBackendSave:

    def test_save_creates_dirs(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path)
        entry = _make_entry("run-1")
        hist = BenchmarkHistory(gpu="pvc", entries=[entry])
        backend.save("pvc", hist)
        assert (tmp_path / "pvc" / "history.json").exists()

    def test_roundtrip(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path)
        entries = [_make_entry(f"run-{i}") for i in range(3)]
        hist = BenchmarkHistory(gpu="pvc", entries=entries)
        backend.save("pvc", hist)

        loaded = backend.load("pvc")
        assert len(loaded.entries) == 3
        assert loaded.entries[0].run_id == "run-0"
        assert loaded.entries[2].results["gemm/triton/{M:1024}"]["tflops"] == 10.0

    def test_roundtrip_preserves_all_fields(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path)
        entry = BenchmarkEntry(
            run_id="r1",
            datetime="2025-06-01",
            tag="pr-42",
            commit_sha="deadbeef",
            agama_version="1.2.3",
            libigc1_version="4.5.6",
            results={"key": {"tflops": 42.5}},
        )
        hist = BenchmarkHistory(gpu="bmg", entries=[entry])
        backend.save("bmg", hist)
        loaded = backend.load("bmg")
        e = loaded.entries[0]
        assert e.run_id == "r1"
        assert e.tag == "pr-42"
        assert e.agama_version == "1.2.3"
        assert e.libigc1_version == "4.5.6"


class TestJsonFileBackendListGpus:

    def test_list_gpus(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path)
        for gpu in ("pvc", "bmg"):
            entry = _make_entry("run-1")
            hist = BenchmarkHistory(gpu=gpu, entries=[entry])
            backend.save(gpu, hist)

        gpus = backend.list_gpus()
        assert "pvc" in gpus
        assert "bmg" in gpus

    def test_list_gpus_empty_dir(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path)
        assert not backend.list_gpus()

    def test_list_gpus_nonexistent_dir(self, tmp_path: Path):
        backend = JsonFileBackend(tmp_path / "nonexistent")
        assert not backend.list_gpus()

    def test_list_gpus_ignores_files(self, tmp_path: Path):
        (tmp_path / "not-a-dir.txt").write_text("junk")
        gpu_dir = tmp_path / "pvc"
        gpu_dir.mkdir()
        (gpu_dir / "history.json").write_text("[]")
        backend = JsonFileBackend(tmp_path)
        assert backend.list_gpus() == ["pvc"]

    def test_list_gpus_ignores_dirs_without_history(self, tmp_path: Path):
        (tmp_path / "pvc").mkdir()  # No history.json
        backend = JsonFileBackend(tmp_path)
        assert not backend.list_gpus()


# ===================================================================
# ParquetBackend
# ===================================================================


class TestParquetBackend:

    def test_load_missing_file(self, tmp_path: Path):
        backend = ParquetBackend(tmp_path)
        hist = backend.load("pvc")
        assert hist.gpu == "pvc"
        assert not hist.entries

    def test_roundtrip(self, tmp_path: Path):
        backend = ParquetBackend(tmp_path)
        entries = [_make_entry(f"run-{i}", tflops=float(i + 1)) for i in range(5)]
        hist = BenchmarkHistory(gpu="pvc", entries=entries)
        backend.save("pvc", hist)

        loaded = backend.load("pvc")
        assert len(loaded.entries) == 5
        assert loaded.entries[0].run_id == "run-0"
        assert loaded.entries[4].results["gemm/triton/{M:1024}"]["tflops"] == 5.0

    def test_list_gpus(self, tmp_path: Path):
        backend = ParquetBackend(tmp_path)
        for gpu in ("pvc", "bmg"):
            entry = _make_entry("run-1")
            hist = BenchmarkHistory(gpu=gpu, entries=[entry])
            backend.save(gpu, hist)
        assert sorted(backend.list_gpus()) == ["bmg", "pvc"]


# ===================================================================
# DatabaseBackend (stub)
# ===================================================================


class TestDatabaseBackend:

    def test_load_raises(self):
        backend = DatabaseBackend("sqlite:///test.db")
        with pytest.raises(NotImplementedError):
            backend.load("pvc")

    def test_save_raises(self):
        backend = DatabaseBackend("sqlite:///test.db")
        with pytest.raises(NotImplementedError):
            backend.save("pvc", BenchmarkHistory(gpu="pvc"))

    def test_list_gpus_raises(self):
        backend = DatabaseBackend("sqlite:///test.db")
        with pytest.raises(NotImplementedError):
            backend.list_gpus()


# ===================================================================
# Dedup and pruning
# ===================================================================


class TestDedupAndPrune:

    def test_dedup_keeps_last(self):
        e1 = _make_entry("run-1", tflops=10.0)
        e2 = _make_entry("run-1", tflops=20.0)  # same run_id, different value
        result = _dedup_and_prune([e1, e2], "pvc")
        assert len(result) == 1
        assert result[0].results["gemm/triton/{M:1024}"]["tflops"] == 20.0

    def test_prune_to_max(self):
        entries = [_make_entry(f"run-{i}") for i in range(MAX_HISTORY_ENTRIES + 50)]
        result = _dedup_and_prune(entries, "pvc")
        assert len(result) == MAX_HISTORY_ENTRIES

    def test_prune_keeps_newest(self):
        entries = [_make_entry(f"run-{i}") for i in range(MAX_HISTORY_ENTRIES + 10)]
        result = _dedup_and_prune(entries, "pvc")
        # Should keep the last MAX_HISTORY_ENTRIES entries
        assert result[0].run_id == "run-10"
        assert result[-1].run_id == f"run-{MAX_HISTORY_ENTRIES + 9}"

    def test_no_dedup_needed(self):
        entries = [_make_entry(f"run-{i}") for i in range(5)]
        result = _dedup_and_prune(entries, "pvc")
        assert len(result) == 5
