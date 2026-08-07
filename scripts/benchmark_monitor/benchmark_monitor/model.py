"""Layer 0 domain model — pure dataclasses with no dependencies beyond stdlib.

Defines the shared vocabulary (data structures, enums, constants, and parsing
helpers) used by all other layers of the benchmark_monitor package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GpuPlatform(str, Enum):
    """Supported GPU platform identifiers."""

    PVC = "pvc"
    BMG = "bmg"

    @classmethod
    def from_runner_label(cls, label: str) -> str | None:
        """Map a CI runner label to a platform string, or None if unknown."""
        return RUNNER_TO_GPU.get(label)

    @classmethod
    def short_name(cls, gpu: str) -> str:
        """Return the human-readable short name for a GPU platform string."""
        return GPU_SHORT_NAMES.get(gpu, gpu.upper())

    @classmethod
    def detect(cls, gpu_device: str) -> str | None:
        """Detect the platform from a GPU device description string.

        Returns the platform string ('pvc' or 'bmg'), or None if unrecognized.
        """
        for keywords, platform in _GPU_PLATFORM_RULES:
            for kw in keywords:
                if kw in gpu_device:
                    return platform
        return None


# ---------------------------------------------------------------------------
# Constants / mappings
# ---------------------------------------------------------------------------

RUNNER_TO_GPU: dict[str, str] = {
    "max1550": "pvc",
    "b580": "bmg",
}

GPU_SHORT_NAMES: dict[str, str] = {
    "pvc": "PVC",
    "bmg": "BMG",
}

# Rules for detecting GPU platform from a device description string.
# Each entry is (list-of-keywords, platform-value).
_GPU_PLATFORM_RULES: list[tuple[list[str], str]] = [
    (["Max 1550", "Max 1100"], GpuPlatform.PVC.value),
    (["B580", "BMG"], GpuPlatform.BMG.value),
]


def detect_platform(gpu_device: str) -> str | None:
    """Backward-compatible alias for ``GpuPlatform.detect()``."""
    return GpuPlatform.detect(gpu_device)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThresholdConfig:
    """Per-benchmark threshold settings.

    Defaults are aligned with thresholds.yaml.
    """

    min_history: int = 8
    rolling_window: int = 20
    z_threshold: float = 3.0
    min_drop_pct: float = 5.0
    improvement_lock_pct: float = 8.0
    max_cv: float = 0.15


@dataclass
class DetectionConfig:
    """Full threshold configuration with per-benchmark overrides."""

    defaults: ThresholdConfig = field(default_factory=ThresholdConfig)
    overrides: dict[str, ThresholdConfig] = field(default_factory=dict)

    def for_benchmark(self, benchmark: str) -> ThresholdConfig:
        """Return the threshold config for *benchmark*, falling back to defaults."""
        return self.overrides.get(benchmark, self.defaults)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEntry:
    """A single benchmark run recorded in history."""

    run_id: str
    datetime: str
    tag: str
    commit_sha: str
    agama_version: str = ""
    libigc1_version: str = ""
    results: dict[str, dict[str, float]] = field(default_factory=dict)

    def driver_changes_from(self, baseline: BenchmarkEntry) -> list[dict[str, str]] | None:
        """Compare driver versions against *baseline* entry.

        Returns a list of per-field changes, or ``None`` if nothing changed.
        """
        changes: list[dict[str, str]] = []
        for field_name in ("agama_version", "libigc1_version"):
            cur = getattr(self, field_name, "")
            base = getattr(baseline, field_name, "")
            if cur != base:
                changes.append({"field": field_name, "from": base, "to": cur})
        return changes or None


@dataclass
class BenchmarkHistory:
    """Historical benchmark data for a single GPU."""

    gpu: str
    entries: list[BenchmarkEntry] = field(default_factory=list)


@dataclass
class MetricResult:  # pylint: disable=too-many-instance-attributes
    """Result of analysing a single metric key against its baseline."""

    key: str
    benchmark: str
    params: str
    current_tflops: float
    baseline_median: float
    change_pct: float
    modified_z: float
    driver_change: list[dict[str, str]] | None = None


@dataclass
class AnalysisResult:
    """Aggregated analysis report for one GPU."""

    regressions: list[MetricResult] = field(default_factory=list)
    improvements: list[MetricResult] = field(default_factory=list)
    skipped: int = 0
    total_checked: int = 0
    driver_change: Any = None


# ---------------------------------------------------------------------------
# Key parsing helpers
# ---------------------------------------------------------------------------


def parse_benchmark_name(key: str) -> str:
    """Extract the benchmark name (first segment before '/')."""
    return key.split("/", 1)[0]


def parse_params(key: str) -> str:
    """Extract the params portion (everything after the second '/')."""
    parts = key.split("/", 2)
    return parts[2] if len(parts) > 2 else ""
