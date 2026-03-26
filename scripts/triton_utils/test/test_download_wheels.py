"""Tests for the download_wheels subcommand."""
# pylint: disable=W0212,C1803
from __future__ import annotations

from pathlib import Path

import pytest

import triton_utils
from triton_utils.gh_utils import GHAWheelDownloader, GHArtifact


def test_config_parse_minimal():
    """Minimal invocation: triton-utils wheels -D /tmp/wheels"""
    config = triton_utils.Config.from_args("download_wheels -D /tmp/wheels")
    assert config.action == "download_wheels"
    assert config._download_dir == "/tmp/wheels"
    assert config.wheel_set == []
    assert config.latest_wf_run is None
    assert config.gh_run_id is None
    assert config.download_for_all_pythons is False


def test_config_parse_alias():
    """Alias: triton-utils wheels -D /tmp/wheels"""
    config = triton_utils.Config.from_args("wheels -D /tmp/wheels")
    assert config.action == "download_wheels"


def test_config_parse_full():
    """Full invocation with all options."""
    config = triton_utils.Config.from_args("wheels -D /tmp/wheels --latest-wf-run benchmarks "
                                           "--wheel-set torch --wheel-set triton --python-version 3.12 "
                                           "-R my/repo -B dev")
    assert config.action == "download_wheels"
    assert config._download_dir == "/tmp/wheels"
    assert config.latest_wf_run == "benchmarks"
    assert config.wheel_set == ["torch", "triton"]
    assert config.python_version == "3.12"
    assert config.repo == "my/repo"
    assert config.branch == "dev"


def test_config_parse_run_id():
    """Specific run ID."""
    config = triton_utils.Config.from_args("wheels -D /tmp/wheels --run 12345")
    assert config.gh_run_id == "12345"


def test_config_parse_download_for_all_pythons():
    """Download for all pythons flag."""
    config = triton_utils.Config.from_args("wheels -D /tmp/wheels --download-for-all-pythons")
    assert config.download_for_all_pythons is True


def test_config_parse_invalid_wheel_set():
    """Invalid wheel set should fail."""
    with pytest.raises(SystemExit):
        triton_utils.Config.from_args("wheels -D /tmp/wheels --wheel-set invalid")


def test_wheel_sets_are_defined():
    """All expected wheel sets are defined."""
    assert "torch" in GHAWheelDownloader.WHEEL_SETS
    assert "triton" in GHAWheelDownloader.WHEEL_SETS
    assert "bench" in GHAWheelDownloader.WHEEL_SETS
    assert "pti" in GHAWheelDownloader.WHEEL_SETS


def test_workflow_presets_are_defined():
    """All expected workflow presets are defined."""
    assert "nightly" in GHAWheelDownloader.WORKFLOW_PRESETS
    assert "benchmarks" in GHAWheelDownloader.WORKFLOW_PRESETS


def test_get_wheel_patterns_empty():
    """No wheel set → no patterns."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"))
    assert d._get_wheel_patterns() == []


def test_get_wheel_patterns_single_set():
    """Single wheel set returns its patterns."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"), wheel_set=["torch"])
    patterns = d._get_wheel_patterns()
    assert "torch-*.whl" in patterns
    assert "torchvision-*.whl" in patterns


def test_get_wheel_patterns_multiple_sets():
    """Multiple wheel sets return union of patterns."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"), wheel_set=["torch", "triton"])
    patterns = d._get_wheel_patterns()
    assert "torch-*.whl" in patterns
    assert "triton-*.whl" in patterns


def test_get_wheel_patterns_with_artifact_pattern():
    """Artifact pattern is appended to wheel patterns."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"), wheel_set=["torch"], artifact_pattern="custom-*.whl")
    patterns = d._get_wheel_patterns()
    assert "custom-*.whl" in patterns
    assert "torch-*.whl" in patterns


def test_filter_artifacts_by_python():
    """Python version filters artifacts by name."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"), python_version="3.12")
    artifacts = [
        GHArtifact(name="wheels-pytorch-py3.12-20250101", size_in_bytes=100, expired=False, created_at="",
                   workflow_run_id="1", repo="test"),
        GHArtifact(name="wheels-pytorch-py3.10-20250101", size_in_bytes=100, expired=False, created_at="",
                   workflow_run_id="1", repo="test"),
    ]
    filtered = d._filter_artifacts_by_python(artifacts)
    assert len(filtered) == 1
    assert filtered[0].name == "wheels-pytorch-py3.12-20250101"


def test_filter_artifacts_by_python_none():
    """No python version → return all."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"), python_version=None)
    artifacts = [
        GHArtifact(name="wheels-py3.10", size_in_bytes=100, expired=False, created_at="", workflow_run_id="1",
                   repo="test"),
        GHArtifact(name="wheels-py3.12", size_in_bytes=100, expired=False, created_at="", workflow_run_id="1",
                   repo="test"),
    ]
    assert len(d._filter_artifacts_by_python(artifacts)) == 2


def test_filter_wheel_files(tmp_path):
    """Filter keeps only matching .whl files and removes others."""
    (tmp_path / "torch-2.7.whl").touch()
    (tmp_path / "triton-3.0.whl").touch()
    (tmp_path / "timm-0.9.whl").touch()

    d = GHAWheelDownloader(download_dir=tmp_path)
    kept = d._filter_wheel_files(tmp_path, ["torch-*.whl"])

    assert len(kept) == 1
    assert kept[0].name == "torch-2.7.whl"
    assert not (tmp_path / "triton-3.0.whl").exists()
    assert not (tmp_path / "timm-0.9.whl").exists()


def test_filter_wheel_files_no_patterns(tmp_path):
    """No patterns → keep all."""
    (tmp_path / "torch-2.7.whl").touch()
    (tmp_path / "triton-3.0.whl").touch()

    d = GHAWheelDownloader(download_dir=tmp_path)
    kept = d._filter_wheel_files(tmp_path, [])
    assert len(kept) == 2


def test_resolve_run_invalid_preset():
    """Invalid preset raises ValueError."""
    d = GHAWheelDownloader(download_dir=Path("/tmp"), latest_wf_run="nonexistent")
    with pytest.raises(ValueError, match="Unknown workflow preset"):
        d._resolve_run()
