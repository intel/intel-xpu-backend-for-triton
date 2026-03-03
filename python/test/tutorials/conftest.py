"""Pytest configuration for tutorial tests."""

import pytest

_ALL_FA_CONFIGS = ["default", "fp8_only", "skip_fp8"]
_FA_MODES = ("fa_only", "fp8_only", "skip_fp8")


def pytest_addoption(parser):
    parser.addoption(
        "--tutorial06-mode",
        default="all",
        choices=["all", "skip", "fa_only", "fp8_only", "skip_fp8"],
        help=("Controls how 06-fused-attention is handled: "
              "'all' = run all tutorials including 06 (default config only), "
              "'skip' = skip 06 entirely, "
              "'fa_only' = run only 06 with default config, "
              "'fp8_only' = run only 06 with HEAD_DIM=128 FWD_FP8_ONLY=1, "
              "'skip_fp8' = run only 06 with HEAD_DIM=128 FWD_FP8_SKIP=1."),
    )


def pytest_generate_tests(metafunc):
    """Always parametrize test_06_fused_attention with all three configs.

    This ensures all variants are always collected and visible to the
    skiplist plugin (pytest-skip with --select-fail-on-missing), regardless
    of which --tutorial06-mode is active.
    """
    if "fa_config" not in metafunc.fixturenames:
        return
    metafunc.parametrize("fa_config", _ALL_FA_CONFIGS)


def _allowed_fa_configs(mode):
    """Return the set of fa_config values that should actually run."""
    if mode in ("all", "fa_only"):
        return {"default"}
    elif mode == "fp8_only":
        return {"fp8_only"}
    elif mode == "skip_fp8":
        return {"skip_fp8"}
    else:  # "skip"
        return set()


def pytest_collection_modifyitems(config, items):
    """Skip or deselect tests based on --tutorial06-mode.

    All three FA variants are always collected so that skiplist entries
    remain valid. Variants that should not run are marked as skipped
    (not deselected) so they appear in reports.
    """
    mode = config.getoption("--tutorial06-mode", default="all")
    allowed = _allowed_fa_configs(mode)

    if mode in _FA_MODES:
        # In dedicated FA modes, only keep tutorial 06 tests.
        items[:] = [i for i in items if "test_06_fused_attention" in i.nodeid]

    for item in items:
        if "test_06_fused_attention" not in item.nodeid:
            continue

        fa_config = item.callspec.params.get("fa_config") if hasattr(item, "callspec") else None

        if fa_config not in allowed:
            item.add_marker(pytest.mark.skip(
                reason=f"fa_config={fa_config} not run in --tutorial06-mode={mode}"))
