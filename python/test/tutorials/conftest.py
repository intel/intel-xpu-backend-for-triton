"""Pytest configuration for tutorial tests."""

_FA_MODES = ("fa_only", "fp8_only", "skip_fp8")


def pytest_addoption(parser):
    parser.addoption(
        "--tutorial06-mode",
        default="all",
        choices=["all", "skip", "fa_only", "fp8_only", "skip_fp8"],
        help=("Controls how 06-fused-attention is handled: "
              "'all' = run all tutorials including 06 (default), "
              "'skip' = skip 06 (it runs in dedicated FA jobs), "
              "'fa_only' = run only 06 with default config, "
              "'fp8_only' = run only 06 with HEAD_DIM=128 FWD_FP8_ONLY=1, "
              "'skip_fp8' = run only 06 with HEAD_DIM=128 FWD_FP8_SKIP=1."),
    )


def pytest_generate_tests(metafunc):
    """Parametrize test_06_fused_attention with the appropriate FA config."""
    if "fa_config" not in metafunc.fixturenames:
        return

    mode = metafunc.config.getoption("--tutorial06-mode", default="all")
    if mode in ("all", "fa_only"):
        metafunc.parametrize("fa_config", ["default"])
    elif mode == "fp8_only":
        metafunc.parametrize("fa_config", ["fp8_only"])
    elif mode == "skip_fp8":
        metafunc.parametrize("fa_config", ["skip_fp8"])
    else:
        metafunc.parametrize("fa_config", [])


def pytest_collection_modifyitems(config, items):
    """Deselect tests based on --tutorial06-mode."""
    mode = config.getoption("--tutorial06-mode", default="all")
    # "skip" is already handled by parametrize(fa_config, []) yielding no items.
    if mode in _FA_MODES:
        items[:] = [i for i in items if "test_06_fused_attention" in i.nodeid]
