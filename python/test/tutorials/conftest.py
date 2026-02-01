"""Pytest configuration for tutorial tests."""


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
