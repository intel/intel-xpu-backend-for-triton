try:
    # Optional helper for reasoning about linear layouts. In trimmed-down
    # builds where the linear_layout bindings are not registered, fall back to
    # a stub so that importing triton.tools does not fail.
    from triton._C.libtriton.linear_layout import LinearLayout  # type: ignore
except (ImportError, ModuleNotFoundError):
    class LinearLayout:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LinearLayout helper is not available in this Triton build."
            )
