"""The Intel backend's errors.

`install()` also exposes `IntelGPUError` as `triton.runtime.errors.IntelGPUError`, where PyTorch and the backend's
launcher (driver.c) look it up.
"""
from triton.errors import TritonError


class IntelGPUError(TritonError):
    """
    Exception raised for Intel GPU/Level Zero errors.

    This exception is used to indicate errors specific to Intel GPU backends,
    particularly those related to Level Zero.
    """
    pass


def install() -> None:
    import triton.runtime.errors as core_errors  # not triton.runtime.errors.X: `triton` may still be importing
    core_errors.IntelGPUError = IntelGPUError
