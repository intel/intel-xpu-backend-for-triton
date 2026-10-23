"""The Intel backend's errors."""
from triton.errors import TritonError


class IntelGPUError(TritonError):
    """
    Exception raised for Intel GPU/Level Zero errors.

    This exception is used to indicate errors specific to Intel GPU backends,
    particularly those related to Level Zero.
    """
    pass
