# The backend's names in triton's own namespaces: triton.runtime.errors.IntelGPUError.
from . import errors as _errors

_errors.install()
