"""The Intel backend's errors."""
from triton.errors import TritonError
from triton.runtime.errors import OutOfResources


class IntelGPUError(TritonError):
    """
    Exception raised for Intel GPU/Level Zero errors.

    This exception is used to indicate errors specific to Intel GPU backends,
    particularly those related to Level Zero.
    """
    pass


class WorkGroupSizeOutOfResources(OutOfResources):
    """`num_warps * warp_size` work-items do not fit in a work-group.

    An `OutOfResources`, so the autotuner skips the config, with a message that
    names the fix: the generic one suggests block sizes and `num_stages`, which do
    not change the work-group size.
    """

    def __init__(self, num_warps, warp_size, limit, smaller_warp_size=None, sub_32_dpas=False):
        super().__init__(num_warps * warp_size, limit, "work-group size")
        self.num_warps = num_warps
        self.warp_size = warp_size
        self.smaller_warp_size = smaller_warp_size
        self.sub_32_dpas = sub_32_dpas

    def __str__(self) -> str:
        fixes = f"use num_warps <= {self.limit // self.warp_size}"
        if self.smaller_warp_size is not None:
            fixes += f" or warp_size={self.smaller_warp_size}"
        msg = (f"out of resource: work-group size, Required: {self.required} work-items "
               f"(num_warps={self.num_warps} * warp_size={self.warp_size}), Hardware limit: {self.limit}. "
               f"To fit, {fixes}.")
        if self.sub_32_dpas:
            msg += (" sub_32_dpas (TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32) is enabled, so DPAS kernels use "
                    "warp_size=32 by default.")
        return msg

    def __reduce__(self):
        return (type(self), (self.num_warps, self.warp_size, self.limit, self.smaller_warp_size, self.sub_32_dpas))
