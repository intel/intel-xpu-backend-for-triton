import torch
import triton
import triton.language as tl

cached_capabilities = {}


@triton.constexpr_function
def is_cuda():
    return tl.target_info.current_target().backend == "cuda"


@triton.constexpr_function
def is_hip():
    return tl.target_info.current_target().backend == "hip"


@triton.constexpr_function
def is_hip_cdna3():
    return tl.target_info.current_target().arch == "gfx942"


@triton.constexpr_function
def is_hip_cdna4():
    return tl.target_info.current_target().arch == "gfx950"


def is_xpu():
    if "is_xpu" not in cached_capabilities:
        target = triton.runtime.driver.active.get_current_target()
        cached_capabilities["is_xpu"] = False if target is None else target.backend == "xpu"
    return cached_capabilities["is_xpu"]
 

@triton.constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = tl.target_info.current_target()
    if target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@triton.constexpr_function
def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = tl.target_info.current_target()
    if target.backend != 'hip':
        return -1
    if target.arch == 'gfx942':
        return 3
    if target.arch == 'gfx950':
        return 4
    return -1


@triton.constexpr_function
def has_tma_gather():
    return cuda_capability_geq(10, 0)


@triton.constexpr_function
def has_native_mxfp():
    return cuda_capability_geq(10, 0)


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties(0).multi_processor_count
    if is_xpu():
        return torch.xpu.get_device_properties(0).max_compute_units
