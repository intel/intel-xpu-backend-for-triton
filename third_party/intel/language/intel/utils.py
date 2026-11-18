from math import gcd

from triton.language import core

# Standard SPV_KHR_shader_clock builtin, Itanium-mangled with its scope argument.
_READ_CLOCK = "_Z27__spirv_ReadClockKHR_Rulongi"
_SCOPE_DEVICE = 1
# The LTS driver does not implement the standard builtin.
_READ_CYCLE_COUNTER = "__builtin_IB_read_cycle_counter"
_SLICE_ID = "__builtin_IB_slice_id"
_DUAL_SUBSLICE_ID = "__builtin_IB_dual_subslice_id"

# Widest sub-slice field across supported architectures is 5 bits (Xe2).
_SUBSLICE_ID_BITS = 8


def _builtin_call(name, dtype, is_pure, _semantic, args=()):
    arg_types = tuple(arg.dtype for arg in args)
    return core.extern_elementwise("", "", list(args), {arg_types: (name, dtype)}, is_pure=is_pure, _semantic=_semantic)


@core.extern
def globaltimer(_semantic=None):
    """Timestamp in nanoseconds."""
    # kHz is cycles per millisecond.
    clock_rate_khz = _semantic.builder.options.core_clock_rate
    if clock_rate_khz <= 0:
        raise RuntimeError("globaltimer needs a positive core clock rate to report nanoseconds")

    if _semantic.builder.options.is_lts:
        ticks = _builtin_call(_READ_CYCLE_COUNTER, core.dtype("int64"), False, _semantic)
    else:
        scope = core.full((), _SCOPE_DEVICE, core.int32, _semantic=_semantic)
        ticks = _builtin_call(_READ_CLOCK, core.dtype("int64"), False, _semantic, args=(scope, ))
    # The counter is unsigned, so divide it as such.
    ticks = _semantic.cast(ticks, core.uint64)
    # Reduce by the GCD so the multiplication does not overflow.
    ns_per_ms = 1000000
    divisor = gcd(ns_per_ms, clock_rate_khz)
    scaled = _semantic.mul(ticks, ns_per_ms // divisor, sanitize_overflow=False)
    return _semantic.cast(_semantic.floordiv(scaled, clock_rate_khz // divisor), core.int64)


@core.extern
def smid(_semantic=None):
    """Opaque id of the XeCore the current thread runs on."""
    slice_id = _builtin_call(_SLICE_ID, core.dtype("int32"), True, _semantic)
    subslice_id = _builtin_call(_DUAL_SUBSLICE_ID, core.dtype("int32"), True, _semantic)
    shifted = _semantic.shl(slice_id, core.full((), _SUBSLICE_ID_BITS, core.int32, _semantic=_semantic))
    return _semantic.or_(shifted, subslice_id)


@core.builtin
def num_threads(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps * 32)


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)


def convert_fp8e4b15_to_float16(arg, _semantic):
    # Need to bitcast the source first because it's represented as tensor of i8 in MLIR.
    tmp_ty = _semantic.builder.get_block_ty(_semantic.builder.get_fp8e4b8_ty(), arg.type.shape)
    tmp = _semantic.builder.create_bitcast(arg.handle, tmp_ty)
    # Now generate FpToFp op for upcast.
    dst_ty = core.block_type(core.float16, arg.type.get_block_shapes())
    upcast = _semantic.builder.create_fp_to_fp(tmp, dst_ty.to_ir(_semantic.builder), None)
    return core.tensor(upcast, dst_ty)


def convert_float_to_fp8e4b15(arg, fp_downcast_rounding, _semantic):
    tmp_ty = _semantic.builder.get_fp8e4b8_ty()
    if arg.type.is_block():
        tmp_ty = _semantic.builder.get_block_ty(tmp_ty, arg.type.shape)
    tmp = _semantic.builder.create_fp_to_fp(arg.handle, tmp_ty, fp_downcast_rounding)
    dst_ty = arg.type.with_element_ty(core.float8e4b15) if arg.type.is_block() else core.float8e4b15
    return core.tensor(_semantic.builder.create_bitcast(tmp, dst_ty.to_ir(_semantic.builder)), dst_ty)


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    if arg.type.scalar.is_fp8e4b15():
        if not (dst_ty.scalar.is_fp16() or dst_ty.scalar.is_fp32()):
            raise AssertionError
        upcast_val = convert_fp8e4b15_to_float16(arg, _semantic=_semantic)
        if dst_ty.scalar.is_fp32():
            upcast_val = upcast_val.to(core.float32, _semantic=_semantic)
        return upcast_val

    if dst_ty.scalar.is_fp8e4b15():
        if not (arg.type.scalar.is_fp16() or arg.type.scalar.is_fp32()):
            raise AssertionError
        return convert_float_to_fp8e4b15(arg, fp_downcast_rounding, _semantic=_semantic)

    raise AssertionError(f"Intel target doesn't provide conversion for {arg.type} to {dst_ty}")
