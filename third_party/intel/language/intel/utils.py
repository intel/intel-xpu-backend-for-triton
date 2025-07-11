from triton.language import core


@core.extern
def globaltimer(_semantic=None):
    return core.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=core.int64, is_pure=False, pack=1,
                                       _semantic=_semantic)


@core.extern
def smid(_semantic=None):
    return core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                                       _semantic=_semantic)


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


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    if arg.type.scalar.is_fp8e4b15():
        if not (dst_ty.scalar.is_fp16() or dst_ty.scalar.is_fp32()):
            raise AssertionError
        upcast_val = convert_fp8e4b15_to_float16(arg, _semantic=_semantic)
        if dst_ty.scalar.is_fp32():
            upcast_val = upcast_val.to(core.float32, _semantic=_semantic)
        return upcast_val

    raise AssertionError(f"Intel target doesn't provide conversion for {arg.type} to {dst_ty}")
