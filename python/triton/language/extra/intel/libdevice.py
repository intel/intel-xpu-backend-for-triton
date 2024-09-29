from triton.language import core


@core.extern
def clz(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__imf_clz", core.dtype("int32")),
            (core.dtype("int64"), ): ("__imf_clzll", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def popc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__imf_popc", core.dtype("int32")),
            (core.dtype("int64"), ): ("__imf_popcll", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def byte_perm(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): ("__imf_byte_perm", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def mulhi(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__imf_mulhi", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__imf_umulhi", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("__imf_mul64hi", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__imf_umul64hi", core.dtype("uint64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul24(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__imf_mul24", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__imf_umul24", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def brev(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__imf_brev", core.dtype("int32")),
            (core.dtype("int64"), ): ("__imf_brevll", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sad(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): ("__imf_sad", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")): ("__imf_usad", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def abs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__imf_abs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__imf_llabs", core.dtype("int64")),
            (core.dtype("fp16"), ): ("__imf_fabsf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_fabsf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_fabs", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def floor(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_floorf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_floorf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_floor", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rsqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_rsqrtf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_rsqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_rsqrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_ceilf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_ceilf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_ceil", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def trunc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_truncf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_truncf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_trunc", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_exp2f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_exp2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_exp2", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def saturatef(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_saturatef", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_dividef(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fdiv_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_ddiv_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fdiv_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_ddiv_rz", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fdiv_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_ddiv_rd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fdiv_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_ddiv_ru", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rn(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_frcp_rn", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_drcp_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rz(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_frcp_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_drcp_rz", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_rd(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_frcp_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_drcp_rd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcp_ru(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_frcp_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_drcp_ru", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_sqrtf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_sqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_sqrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dadd_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fadd_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dadd_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fadd_rz", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dadd_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fadd_rd", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def add_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dadd_ru", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fadd_ru", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dmul_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fmul_rn", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dmul_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fmul_rz", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dmul_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fmul_rd", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def mul_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dmul_ru", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fmul_ru", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2int_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2int_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2int_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2int_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2uint_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2uint_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2uint_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2uint_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2uint_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__imf_int2double_rn", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__imf_uint2double_rn", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2int_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2int_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2int_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2int_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2int_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2uint_rn", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2uint_rz", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2uint_rd", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2uint_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2uint_ru", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__imf_int2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__imf_int2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__imf_int2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__imf_int2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__imf_uint2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__imf_uint2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__imf_uint2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__imf_uint2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def hiloint2double(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("int32")): ("__imf_hiloint2double", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2loint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2loint", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2hiint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2hiint", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ll_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ll_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ll_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ll_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ull_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ull_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ull_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float2ull_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ll_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ll_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ll_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ll_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ll_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ull_rn", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ull_rz", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ull_rd", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double2ull_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double2ull_ru", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2float_rn", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2float_rz", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2float_rd", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2float_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2float_ru", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2double_rn", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2double_rz", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2double_rd", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ll2double_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_ll2double_ru", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rn(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2double_rn", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rz(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2double_rz", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_rd(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2double_rd", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def ull2double_ru(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__imf_ull2double_ru", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__imf_int_as_float", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_int(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float_as_int", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def uint_as_float(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__imf_uint_as_float", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def float_as_uint(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_float_as_uint", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def longlong_as_double(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__imf_longlong_as_double", core.dtype("fp64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def double_as_longlong(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__imf_double_as_longlong", core.dtype("int64")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_log2f(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_fast_log2f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_logf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_fast_logf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_expf(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_exp10f(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_fast_exp10f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_log10f(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_fast_log10f", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def fast_powf(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fast_powf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def hadd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__imf_hadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__imf_uhadd", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rhadd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__imf_rhadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__imf_urhadd", core.dtype("uint32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fsub_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dsub_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fsub_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dsub_rz", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fsub_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dsub_rd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sub_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fsub_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_dsub_ru", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ffs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__imf_ffs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__imf_ffsll", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__imf_rintf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_rintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_rint", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llrint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__imf_llrintf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__imf_llrint", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nearbyint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp16"), ): ("__imf_nearbyintf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_nearbyintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_nearbyint", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__imf_isnanf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__imf_isnan", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def signbit(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__imf_signbitf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__imf_signbitd", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def copysign(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp16"), core.dtype("fp16")): ("__imf_copysignf16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_copysignf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_copysign", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def finitef(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__imf_finitef", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_isinff", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__imf_isinf", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def nextafter(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_nextafterf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_nextafter", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_sinf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_sinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_sin", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_cosf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_cosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_cos", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinpi(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_sinpif", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_sinpi", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cospi(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_cospif", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_cospi", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_tanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_tan", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log2(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_log2f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_log2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_log2", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_expf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_expf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_exp", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def exp10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_exp10f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_exp10f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_exp10", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_coshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_cosh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def sinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_sinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_sinh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_tanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_tanh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan2(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_atan2f", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_atan2", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_atanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_atan", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asin(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_asinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_asin", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acos(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_acosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_acos", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_logf16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_logf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_log", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log10(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp16"), ): ("__imf_log10f16", core.dtype("fp16")),
            (core.dtype("fp32"), ): ("__imf_log10f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_log10", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def log1p(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_log1pf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_log1p", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def acosh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_acoshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_acosh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def asinh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_asinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_asinh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def atanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_atanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_atanh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def expm1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_expm1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_expm1", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def hypot(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_hypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_hypot", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rhypot(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_rhypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_rhypot", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def norm3d(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__imf_norm3df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__imf_norm3d", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rnorm3d(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__imf_rnorm3df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__imf_rnorm3d", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__imf_norm4df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("__imf_norm4d", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__imf_rnorm4df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("__imf_rnorm4d", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cbrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_cbrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_cbrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def rcbrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_rcbrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_rcbrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def j0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_j0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_j0", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def j1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_j1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_j1", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def y0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_y0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_y0", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def y1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_y1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_y1", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def yn(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("fp32")): ("__imf_ynf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def jn(arg0, arg1, _builder=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("fp32")): ("__imf_jnf", core.dtype("fp32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def cyl_bessel_i0(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_i0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_i0", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def cyl_bessel_i1(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_i1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_i1", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_erff", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_erf", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_erfinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_erfinv", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfc(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_erfcf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_erfc", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfcx(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_erfcxf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_erfcx", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def erfcinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_erfcinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_erfcinv", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def normcdfinv(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cdnorminvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cdnorminv", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def normcdf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cdnormf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cdnorm", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def lgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_lgammaf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_lgamma", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ldexp(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__imf_ldexpf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__imf_ldexp", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def scalbn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__imf_scalbnf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__imf_scalbn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fmodf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_fmod", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def remainder(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_remainderf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_remainder", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fma(arg0, arg1, arg2, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp16"), core.dtype("fp16"), core.dtype("fp16")): ("__imf_fmaf16", core.dtype("fp16")),
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__imf_fmaf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__imf_fma", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__imf_powif", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__imf_powi", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_powf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_pow", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def tgamma(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_tgammaf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_tgamma", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_roundf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_round", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def llround(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_llroundf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__imf_llround", core.dtype("int64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def fdim(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__imf_fdimf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__imf_fdim", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def ilogb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_ilogbf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__imf_ilogb", core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.extern
def logb(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_logbf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_logb", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def isfinited(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__imf_isfinitef", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__imf_isfinite", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)
