import torch
import pytest
import triton
import triton.language as tl

NAME_TO_DTYPE = {
    "float8e4nv": torch.float8_e4m3fn, "float8e5": torch.float8_e5m2, "float16": torch.float16, "bfloat16":
    torch.bfloat16, "float32": torch.float32
}


@triton.jit
def type_convert(src, dst, rounding: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + idxs)
    y = x.to(dst.dtype.element_ty, fp_downcast_rounding=rounding)
    tl.store(dst + idxs, y)


@pytest.mark.parametrize("dst_dtype", ["float8e4nv", "float8e5"])
@pytest.mark.parametrize("src_dtype", ["float16", "bfloat16", "float32"])
def test_convert_to_fp8(src_dtype, dst_dtype, device):
    src_dtype = NAME_TO_DTYPE[src_dtype]
    dst_dtype = NAME_TO_DTYPE[dst_dtype]
    src_idtype = torch.int32 if src_dtype == torch.float32 else torch.int16
    max_val = torch.finfo(dst_dtype).max
    min_val = torch.tensor(-1.0 * max_val, dtype=dst_dtype).view(torch.uint8).item()
    max_val = torch.tensor(max_val, dtype=dst_dtype).view(torch.uint8).item()
    SIZE = 2**16
    BLOCK_SIZE = SIZE // 32
    src = torch.arange(0, SIZE, dtype=src_idtype, device=device)
    if src_dtype == torch.float32:
        src = src << 16 | src
    src = src.view(src_dtype)
    dst = torch.empty_like(src, dtype=dst_dtype, device=device)

    def run_kernel():
        type_convert[(SIZE // BLOCK_SIZE, )](triton.reinterpret(src, src_dtype), triton.reinterpret(dst, dst_dtype),
                                             'rtne', BLOCK_SIZE)
        torch.xpu.synchronize(device)

    run_kernel()

    # import timeit
    # time = timeit.timeit(run_kernel, number=100000)
    # print(f"{src_dtype} -> {dst_dtype} time: {time}.")

    dst = dst.view(torch.uint8)
    expect = src.to(dtype=dst_dtype).view(torch.uint8)
    diff_mask = dst != expect
    src = src[diff_mask]
    dst = dst[diff_mask]
    expect = expect[diff_mask]

    for s, si, e, d in zip(src, src.view(src_idtype), expect.view(torch.uint8), dst.view(torch.uint8)):
        if torch.isnan(s):
            e = 0b01111111
        elif torch.isposinf(s) or (s >= 57344.) or (s >= 464. and dst_dtype == torch.float8_e4m3fn):
            e = max_val
        elif torch.isneginf(s) or (s <= -57344.) or (s <= -464. and dst_dtype == torch.float8_e4m3fn):
            e = min_val
        elif si == 0b1000000000000000:  # -0.0
            e = 0b10000000

        if d != e:
            sfmt = "032b" if src_dtype == torch.float32 else "016b"
            dfmt = "08b"
            msg = f"Src={s}({format(si, sfmt)}). Expected={format(e, dfmt)}. Actual={format(d, dfmt)}."
            pytest.fail(msg)
