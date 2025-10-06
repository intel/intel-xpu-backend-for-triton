import torch
import triton

import ctypes
import sys


def run_load_ir(temp_file, elem_size, *args):
    out_type = f"i{int(elem_size) * 4}"
    ir = f"""
    module attributes {{
        ttg.target = "xpu",
        "ttg.num-warps" = 32 : i32,
        "ttg.num-ctas" = 1 : i32,
        "ttg.threads-per-warp" = 16 : i32
        }} {{
        tt.func @dyn_block(
            %iptr : i64, %base_width : i32,
            %base_height : i32, %base_pitch : i32,
            %x : i32, %y : i32) {{
            %p0 = llvm.inttoptr %iptr : i64 to !llvm.ptr

            %v = triton_gen.2Dblockload %p0, %base_width, %base_height,
                %base_pitch, %x, %y
                {{ elem_size_in_bits = {elem_size}, tile_width = 8, tile_height = 8,
                v_blocks = 1, transpose = false,
                vnni_transform = false, cache_control = Default }}
                : (!llvm.ptr, i32, i32, i32, i32, i32)
                -> vector<1x{out_type}>

            // prevent GluonInline
            %v_cast = llvm.bitcast %v : vector<1x{out_type}> to {out_type}
            llvm.inline_asm has_side_effects asm_dialect = att
                "", "r" %v_cast : ({out_type}) -> ()

            tt.return
        }}
    }}
    """

    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(ir)

    kernel = triton.compile(temp_file)

    a = torch.zeros((256, 64), dtype=torch.float32, device="xpu")

    addr = ctypes.c_int64(a.data_ptr()).value

    kernel[(1, 1, 1)](addr, *map(int, args), 0)


if __name__ == "__main__":
    fn = globals()[sys.argv[1]]
    fn(*sys.argv[2:])
