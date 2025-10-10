import torch
import triton

import ctypes
import sys


def run_ir(device, temp_file):
    ir = r"""
    module attributes {
        ttg.target = "xpu",
        "ttg.num-warps" = 32 : i32,
        "ttg.num-ctas" = 1 : i32,
        "ttg.threads-per-warp" = 16 : i32
        } {
        tt.func @dyn_block(
            %iptr : i64, %base_width : i32,
            %base_height : i32, %base_pitch : i32,
            %x : i32, %y : i32) {
            %p0 = llvm.inttoptr %iptr : i64 to !llvm.ptr

            %0 = triton_gen.2Dblockload %p0, %base_width, %base_height,
                %base_pitch, %x, %y
                { elem_size_in_bits = 8, tile_width = 8, tile_height = 8,
                v_blocks = 1, transpose = false,
                vnni_transform = false, cache_control = Default }
                : (!llvm.ptr, i32, i32, i32, i32, i32)
                -> vector<2xi16>
            tt.return
        }
    }
    """

    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(ir)

    kernel = triton.compile(temp_file)

    a = torch.randn((256, 64), dtype=torch.float32, device=device)

    addr = ctypes.c_int64(a.data_ptr()).value

    kernel[(1, 1, 1)](addr, 64, 64, 1, 0, 0)


if __name__ == "__main__":
    fn = globals()[sys.argv[1]]
    fn(*sys.argv[2:])
