import torch
import triton
import triton.language as tl
import itertools
import tempfile


def test_scan_layouts(M, N):

    ir = f"""
    #mma = #triton_intel_gpu.dpas<{{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}}>
    #dot_b = #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>
    module attributes {{"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block}} {{
    tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %col_stride: i32 {{tt.divisibility = 16 : i32}}) {{
        %c64_i32 = arith.constant 64 : i32
        %c64_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32
        %c32_i64 = arith.constant 32 : i64
        %20 = arith.extsi %col_stride : i32 to i64
        %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %20], [%c0_i32, %c0_i32] {{order = array<i32: 0, 1>}} : <tensor<64x32xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>>
        %45 = tt.load %21 {{triton_intel_gpu.block_io = "column_major"}} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>>
        tt.print "fp16 tensor: " {{hex = false, isSigned = array<i32: 0>}} : %45 : tensor<64x32xf16, #triton_gpu.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>
        tt.return
    }}
    }}
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    k = torch.arange(M * N).view(M, N).transpose(0, 1).contiguous() * 0.01
    k = k.to(torch.float16).to("xpu")
    print(k.cpu())

    kernel[(1, 1, 1)](k, k.stride()[0])


if __name__ == "__main__":
    test_scan_layouts(64, 64)
