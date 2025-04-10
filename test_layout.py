import torch
import triton
import tempfile


def test_2d_block_io():

    ir = """
    #blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#loc = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1519:0)
#loc5 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1481:0)
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 8], B = [8, 16], C = [32, 16]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1519:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1519:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1519:0)) attributes {noinline = false} {
    %0 = tt.load %arg0 : !tt.ptr<f32> loc(#loc1)
    %1 = tt.load %arg1 : !tt.ptr<f32> loc(#loc2)
    tt.call @noinline_shared_fn__fp32_fp32_Pfp32__(%0, %1, %arg2) : (f32, f32, !tt.ptr<f32>) -> () loc(#loc3)
    tt.return loc(#loc4)
  } loc(#loc)
  tt.func private @noinline_shared_fn__fp32_fp32_Pfp32__(%arg0: f32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64} loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1481:0), %arg1: f32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64} loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1481:0), %arg2: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64} loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1481:0)) attributes {noinline = true} {
    %cst = arith.constant dense<32> : tensor<32x1xi32, #blocked> loc(#loc6)
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc6)
    %cst_1 = arith.constant dense<32> : tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc6)
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> loc(#loc7)
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>}>> loc(#loc7)
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc7)
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> -> tensor<32x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc7)
    %4 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>}>> -> tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc7)
    %5 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked> loc(#loc7)
    %6 = arith.muli %3, %cst_0 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc8)
    %7 = arith.muli %4, %cst_1 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc8)
    %8 = arith.muli %5, %cst : tensor<32x1xi32, #blocked> loc(#loc8)
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> loc(#loc9)
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>}>> loc(#loc9)
    %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc9)
    %12 = tt.expand_dims %9 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc9)
    %13 = tt.expand_dims %10 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc9)
    %14 = tt.expand_dims %11 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc9)
    %15 = tt.broadcast %6 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<32x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc10)
    %16 = tt.broadcast %7 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc10)
    %17 = tt.broadcast %8 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked> loc(#loc10)
    %18 = tt.broadcast %12 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<32x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc10)
    %19 = tt.broadcast %13 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc10)
    %20 = tt.broadcast %14 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked> loc(#loc10)
    %21 = arith.addi %15, %18 : tensor<32x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc10)
    %22 = arith.addi %16, %19 : tensor<32x32xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc10)
    %23 = arith.addi %17, %20 : tensor<32x32xi32, #blocked> loc(#loc10)
    %24 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc11)
    %25 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc11)
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked> loc(#loc11)
    %27 = tt.addptr %24, %21 : tensor<32x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<32x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc11)
    %28 = tt.addptr %25, %22 : tensor<32x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>, tensor<32x32xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc11)
    %29 = tt.addptr %26, %23 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked> loc(#loc11)
    # %30 = tt.load %27 {triton_intel_gpu.block_io = "row_major"} : tensor<32x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> loc(#loc12)
    %31 = tt.load %28 {triton_intel_gpu.block_io = "row_major"} : tensor<32x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> loc(#loc12)
    tt.print " Z a: " {hex = false, isSigned = array<i32: 1>} : %30 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %32 = tt.splat %arg0 : f32 -> tensor<32x32xf32, #mma> loc(#loc13)
    %33 = tt.dot %30, %31, %32, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma> loc(#loc14)
    %34 = tt.splat %arg1 : f32 -> tensor<32x32xf32, #mma> loc(#loc15)
    %35 = arith.addf %33, %34 : tensor<32x32xf32, #mma> loc(#loc15)
    %36 = ttg.convert_layout %35 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked> loc(#loc16)
    tt.store %29, %36 : tensor<32x32x!tt.ptr<f32>, #blocked> loc(#loc16)
    tt.return loc(#loc17)
  } loc(#loc5)
} loc(#loc)
#loc1 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1520:16)
#loc2 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1521:16)
#loc3 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1522:29)
#loc4 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1522:4)
#loc6 = loc(unknown)
#loc7 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1482:28)
#loc8 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1482:39)
#loc9 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1482:61)
#loc10 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1482:44)
#loc11 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1483:20)
#loc12 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1483:16)
#loc13 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1484:23)
#loc14 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1484:18)
#loc15 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1484:27)
#loc16 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1485:23)
#loc17 = loc("/home/jovyan/workspace/triton/python/test/unit/language/test_core.py":1485:4)
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    M, N = 32, 32
    x = torch.tensor([1.0], device="xpu", dtype=torch.float32)
    y = torch.tensor([2.0], device="xpu", dtype=torch.float32)
    z = torch.ones((M, N), device="xpu", dtype=torch.float32)
    kernel[(1, 1, 1)](x, y, z)
    ref = torch.full((M, N), 16, device="xpu", dtype=torch.float32)
    assert torch.equal(z, ref + x + y)


if __name__ == "__main__":
    test_2d_block_io()
