import torch
import triton
import triton.language as tl
import itertools
import tempfile


def test_case():

    ir = """
#loc = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0)
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_tensor_pointer_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg1: !tt.ptr<f16> {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg2: !tt.ptr<f32> {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg3: i32 {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg4: i32 {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg5: i32 {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg6: i32 {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg7: i32 {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0), %arg8: i32 {tt.divisibility = 32 : i32} loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":242:0)) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32 loc(#loc1)
    %c255_i32 = arith.constant 255 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c4_i32 = arith.constant 4 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<32> : tensor<128x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc1)
    %cst_1 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc1)
    %cst_2 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc1)
    %cst_3 = arith.constant dense<0.0> : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc1)
    %cst_4 = arith.constant dense<0.0> : tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.addi %arg3, %c127_i32 : i32 loc(#loc60)
    %2 = arith.divsi %1, %c128_i32 : i32 loc(#loc61)
    %3 = arith.addi %arg4, %c255_i32 : i32 loc(#loc62)
    %4 = arith.divsi %3, %c256_i32 : i32 loc(#loc63)
    %5 = arith.muli %4, %c4_i32 : i32 loc(#loc7)
    %6 = arith.divsi %0, %5 : i32 loc(#loc8)
    %7 = arith.muli %6, %c4_i32 : i32 loc(#loc9)
    %8 = arith.subi %2, %7 : i32 loc(#loc10)
    %9 = arith.minsi %8, %c4_i32 : i32 loc(#loc11)
    %10 = arith.remsi %0, %9 : i32 loc(#loc12)
    %11 = arith.addi %7, %10 : i32 loc(#loc13)
    %12 = arith.remsi %0, %5 : i32 loc(#loc14)
    %13 = arith.divsi %12, %9 : i32 loc(#loc15)
    %14 = arith.muli %11, %c128_i32 : i32 loc(#loc16)
    %15 = arith.muli %13, %c256_i32 : i32 loc(#loc17)
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc18)
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc19)
    %20 = arith.addi %18, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc19)
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc20)
    %24 = tt.splat %15 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc21)
    %26 = arith.addi %24, %22 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc21)
    %28 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc22)
    %29 = arith.cmpi slt, %20, %28 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc22)
    %30 = arith.select %29, %20, %cst_2 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc23)
    %31 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc24)
    %32 = arith.cmpi slt, %26, %31 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc24)
    %33 = arith.select %32, %26, %cst_1 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc25)
    %34 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc26)
    %35 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc27)
    %36 = arith.muli %34, %35 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc27)
    %37 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> loc(#loc28)
    %38 = tt.expand_dims %37 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc28)
    %39 = tt.broadcast %36 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc29)
    %40 = tt.broadcast %38 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc29)
    %41 = arith.addi %39, %40 : tensor<128x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc29)
    %42 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc30)
    %43 = tt.addptr %42, %41 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc30)
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> loc(#loc31)
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> -> tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc31)
    %46 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc32)
    %47 = arith.muli %45, %46 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc32)
    %48 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>}>> -> tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc33)
    %49 = tt.broadcast %47 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc34)
    %50 = tt.broadcast %48 : tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc34)
    %51 = arith.addi %49, %50 : tensor<32x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc34)
    %52 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc35)
    %53 = tt.addptr %52, %51 : tensor<32x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<32x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc35)
    %54 = arith.addi %arg5, %c31_i32 : i32 loc(#loc64)
    %55 = arith.divsi %54, %c32_i32 : i32 loc(#loc65)
    %56 = arith.muli %arg7, %c32_i32 : i32 loc(#loc37)
    %57 = tt.splat %56 : i32 -> tensor<32x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc38)
    %58:2 = scf.for %arg9 = %c0_i32 to %55 step %c1_i32 iter_args(%arg11 = %43, %arg12 = %53) -> (tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<32x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>)  : i32 {
      %78 = arith.muli %arg9, %c32_i32 : i32 loc(#loc40)
      %79 = arith.subi %arg5, %78 : i32 loc(#loc41)
      %80 = tt.splat %79 : i32 -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc42)
      %81 = arith.cmpi slt, %38, %80 : tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc42)
      %82 = tt.broadcast %81 : tensor<1x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc43)
      tt.print " A ptr: " {hex = false, isSigned = array<i32: 0>} : %arg11 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      //tt.print " A mask: " {hex = false, isSigned = array<i32: 0>} : %82 : tensor<128x32xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %83 = tt.load %arg11, %82, %cst_3 {triton_intel_gpu.block_io = "row_major"} : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc43)
      tt.print " A tensor: " {hex = false, isSigned = array<i32: 0>} : %83 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %84 = tt.splat %79 : i32 -> tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc44)
      %85 = arith.cmpi slt, %45, %84 : tensor<32x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc44)
      %86 = tt.broadcast %85 : tensor<32x1xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x256xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc45)
      %87 = tt.load %arg12, %86, %cst_4 {triton_intel_gpu.block_io = "row_major"} : tensor<32x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc45)
      //tt.print "B tensor: " {hex = false, isSigned = array<i32: 0>} : %87 : tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %89 = tt.addptr %arg11, %cst_0 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc47)
      %90 = tt.addptr %arg12, %57 : tensor<32x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<32x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc38)
      scf.yield %89, %90 : tensor<128x32x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<32x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> loc(#loc48)
    } loc(#loc39)
    tt.return loc(#loc59)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":252:24)
#loc3 = loc("/home/jovyan/workspace/triton/python/triton/language/standard.py":40:22)
#loc4 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":253:27)
#loc5 = loc("/home/jovyan/workspace/triton/python/triton/language/standard.py":40:28)
#loc6 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":254:27)
#loc7 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":255:38)
#loc8 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":256:22)
#loc9 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":257:29)
#loc10 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":258:35)
#loc11 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":258:48)
#loc12 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":259:33)
#loc13 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":259:27)
#loc14 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":260:19)
#loc15 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":260:40)
#loc16 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":262:22)
#loc17 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":263:22)
#loc18 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":265:37)
#loc19 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":265:24)
#loc20 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":266:37)
#loc21 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":266:24)
#loc22 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":267:33)
#loc23 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":267:45)
#loc24 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":268:33)
#loc25 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":268:45)
#loc26 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":273:30)
#loc27 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":273:41)
#loc28 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":273:60)
#loc29 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":273:53)
#loc30 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":273:22)
#loc31 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":274:29)
#loc32 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":274:40)
#loc33 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":274:60)
#loc34 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":274:52)
#loc35 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":274:22)
#loc36 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":278:33)
#loc37 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":283:33)
#loc38 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":283:18)
#loc39 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":278:22)
#loc40 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":279:59)
#loc41 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":279:55)
#loc42 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":279:51)
#loc43 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":279:20)
#loc44 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":280:51)
#loc45 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":280:20)
#loc46 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":281:35)
#loc47 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":282:18)
#loc48 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":283:8)
#loc49 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":288:27)
#loc50 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":292:41)
#loc51 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":292:33)
#loc52 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":292:21)
#loc53 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":292:72)
#loc54 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":292:52)
#loc55 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":293:33)
#loc56 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":293:58)
#loc57 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":293:39)
#loc58 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":294:21)
#loc59 = loc("/home/jovyan/workspace/triton/python/../benchmarks/triton_kernels_benchmark/gemm_benchmark.py":294:4)
#loc60 = loc(callsite(#loc3 at #loc4))
#loc61 = loc(callsite(#loc5 at #loc4))
#loc62 = loc(callsite(#loc3 at #loc6))
#loc63 = loc(callsite(#loc5 at #loc6))
#loc64 = loc(callsite(#loc3 at #loc36))
#loc65 = loc(callsite(#loc5 at #loc36))

    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    M, N, K = 128, 256, 64

    a = torch.arange(M * K).to(torch.uint8).to(device='xpu')
    # a = torch.arange(M*K) * 0.0001
    # a = a.to(dtype=torch.float16).to(device='xpu')
    b = torch.arange(K * N).to(torch.uint8).to(device='xpu')
    # b = torch.arange(K*N) * 0.0001
    # b = b.to(dtype=torch.float16).to(device='xpu')
    c = torch.zeros(M * N, dtype=torch.float32, device='xpu')
    # torch.set_printoptions(profile="full")
    # print(b.cpu())

    # 64 byte aligned base.
    kernel[(1, 1, 1)](a, b, c, M, N, K, 64, 256, 256)
    # torch.set_printoptions(profile="full")
    # print(c.cpu())


#     launch num param:9
# gridx: 1
# gridY: 1
# gridZ: 1
# num_warps: 32
# threads_per_warp: 16
# global range:[x:512, y:1, z:1]
# local range:[x:512, y:1, z:1]
# shared_memory: 66560
# param 0:0xff00ffffffc00000
# param 1:0xff00ffffffc04000
# param 2:0xff00ffffffc0c000
# param 3:128
# param 4:256
# param 5:64
# param 6:64
# param 7:256
# param 8:256

if __name__ == "__main__":
    test_case()
