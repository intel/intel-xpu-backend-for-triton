// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// CHECK:   llvm.func spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 33280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_tensor_pointer_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !llvm.ptr<3>) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant dense<64> : tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %cst_2 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %20 = arith.addi %18, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %28 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %29 = arith.cmpi slt, %20, %28 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %30 = arith.select %29, %20, %cst_2 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %34 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %35 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %36 = arith.muli %34, %35 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %37 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %38 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> -> tensor<1x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %39 = tt.broadcast %36 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %40 = tt.broadcast %38 : tensor<1x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %41 = arith.addi %39, %40 : tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %42 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %43 = tt.addptr %42, %41 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %54 = arith.addi %arg5, %c63_i32 : i32
    %55 = arith.divsi %54, %c64_i32 : i32
    cf.br ^bb1(%c0_i32, %43 : i32, tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>)
  ^bb1(%58: i32, %60: tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>):  // 2 preds: ^bb0, ^bb2
    %62 = arith.cmpi slt, %58, %55 : i32
    cf.cond_br %62, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %63 = arith.muli %58, %c64_i32 : i32
    %64 = arith.subi %arg5, %63 : i32
    %65 = tt.splat %64 : i32 -> tensor<1x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %66 = arith.cmpi slt, %38, %65 : tensor<1x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %67 = tt.broadcast %66 : tensor<1x64xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<128x64xi1, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK-32: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt
    %68 = tt.load %60, %67, %cst_3 {triton_intel_gpu.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %74 = tt.addptr %60, %cst_0 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %76 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%76, %74 : i32, tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}

// -----

// CHECK:   llvm.func spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x1cPU3AS1viiiDv2_iPj
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 33280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_tensor_pointer_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !llvm.ptr<3>) attributes {noinline = false} {
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %15 = arith.muli %13, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %24 = tt.splat %15 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %26 = arith.addi %24, %22 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>%31 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %32 = arith.cmpi slt, %26, %31 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %33 = arith.select %32, %26, %cst_1 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %46 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %47 = arith.muli %45, %46 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %48 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %49 = tt.broadcast %47 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %50 = tt.broadcast %48 : tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %51 = arith.addi %49, %50 : tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %52 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %53 = tt.addptr %52, %51 : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %54 = arith.addi %arg5, %c63_i32 : i32
    %55 = arith.divsi %54, %c64_i32 : i32
    %56 = arith.muli %arg7, %c64_i32 : i32
    %57 = tt.splat %56 : i32 -> tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    cf.br ^bb1(%c0_i32, %53 : i32, tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>)
  ^bb1(%58: i32, %61: tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>):  // 2 preds: ^bb0, ^bb2
    %62 = arith.cmpi slt, %58, %55 : i32
    cf.cond_br %62, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %63 = arith.muli %58, %c64_i32 : i32
    %64 = arith.subi %arg5, %63 : i32
    %69 = tt.splat %64 : i32 -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %70 = arith.cmpi slt, %45, %69 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %71 = tt.broadcast %70 : tensor<64x1xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x256xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    // CHECK-16:   llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x1cPU3AS1viiiDv2_iPj
    %72 = tt.load %61, %71, %cst_4 {triton_intel_gpu.block_io = "row_major"} : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %75 = tt.addptr %61, %57 : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %76 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%76, %75 : i32, tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}
