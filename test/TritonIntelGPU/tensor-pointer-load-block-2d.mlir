// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 33280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
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
    %34 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %35 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %cst_1 = arith.constant dense<512> : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %36 = arith.muli %34, %cst_1 : tensor<128x1xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
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
    // CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1
    %68 = tt.load %60, %67, %cst_3 {ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %74 = tt.addptr %60, %cst_0 : tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %76 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%76, %74 : i32, tensor<128x64x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 33280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_tensor_pointer_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 4 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !llvm.ptr<3>) attributes {noinline = false} {
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
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_2 = arith.constant dense<512> : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %47 = arith.muli %45, %cst_2 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %48 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
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
    // CHECK-COUNT-32: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 4, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    %72 = tt.load %61, %71, %cst_4 {ttig.block_io = "row_major"} : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %75 = tt.addptr %61, %57 : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %76 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%76, %75 : i32, tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 33280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
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
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_2 = arith.constant dense<512> : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %47 = arith.muli %45, %cst_2 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %48 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
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
    // CHECK-COUNT-16: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = true, cache_control = Default}
    %72 = tt.load %61, %71, %cst_4 {ttig.block_io = "row_major"} : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %75 = tt.addptr %61, %57 : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %76 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%76, %75 : i32, tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 33280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
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
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_2 = arith.constant dense<512> : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %47 = arith.muli %45, %cst_2 : tensor<64x1xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %48 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>> -> tensor<1x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
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
    // CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
    %72 = tt.load %61, %71, %cst_4 {ttig.block_io = "row_major"} : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %75 = tt.addptr %61, %57 : tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, tensor<64x256xi32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %76 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%76, %75 : i32, tensor<64x256x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
#mma_1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1]}>
#mma_2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [4, 2]}>
module attributes {ttig.support_sg_2d_block, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<f16>,
                                           %arg1: !tt.ptr<f16>,
                                           %arg2: !tt.ptr<f16>,
                                           %arg3: !tt.ptr<f16>) {
    // CHECK-NOT: llvm.cond_br
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xi32, #mma>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #mma>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #mma>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #mma> -> tensor<256x64xi32, #mma>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #mma> -> tensor<256x64xi32, #mma>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #mma>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #mma>
    %10 = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<f16>, #mma>, tensor<256x64xi32, #mma>
    // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2
    %11 = tt.load %10 {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #mma>

    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma_1}>>
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma_1}>> -> tensor<256x1xi32, #mma_1>
    %22 = arith.constant dense<64> : tensor<256x1xi32, #mma_1>
    %23 = arith.muli %21, %22 : tensor<256x1xi32, #mma_1>
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma_1}>>
    %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma_1}>> -> tensor<1x64xi32, #mma_1>
    %26 = tt.broadcast %23 : tensor<256x1xi32, #mma_1> -> tensor<256x64xi32, #mma_1>
    %27 = tt.broadcast %25 : tensor<1x64xi32, #mma_1> -> tensor<256x64xi32, #mma_1>
    %28 = arith.addi %26, %27 : tensor<256x64xi32, #mma_1>
    %29 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #mma_1>
    %30 = tt.addptr %29, %28 : tensor<256x64x!tt.ptr<f16>, #mma_1>, tensor<256x64xi32, #mma_1>
    // CHECK-COUNT-16: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1
    %31 = tt.load %30 {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #mma_1>

    %40 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma_2}>>
    %41 = tt.expand_dims %40 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma_2}>> -> tensor<256x1xi32, #mma_2>
    %42 = arith.constant dense<64> : tensor<256x1xi32, #mma_2>
    %43 = arith.muli %41, %42 : tensor<256x1xi32, #mma_2>
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma_2}>>
    %45 = tt.expand_dims %44 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma_2}>> -> tensor<1x64xi32, #mma_2>
    %46 = tt.broadcast %43 : tensor<256x1xi32, #mma_2> -> tensor<256x64xi32, #mma_2>
    %47 = tt.broadcast %45 : tensor<1x64xi32, #mma_2> -> tensor<256x64xi32, #mma_2>
    %48 = arith.addi %46, %47 : tensor<256x64xi32, #mma_2>
    %49 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #mma_2>
    %50 = tt.addptr %49, %48 : tensor<256x64x!tt.ptr<f16>, #mma_2>, tensor<256x64xi32, #mma_2>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2
    %51 = tt.load %50 {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #mma_2>

    %60 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma_2}>>
    %61 = tt.expand_dims %60 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma_2}>> -> tensor<128x1xi32, #mma_2>
    %62 = arith.constant dense<64> : tensor<128x1xi32, #mma_2>
    %63 = arith.muli %61, %62 : tensor<128x1xi32, #mma_2>
    %66 = tt.broadcast %63 : tensor<128x1xi32, #mma_2> -> tensor<128x64xi32, #mma_2>
    %67 = tt.broadcast %45 : tensor<1x64xi32, #mma_2> -> tensor<128x64xi32, #mma_2>
    %68 = arith.addi %66, %67 : tensor<128x64xi32, #mma_2>
    %69 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #mma_2>
    %70 = tt.addptr %69, %68 : tensor<128x64x!tt.ptr<f16>, #mma_2>, tensor<128x64xi32, #mma_2>
    // COM: The data is duplicated in the warps because the warp shape is 32*8=256 larger than the tensor shape 128
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2
    %71 = tt.load %70 {ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #mma_2>
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
#mma_1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 8, 1], repCluster = [1, 2, 2]}>
#mma_32 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.support_sg_2d_block, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_gather_io
  tt.func public @regular_pointer_gather_io(%arg0: tensor<128x64x!tt.ptr<f16>, #mma>,
                                            %arg1: tensor<128x64x!tt.ptr<f16>, #mma_32>,
                                            %arg2: tensor<2x128x64x!tt.ptr<f16>, #mma_1>) {
    // COM: The pitch is not available in the current implementation.
    // COM: Not from axis info or ptrs[{0, 0}] and ptrs[{1, 0}] in the same work item.
    // CHECK-COUNT-32: llvm.load {{.*}} {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    %0 = tt.load %arg1 {ttig.block_io = "row_major"} : tensor<128x64x!tt.ptr<f16>, #mma_32>

    // COM: Not support column major block io.
    // CHECK-COUNT-32: llvm.load {{.*}} {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    %1 = tt.load %arg0 {ttig.block_io = "column_major"} : tensor<128x64x!tt.ptr<f16>, #mma>

    // COM: Not support rank size > 2.
    // CHECK-COUNT-128: llvm.load {{.*}} {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    %2 = tt.load %arg2 {ttig.block_io = "column_major"} : tensor<2x128x64x!tt.ptr<f16>, #mma_1>
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.support_sg_2d_block, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<f16>) {

    %a_mask = arith.constant dense<true> : tensor<256x64xi1, #mma>
    %a_other = arith.constant dense<1.00e+00> : tensor<256x64xf16, #mma>
    // CHECK-NOT: llvm.cond_br

    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xi32, #mma>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #mma>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #mma>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #mma> -> tensor<256x64xi32, #mma>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #mma> -> tensor<256x64xi32, #mma>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #mma>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #mma>
    %10 = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<f16>, #mma>, tensor<256x64xi32, #mma>

    // CHECK: %[[TOP_LEFT_MASK_BOOL_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i1, i1, {{.*}}
    // CHECK: %[[TOP_LEFT_MASK_BOOL_32:.*]] = llvm.extractvalue {{.*}}[32] : !llvm.struct<(i1, i1, {{.*}}
    // CHECK: %[[TOP_LEFT_MASK_BOOL_64:.*]] = llvm.extractvalue {{.*}}[64] : !llvm.struct<(i1, i1, {{.*}}
    // CHECK: %[[TOP_LEFT_MASK_BOOL_96:.*]] = llvm.extractvalue {{.*}}[96] : !llvm.struct<(i1, i1, {{.*}}

    // CHECK: %[[BLOCK_SHAPE_Y:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: %[[TOP_LEFT_PTR:.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_2886:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[TOP_LEFT_PTR]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[UNIFORM_PTR:.*]] = llvm.inttoptr %[[VAL_2886]] : i64 to !llvm.ptr<1>
    // CHECK: %[[CST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[TOP_LEFT_MASK:.*]] = llvm.zext %[[TOP_LEFT_MASK_BOOL_0]] : i1 to i8
    // CHECK: %[[PRED:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[TOP_LEFT_MASK]], %[[CST0_2]])
    // CHECK: %[[PRED_BOOL:.*]] =  llvm.trunc %[[PRED]] : i8 to i1
    // CHECK: %[[BASE_Y_0:.*]] = llvm.select %[[PRED_BOOL]], %[[CST0_1]], %[[BLOCK_SHAPE_Y]] : i1, i32
    // CHECK: %[[LOAD_0:.*]] = triton_gen.2Dblockload {{.*}}, %[[BASE_Y_0]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>

    // CHECK: %[[TOP_LEFT_PTR:.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_3046:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[TOP_LEFT_PTR]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[UNIFORM_PTR:.*]] = llvm.inttoptr %[[VAL_3046]] : i64 to !llvm.ptr<1>
    // CHECK: %[[CST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[TOP_LEFT_MASK:.*]] = llvm.zext %[[TOP_LEFT_MASK_BOOL_32]] : i1 to i8
    // CHECK: %[[PRED:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[TOP_LEFT_MASK]], %[[CST0_2]])
    // CHECK: %[[PRED_BOOL:.*]] =  llvm.trunc %[[PRED]] : i8 to i1
    // CHECK: %[[BASE_Y_0:.*]] = llvm.select %[[PRED_BOOL]], %[[CST0_1]], %[[BLOCK_SHAPE_Y]] : i1, i32
    // CHECK: %[[LOAD_0:.*]] = triton_gen.2Dblockload {{.*}}, %[[BASE_Y_0]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>

    // CHECK: %[[TOP_LEFT_PTR:.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_3046:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[TOP_LEFT_PTR]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[UNIFORM_PTR:.*]] = llvm.inttoptr %[[VAL_3046]] : i64 to !llvm.ptr<1>
    // CHECK: %[[CST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[TOP_LEFT_MASK:.*]] = llvm.zext %[[TOP_LEFT_MASK_BOOL_64]] : i1 to i8
    // CHECK: %[[PRED:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[TOP_LEFT_MASK]], %[[CST0_2]])
    // CHECK: %[[PRED_BOOL:.*]] =  llvm.trunc %[[PRED]] : i8 to i1
    // CHECK: %[[BASE_Y_0:.*]] = llvm.select %[[PRED_BOOL]], %[[CST0_1]], %[[BLOCK_SHAPE_Y]] : i1, i32
    // CHECK: %[[LOAD_0:.*]] = triton_gen.2Dblockload {{.*}}, %[[BASE_Y_0]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>

    // CHECK: %[[TOP_LEFT_PTR:.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr<1> to i64
    // CHECK: %[[VAL_3046:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[TOP_LEFT_PTR]], {{.*}}) {convergent, no_unwind, will_return} : (i64, i32) -> i64
    // CHECK: %[[UNIFORM_PTR:.*]] = llvm.inttoptr %[[VAL_3046]] : i64 to !llvm.ptr<1>
    // CHECK: %[[CST0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[CST0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[TOP_LEFT_MASK:.*]] = llvm.zext %[[TOP_LEFT_MASK_BOOL_96]] : i1 to i8
    // CHECK: %[[PRED:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[TOP_LEFT_MASK]], %[[CST0_2]])
    // CHECK: %[[PRED_BOOL:.*]] =  llvm.trunc %[[PRED]] : i8 to i1
    // CHECK: %[[BASE_Y_0:.*]] = llvm.select %[[PRED_BOOL]], %[[CST0_1]], %[[BLOCK_SHAPE_Y]] : i1, i32
    // CHECK: %[[LOAD_0:.*]] = triton_gen.2Dblockload {{.*}}, %[[BASE_Y_0]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    // CHECK: %[[DECOMPOSED_DATA:.*]] = llvm.shufflevector %[[LOAD_0]], %[[LOAD_0]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi16>
    // CHECK-NEXT: %[[UNPACKED_TYPE:.*]] = llvm.bitcast %[[DECOMPOSED_DATA]] : vector<8xi16> to vector<8xf16>
    // CHECK: llvm.select %[[PRED_BOOL]], %[[UNPACKED_TYPE]], {{.*}} : i1, vector<8xf16>
    %11 = tt.load %10, %a_mask, %a_other {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f16>, #mma>

    tt.return
  }
}

// -----

// COM: Check codegen when base height is 1 and tile height is > 1.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  // CHECK-LABEL: @baseheight1
  tt.func public @baseheight1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>>
    %19 = tt.expand_dims %18 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>}>> -> tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %20 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %21 = tt.addptr %20, %19 : tensor<1x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<1x32xi32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %22 = tt.broadcast %21 : tensor<1x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> -> tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %50 = tt.load %22 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f32>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[LOAD:%.*]] = triton_gen.2Dblockload %{{.*}}, %{{.*}}, [[C1]], %{{.*}}, %{{.*}}, %{{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 2

    // CHECK: [[VEC:%.*]] = llvm.mlir.undef : vector<2xi32>

    // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[OLDVAL:%.*]] = llvm.extractelement [[LOAD]][[[C0]] : i32] : vector<16xi32>
    // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[THREADID_i64:%.*]] = llvm.call spir_funccc @_Z12get_local_idj([[C0]])
    // CHECK: [[THREADID:%.*]] = llvm.trunc [[THREADID_i64]] : i64 to i32
    // CHECK: [[C127:%.*]] = llvm.mlir.constant(127 : i32) : i32
    // CHECK: [[RTID:%.*]] = llvm.and [[THREADID]], [[C127]] : i32
    // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: [[REM:%.*]] = llvm.urem [[RTID]], [[C8]] : i32

    // CHECK: [[NEWVAL:%.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij([[OLDVAL]], [[REM]])
    // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VEC1:%.*]] = llvm.insertelement [[NEWVAL]], [[VEC]][[[C0]] : i32] : vector<2xi32>

    // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: [[OLDVAL:%.*]] = llvm.extractelement [[LOAD]][[[C8]] : i32] : vector<16xi32>
    // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[THREADID_i64:%.*]] = llvm.call spir_funccc @_Z12get_local_idj([[C0]])
    // CHECK: [[THREADID:%.*]] = llvm.trunc [[THREADID_i64]] : i64 to i32
    // CHECK: [[C127:%.*]] = llvm.mlir.constant(127 : i32) : i32
    // CHECK: [[RTID:%.*]] = llvm.and [[THREADID]], [[C127]] : i32
    // CHECK: [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: [[REM:%.*]] = llvm.urem [[RTID]], [[C8]] : i32
    // CHECK: [[NEWVAL:%.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij([[OLDVAL]], [[REM]])
    // CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[VEC2:%.*]] = llvm.insertelement [[NEWVAL]], [[VEC1]][[[C1]] : i32] : vector<2xi32>

    // CHECK: llvm.shufflevector [[VEC2]], [[VEC2]] [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_sg_2d_block, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_a}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_a}>> -> tensor<256x1xi32, #dot_a>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #dot_a>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #dot_a>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_a}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_a}>> -> tensor<1x64xi32, #dot_a>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #dot_a> -> tensor<256x64xi32, #dot_a>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dot_a> -> tensor<256x64xi32, #dot_a>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #dot_a>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>, #dot_a>
    %10 = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<f32>, #dot_a>, tensor<256x64xi32, #dot_a>
    // COM: 32 x 4 = 128 bytes, which is >64 bytes
    // CHECK-NOT:    triton_gen.2Dblockload
    %11 = tt.load %10 {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f32>, #dot_a>

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.support_sg_2d_block} {
  tt.func public @trans_block_load_i32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<1x64xi32, #blocked>
    %8 = arith.muli %4, %cst_0 : tensor<1x64xi32, #blocked>
    %9 = tt.broadcast %1 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %10 = tt.broadcast %8 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %11 = arith.addi %9, %10 : tensor<32x64xi32, #blocked>
    %12 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %13 = tt.addptr %12, %11 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>
    // COM: Transpose 2D block load with i32 type.
    // CHECK-COUNT-16: triton_gen.2Dblockload {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} {elem_size_in_bits = 32, tile_width = 2, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<1xi32>
    %14 = tt.load %13 {ttig.block_io = "column_major"} : tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 2], A = [8, 32], B = [32, 32], C = [8, 32]}>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  tt.func public @trans_block_load_i16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<64> : tensor<32x1xi32, #mma>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<32x1xi32, #mma>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %cst_0 = arith.constant dense<32> : tensor<1x64xi32, #mma>
    %8 = arith.muli %4, %cst_0 : tensor<1x64xi32, #mma>
    %9 = tt.broadcast %1 : tensor<32x1xi32, #mma> -> tensor<32x64xi32, #mma>
    %10 = tt.broadcast %8 : tensor<1x64xi32, #mma> -> tensor<32x64xi32, #mma>
    %11 = arith.addi %9, %10 : tensor<32x64xi32, #mma>
    %12 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #mma>
    %13 = tt.addptr %12, %11 : tensor<32x64x!tt.ptr<f16>, #mma>, tensor<32x64xi32, #mma>
    // COM: Transpose 2D block load with f16 type. Pack the loaded vector to the i32 type. Then transpose the loaded i32 vector with bitcast op.
    // CHECK: %[[LOADED:.*]] = triton_gen.2Dblockload {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
    // CHECK: %[[PACKED_I32:.*]] = llvm.shufflevector %[[LOADED]], %[[LOADED]] [0, 1, 2, 3] : vector<8xi32>
    // CHECK: llvm.bitcast %[[PACKED_I32]] : vector<4xi32> to vector<8xf16>
    // CHECK-COUNT-3: triton_gen.2Dblockload {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
    %14 = tt.load %13 {ttig.block_io = "column_major"} : tensor<32x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 2], A = [8, 32], B = [32, 32], C = [8, 32]}>
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  tt.func public @trans_block_load_i8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<128> : tensor<128x1xi32, #mma>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %2 = arith.muli %1, %cst : tensor<128x1xi32, #mma>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %5 = tt.broadcast %2 : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
    %6 = tt.broadcast %4 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %7 = arith.addi %5, %6 : tensor<128x128xi32, #mma>
    %cst_0 = arith.constant dense<128> : tensor<1x128xi32, #mma>
    %8 = arith.muli %4, %cst_0 : tensor<1x128xi32, #mma>
    %9 = tt.broadcast %1 : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
    %10 = tt.broadcast %8 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %11 = arith.addi %9, %10 : tensor<128x128xi32, #mma>
    %12 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #mma>
    %13 = tt.addptr %12, %11 : tensor<128x128x!tt.ptr<i8>, #mma>, tensor<128x128xi32, #mma>
    // COM: Transpose 2D block load with i8 type. Pack the loaded vector to the i32 type. Then transpose the loaded i32 vector with bitcast op.
    // CHECK: %[[LOADED:.*]] = triton_gen.2Dblockload {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
    // COM: We do the shuffle and then the bitcast. Maybe it is efficient to do bitcast first then shuffle?
    // CHECK: %[[PACKED_1ST_HALF:.*]] = llvm.shufflevector %[[LOADED]], %[[LOADED]] [0, 1] : vector<8xi32>
    // CHECK: llvm.bitcast %[[PACKED_1ST_HALF]] : vector<2xi32> to vector<8xi8>
    // CHECK: %[[PACKED_2ND_HALF:.*]] = llvm.shufflevector %[[LOADED]], %[[LOADED]] [2, 3] : vector<8xi32>
    // CHECK: llvm.bitcast %[[PACKED_2ND_HALF]] : vector<2xi32> to vector<8xi8>
    // CHECK-COUNT-7: triton_gen.2Dblockload {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}},  {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
    %14 = tt.load %13 {ttig.block_io = "column_major"} : tensor<128x128x!tt.ptr<i8>, #mma>
    tt.return
  }
}
