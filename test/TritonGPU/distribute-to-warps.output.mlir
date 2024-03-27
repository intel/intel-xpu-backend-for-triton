// RUN: triton-opt %s -split-input-file | FileCheck %s

#warp = #triton_gpu.warp<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#warp1 = #triton_gpu.warp<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
//       CHECK: gpu.subgroup_id : index
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_with_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #warp>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.addi %arg3, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg4, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %2, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %4, %9 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %9, %12 : i32
    %14 = arith.remsi %2, %7 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = arith.extsi %arg3 : i32 to i64
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = arith.extsi %arg6 : i32 to i64
    %20 = arith.muli %1, %c32_i32 : i32
    %21 = arith.addi %20, %16 : i32
    %22 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #warp1>, 1>
    %23 = arith.muli %15, %c128_i32 : i32
    %24 = arith.extsi %arg4 : i32 to i64
    %25 = arith.extsi %arg7 : i32 to i64
    %26 = arith.addi %20, %23 : i32
    %27 = tt.make_tensor_ptr %arg1, [%18, %24], [%25, %c1_i64], [%c0_i32, %26] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #warp1>, 1>
    %28:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %22, %arg12 = %27) -> (tensor<64x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #warp1>, 1>, !tt.ptr<tensor<32x32xf16, #warp1>, 1>)  : i32 {
      %40 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #warp1>, 1> -> tensor<32x32xf16, #warp1>
      %41 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #warp1>, 1> -> tensor<32x32xf16, #warp1>
      %42 = triton_gpu.alloc : <f16, 1>
      %43 = tt.make_tensor_ptr %42, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%20, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #warp1>, 3>
      tt.store %43, %40 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x32xf16, #warp1>, 3>, tensor<32x32xf16, #warp1>
      gpu.barrier
      %44 = arith.divsi %1, %c2_i32 : i32
      %45 = arith.remsi %44, %c2_i32 : i32
      %46 = arith.muli %45, %c64_i32 : i32
      %47 = tt.make_tensor_ptr %42, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%46, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 3>
      %48 = tt.load %47 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 3> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>
      %49 = triton_gpu.alloc : <f16, 1>
      %50 = tt.make_tensor_ptr %49, [%c32_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %20] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #warp1>, 3>
      tt.store %50, %41 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x32xf16, #warp1>, 3>, tensor<32x32xf16, #warp1>
      gpu.barrier
      %51 = arith.remsi %1, %c2_i32 : i32
      %52 = arith.remsi %51, %c2_i32 : i32
      %53 = arith.muli %52, %c64_i32 : i32
      %54 = tt.make_tensor_ptr %49, [%c32_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %53] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 3>
      %55 = tt.load %54 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 3> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>
      %56 = tt.dot %48, %55, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<64x64xf32, #warp>
      %57 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #warp1>, 1>
      %58 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x32xf16, #warp1>, 1>
      scf.yield %56, %57, %58 : tensor<64x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #warp1>, 1>, !tt.ptr<tensor<32x32xf16, #warp1>, 1>
    }
    %29 = arith.truncf %28#0 : tensor<64x64xf32, #warp> to tensor<64x64xf16, #warp>
    %30 = arith.extsi %arg8 : i32 to i64
    %31 = arith.divsi %1, %c2_i32 : i32
    %32 = arith.remsi %31, %c2_i32 : i32
    %33 = arith.muli %32, %c64_i32 : i32
    %34 = arith.addi %33, %16 : i32
    %35 = arith.remsi %1, %c2_i32 : i32
    %36 = arith.remsi %35, %c2_i32 : i32
    %37 = arith.muli %36, %c64_i32 : i32
    %38 = arith.addi %37, %23 : i32
    %39 = tt.make_tensor_ptr %arg2, [%17, %24], [%30, %c1_i64], [%34, %38] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #warp>, 1>
    tt.store %39, %29 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x64xf16, #warp>, 1>, tensor<64x64xf16, #warp>
    tt.return
  }
  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #warp>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.addi %arg3, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg4, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %2, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %4, %9 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %9, %12 : i32
    %14 = arith.remsi %2, %7 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = arith.extsi %arg3 : i32 to i64
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = arith.extsi %arg6 : i32 to i64
    %20 = arith.divsi %1, %c2_i32 : i32
    %21 = arith.remsi %20, %c2_i32 : i32
    %22 = arith.muli %21, %c64_i32 : i32
    %23 = arith.addi %22, %16 : i32
    %24 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 1>
    %25 = arith.muli %15, %c128_i32 : i32
    %26 = arith.extsi %arg4 : i32 to i64
    %27 = arith.extsi %arg7 : i32 to i64
    %28 = arith.remsi %1, %c2_i32 : i32
    %29 = arith.remsi %28, %c2_i32 : i32
    %30 = arith.muli %29, %c64_i32 : i32
    %31 = arith.addi %30, %25 : i32
    %32 = tt.make_tensor_ptr %arg1, [%18, %26], [%27, %c1_i64], [%c0_i32, %31] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 1>
    %33:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %24, %arg12 = %32) -> (tensor<64x64xf32, #warp>, !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 1>, !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 1>)  : i32 {
      %37 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 1> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>
      %38 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 1> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>
      %39 = tt.dot %37, %38, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<64x64xf32, #warp>
      %40 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 1>
      %41 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 1>
      scf.yield %39, %40, %41 : tensor<64x64xf32, #warp>, !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>, 1>, !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>, 1>
    }
    %34 = arith.truncf %33#0 : tensor<64x64xf32, #warp> to tensor<64x64xf16, #warp>
    %35 = arith.extsi %arg8 : i32 to i64
    %36 = tt.make_tensor_ptr %arg2, [%17, %26], [%35, %c1_i64], [%23, %31] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #warp>, 1>
    tt.store %36, %34 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x64xf16, #warp>, 1>, tensor<64x64xf16, #warp>
    tt.return
  }
}
