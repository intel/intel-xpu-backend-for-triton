// RUN: triton-opt %s -split-input-file -tritongpu-distribute-to-warps -canonicalize -cse | FileCheck %s

#blocked1 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blockedC = #triton_gpu.blocked<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blockedA = #triton_gpu.dot_op<{opIdx = 0, parent = #blockedC}>
#blockedB = #triton_gpu.dot_op<{opIdx = 1, parent = #blockedC}>

//       CHECK: gpu.subgroup_id : index
// in the loop body:
// thread block works on 128x128xf32 = 128x32xf16 * 32x128xf16
//    each warp works on   64x64xf32 =  64x32xf16 *  32x64xf16
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_with_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blockedC>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16, #blocked1>, 1>
    %19 = arith.muli %13, %c128_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x128xf16, #blocked2>, 1>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x128xf32, #blockedC>, !tt.ptr<tensor<128x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x128xf16, #blocked2>, 1>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x32xf16, #blocked1>, 1> -> tensor<128x32xf16, #blocked1>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x128xf16, #blocked2>, 1> -> tensor<32x128xf16, #blocked2>
      %30 = triton_gpu.convert_layout %28 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blockedA>
      %31 = triton_gpu.convert_layout %29 : tensor<32x128xf16, #blocked2> -> tensor<32x128xf16, #blockedB>
      %32 = tt.dot %30, %31, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #blockedA> * tensor<32x128xf16, #blockedB> -> tensor<128x128xf32, #blockedC>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<128x32xf16, #blocked1>, 1>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x128xf16, #blocked2>, 1>
      scf.yield %32, %33, %34 : tensor<128x128xf32, #blockedC>, !tt.ptr<tensor<128x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x128xf16, #blocked2>, 1>
    }
    %24 = arith.truncf %23#0 : tensor<128x128xf32, #blockedC> to tensor<128x128xf16, #blockedC>
    %25 = arith.extsi %arg8 : i32 to i64
    %26 = tt.make_tensor_ptr %arg2, [%15, %20], [%25, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blockedC>, 1>
    tt.store %26, %24 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x128xf16, #blockedC>, 1>, tensor<128x128xf16, #blockedC>
    tt.return
  }

  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blockedC>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c127_i32 = arith.constant 127 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16, #blockedA>, 1>
    %19 = arith.muli %13, %c128_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x128xf16, #blockedB>, 1>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x128xf32, #blockedC>, !tt.ptr<tensor<128x32xf16, #blockedA>, 1>, !tt.ptr<tensor<32x128xf16, #blockedB>, 1>)  : i32 {
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x32xf16, #blockedA>, 1> -> tensor<128x32xf16, #blockedA>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x128xf16, #blockedB>, 1> -> tensor<32x128xf16, #blockedB>
      %32 = tt.dot %28, %29, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #blockedA> * tensor<32x128xf16, #blockedB> -> tensor<128x128xf32, #blockedC>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<128x32xf16, #blockedA>, 1>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x128xf16, #blockedB>, 1>
      scf.yield %32, %33, %34 : tensor<128x128xf32, #blockedC>, !tt.ptr<tensor<128x32xf16, #blockedA>, 1>, !tt.ptr<tensor<32x128xf16, #blockedB>, 1>
    }
    %24 = arith.truncf %23#0 : tensor<128x128xf32, #blockedC> to tensor<128x128xf16, #blockedC>
    %25 = arith.extsi %arg8 : i32 to i64
    %26 = tt.make_tensor_ptr %arg2, [%15, %20], [%25, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blockedC>, 1>
    tt.store %26, %24 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x128xf16, #blockedC>, 1>, tensor<128x128xf16, #blockedC>
    tt.return
  }
}
