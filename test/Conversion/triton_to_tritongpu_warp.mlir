// RUN: triton-opt %s -split-input-file --convert-triton-to-tritongpu-warp="num-warps=32"  | FileCheck %s
// CHECK: #triton_gpu.blocked<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: "triton_gpu.num-warps" = 32
module {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK: tt.load
    // CHECK-SAME: tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}
    // CHECK: tt.load
    // CHECK-SAME: tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}
    // CHECK: tt.dot
    // CHECK-SAME: -> tensor<256x256xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c16_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %4 : i32
    %6 = arith.addi %2, %5 : i32
    %7 = arith.remsi %0, %c64_i32 : i32
    %8 = arith.divsi %7, %4 : i32
    %9 = arith.muli %6, %c256_i32 : i32
    %10 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16>, 1>
    %11 = arith.muli %8, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %11] {order = array<i32: 1, 0>} : <tensor<32x256xf16>, 1>
    %13:3 = scf.for %arg6 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg7 = %cst, %arg8 = %10, %arg9 = %12) -> (tensor<256x256xf32>, !tt.ptr<tensor<256x32xf16>, 1>, !tt.ptr<tensor<32x256xf16>, 1>)  : i32 {
      %15 = tt.load %arg8 : !tt.ptr<tensor<256x32xf16>, 1>
      %16 = tt.load %arg9 : !tt.ptr<tensor<32x256xf16>, 1>
      %17 = tt.dot %15, %16, %arg7 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16> * tensor<32x256xf16> -> tensor<256x256xf32>
      %18 = tt.advance %arg8, [%c0_i32, %c32_i32] : <tensor<256x32xf16>, 1>
      %19 = tt.advance %arg9, [%c32_i32, %c0_i32] : <tensor<32x256xf16>, 1>
      scf.yield %17, %18, %19 : tensor<256x256xf32>, !tt.ptr<tensor<256x32xf16>, 1>, !tt.ptr<tensor<32x256xf16>, 1>
    }
    %14 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %11] {order = array<i32: 1, 0>} : <tensor<256x256xf32>, 1>
    tt.store %14, %13#0 : !tt.ptr<tensor<256x256xf32>, 1>
    tt.return
  }
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16, 1>, %arg1: !tt.ptr<f16, 1>, %arg2: !tt.ptr<f16, 1>, %arg3: f32, %arg4: !tt.ptr<f32, 1>, %arg5: !tt.ptr<f32, 1>) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %c65536_i64 = arith.constant 65536 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %cst_2 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c2_i32 : i32
    %3 = arith.remsi %1, %c2_i32 : i32
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.muli %4, %c131072_i64 : i64
    %6 = arith.extsi %3 : i32 to i64
    %7 = arith.muli %6, %c65536_i64 : i64
    %8 = arith.addi %5, %7 : i64
    %9 = tt.addptr %arg0, %8 : !tt.ptr<f16, 1>, i64
    %10 = arith.muli %0, %c128_i32 : i32
    %11 = tt.make_tensor_ptr %9, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%10, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16>, 1>
    %12 = tt.addptr %arg2, %8 : !tt.ptr<f16, 1>, i64
    %13 = tt.make_tensor_ptr %12, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16>, 1>
    %14 = tt.addptr %arg1, %8 : !tt.ptr<f16, 1>, i64
    %15 = tt.make_tensor_ptr %14, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16>, 1>
    %16 = tt.addptr %arg5, %8 : !tt.ptr<f32, 1>, i64
    %17 = tt.make_tensor_ptr %16, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%10, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf32>, 1>
    %18 = arith.mulf %arg3, %cst_2 : f32
    %19 = tt.load %11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xf16>, 1>
    %20 = tt.splat %18 : f32 -> tensor<128xf32>
    %21 = tt.splat %18 : f32 -> tensor<128x64xf32>
    %22:5 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0, %arg10 = %13, %arg11 = %15) -> (tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, !tt.ptr<tensor<64x64xf16>, 1>, !tt.ptr<tensor<64x64xf16>, 1>)  : i32 {
      %26 = tt.load %arg11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x64xf16>, 1>
      %27 = tt.dot %19, %26, %cst_1 {maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
      %28 = "tt.reduce"(%27) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %49 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %49 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %29 = arith.mulf %28, %20 : tensor<128xf32>
      %30 = arith.maxnumf %arg9, %29 : tensor<128xf32>
      %31 = arith.mulf %27, %21 : tensor<128x64xf32>
      %32 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %33 = tt.broadcast %32 : tensor<128x1xf32> -> tensor<128x64xf32>
      %34 = arith.subf %31, %33 : tensor<128x64xf32>
      %35 = math.exp2 %34 : tensor<128x64xf32>
      %36 = "tt.reduce"(%35) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %49 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %49 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %37 = arith.subf %arg9, %30 : tensor<128xf32>
      %38 = math.exp2 %37 : tensor<128xf32>
      %39 = arith.mulf %arg7, %38 : tensor<128xf32>
      %40 = arith.addf %39, %36 : tensor<128xf32>
      %41 = tt.expand_dims %38 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %42 = tt.broadcast %41 : tensor<128x1xf32> -> tensor<128x64xf32>
      %43 = arith.mulf %arg8, %42 : tensor<128x64xf32>
      %44 = tt.load %arg10 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x64xf16>, 1>
      %45 = arith.truncf %35 : tensor<128x64xf32> to tensor<128x64xf16>
      %46 = tt.dot %45, %44, %43 {maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
      %47 = tt.advance %arg10, [%c64_i32, %c0_i32] : <tensor<64x64xf16>, 1>
      %48 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16>, 1>
      scf.yield %40, %46, %30, %47, %48 : tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, !tt.ptr<tensor<64x64xf16>, 1>, !tt.ptr<tensor<64x64xf16>, 1>
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %23 = tt.expand_dims %22#0 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %24 = tt.broadcast %23 : tensor<128x1xf32> -> tensor<128x64xf32>
    %25 = arith.divf %22#1, %24 : tensor<128x64xf32>
    tt.store %17, %25 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xf32>, 1>
    tt.return
  }
}
