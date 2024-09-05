// RUN: triton-opt %s -split-input-file --convert-triton-to-tritongpu-warp="num-warps=32"  | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s -split-input-file --convert-triton-to-tritongpu-warp="num-warps=8"  | FileCheck %s --check-prefix=CHECK1

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
    // CHECK: {triton_gpu.workload = 3 : i32}
    }
    %14 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %11] {order = array<i32: 1, 0>} : <tensor<256x256xf32>, 1>
    tt.store %14, %13#0 : !tt.ptr<tensor<256x256xf32>, 1>
    tt.return
  }
}

// -----

// CHECK1: #triton_gpu.blocked<{sizePerThread = [8, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 8], order = [1, 0]}>
// CHECK1: "triton_gpu.num-warps" = 8
module {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK1: tt.load
    // CHECK1-SAME: tensor<8x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}
    // CHECK1: tt.load
    // CHECK1-SAME: tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}
    // CHECK1: tt.dot
    // CHECK1-SAME: -> tensor<8x256xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x256xf32>
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
    %10 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xf16>, 1>
    %11 = arith.muli %8, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %11] {order = array<i32: 1, 0>} : <tensor<32x256xf16>, 1>
    %13:3 = scf.for %arg6 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg7 = %cst, %arg8 = %10, %arg9 = %12) -> (tensor<8x256xf32>, !tt.ptr<tensor<8x32xf16>, 1>, !tt.ptr<tensor<32x256xf16>, 1>)  : i32 {
      %15 = tt.load %arg8 : !tt.ptr<tensor<8x32xf16>, 1>
      %16 = tt.load %arg9 : !tt.ptr<tensor<32x256xf16>, 1>
      %17 = tt.dot %15, %16, %arg7 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<8x32xf16> * tensor<32x256xf16> -> tensor<8x256xf32>
      %18 = tt.advance %arg8, [%c0_i32, %c32_i32] : <tensor<8x32xf16>, 1>
      %19 = tt.advance %arg9, [%c32_i32, %c0_i32] : <tensor<32x256xf16>, 1>
      scf.yield %17, %18, %19 : tensor<8x256xf32>, !tt.ptr<tensor<8x32xf16>, 1>, !tt.ptr<tensor<32x256xf16>, 1>
    // CHECK: {triton_gpu.workload = 3 : i32}
    }
    %14 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %11] {order = array<i32: 1, 0>} : <tensor<8x256xf32>, 1>
    tt.store %14, %13#0 : !tt.ptr<tensor<8x256xf32>, 1>
    tt.return
  }
}

// -----
// CHECK1: [[BLOCKED:#.*]] = #triton_gpu.blocked<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK1: "triton_gpu.num-warps" = 8
module {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: f32, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>) {
    // CHECK1: tt.load {{.*}} : !tt.ptr<tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[BLOCKED]]}>>>
    // CHECK1: tt.splat {{.*}} : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = [[BLOCKED]]}>>
    // CHECK1: tt.splat {{.*}} : f32 -> tensor<128x64xf32, [[BLOCKED]]>
    // CHECK1: tt.load {{.*}} : !tt.ptr<tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[BLOCKED]]}>>>
    // CHECK1: tt.dot {{.*}} -> tensor<128x64xf32, #blocked>
    // CHECK1: tt.load {{.*}} : !tt.ptr<tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[BLOCKED]]}>>>
    // CHECK1: tt.dot {{.*}} -> tensor<128x64xf32, [[BLOCKED]]>
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
    %0 = tt.get_program_id z : i32
    %1 = tt.get_program_id x : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %1 : i32 to i64
    %4 = arith.muli %3, %c131072_i64 : i64
    %5 = arith.extsi %2 : i32 to i64
    %6 = arith.muli %5, %c65536_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
    %9 = arith.muli %0, %c128_i32 : i32
    %10 = tt.make_tensor_ptr %8, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16>>
    %11 = tt.addptr %arg2, %7 : !tt.ptr<f16>, i64
    %12 = tt.make_tensor_ptr %11, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16>>
    %13 = tt.addptr %arg1, %7 : !tt.ptr<f16>, i64
    %14 = tt.make_tensor_ptr %13, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16>>
    %15 = tt.addptr %arg5, %7 : !tt.ptr<f32>, i64
    %16 = tt.make_tensor_ptr %15, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf32>>
    %17 = arith.mulf %arg3, %cst_2 : f32
    %18 = tt.load %10 : !tt.ptr<tensor<128x64xf16>>
    %19 = tt.splat %17 : f32 -> tensor<128xf32>
    %20 = tt.splat %17 : f32 -> tensor<128x64xf32>
    %21:5 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0, %arg10 = %12, %arg11 = %14) -> (tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, !tt.ptr<tensor<64x64xf16>>, !tt.ptr<tensor<64x64xf16>>)  : i32 {
      %25 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16>>
      %26 = tt.dot %18, %25, %cst_1, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
      %27 = "tt.reduce"(%26) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %48 = arith.maxnumf %arg12, %arg13 : f32
        tt.reduce.return %48 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %28 = arith.mulf %27, %19 : tensor<128xf32>
      %29 = arith.maxnumf %arg9, %28 : tensor<128xf32>
      %30 = arith.mulf %26, %20 : tensor<128x64xf32>
      %31 = tt.expand_dims %29 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %32 = tt.broadcast %31 : tensor<128x1xf32> -> tensor<128x64xf32>
      %33 = arith.subf %30, %32 : tensor<128x64xf32>
      %34 = math.exp2 %33 : tensor<128x64xf32>
      %35 = "tt.reduce"(%34) <{axis = 1 : i32}> ({
      ^bb0(%arg12: f32, %arg13: f32):
        %48 = arith.addf %arg12, %arg13 : f32
        tt.reduce.return %48 : f32
      }) : (tensor<128x64xf32>) -> tensor<128xf32>
      %36 = arith.subf %arg9, %29 : tensor<128xf32>
      %37 = math.exp2 %36 : tensor<128xf32>
      %38 = arith.mulf %arg7, %37 : tensor<128xf32>
      %39 = arith.addf %38, %35 : tensor<128xf32>
      %40 = tt.expand_dims %37 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %41 = tt.broadcast %40 : tensor<128x1xf32> -> tensor<128x64xf32>
      %42 = arith.mulf %arg8, %41 : tensor<128x64xf32>
      %43 = tt.load %arg10 : !tt.ptr<tensor<64x64xf16>>
      %44 = arith.truncf %34 : tensor<128x64xf32> to tensor<128x64xf16>
      %45 = tt.dot %44, %43, %42, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
      %46 = tt.advance %arg10, [%c64_i32, %c0_i32] : <tensor<64x64xf16>>
      %47 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16>>
      scf.yield %39, %45, %29, %46, %47 : tensor<128xf32>, tensor<128x64xf32>, tensor<128xf32>, !tt.ptr<tensor<64x64xf16>>, !tt.ptr<tensor<64x64xf16>>
    // CHECK: {triton_gpu.workload = 4 : i32}
    }
    %22 = tt.expand_dims %21#0 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    %23 = tt.broadcast %22 : tensor<128x1xf32> -> tensor<128x64xf32>
    %24 = arith.divf %21#1, %23 : tensor<128x64xf32>
    tt.store %16, %24 : !tt.ptr<tensor<128x64xf32>>
    tt.return
  }
}
