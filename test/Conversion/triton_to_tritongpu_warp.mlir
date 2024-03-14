// RUN: triton-opt %s -split-input-file --convert-triton-to-tritongpu-warp="num-warps=32"  | FileCheck %s
// CHECK: #triton_gpu.blocked<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], warpsPerCTA = [8, 4], order = [1, 0]}>
module {
  tt.func public @matmul_kernel_with_block_pointers_0d1d2d3d4d5d(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
      %15 = tt.load %arg8 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<256x32xf16>, 1> -> tensor<256x32xf16>
      %16 = tt.load %arg9 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x256xf16>, 1> -> tensor<32x256xf16>
      %17 = tt.dot %15, %16, %arg7 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16> * tensor<32x256xf16> -> tensor<256x256xf32>
      %18 = tt.advance %arg8, [%c0_i32, %c32_i32] : <tensor<256x32xf16>, 1>
      %19 = tt.advance %arg9, [%c32_i32, %c0_i32] : <tensor<32x256xf16>, 1>
      scf.yield %17, %18, %19 : tensor<256x256xf32>, !tt.ptr<tensor<256x32xf16>, 1>, !tt.ptr<tensor<32x256xf16>, 1>
    }
    %14 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %11] {order = array<i32: 1, 0>} : <tensor<256x256xf32>, 1>
    tt.store %14, %13#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<256x256xf32>, 1>, tensor<256x256xf32>
    tt.return
  }
}
