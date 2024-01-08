module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func @test_kernel(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c1024_i64], [%c1024_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %1 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %2 = tt.make_tensor_ptr %arg1, [%c1024_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %3:5 = scf.for %arg3 = %c0_i32 to %c1024_i32 step %c16_i32 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %0, %arg7 = %1, %arg8 = %2) -> (tensor<16x16xf32>, tensor<16x16xf32>, !tt.ptr<tensor<16x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>)  : i32 {
      %8 = tt.load %arg6 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x16xf16>, 1> -> tensor<16x16xf16>
      %9 = tt.load %arg7 {DotB = true, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x16xf16>, 1> -> tensor<16x16xf16>
      %10 = tt.load %arg8 {DotB = true, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x16xf16>, 1> -> tensor<16x16xf16>
      %11 = tt.extract %arg4, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %12 = tt.extract %8, 0 : tensor<16x16xf16> -> tensor<8x16xf16>
      %13 = tt.dot %12, %9, %11 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %14 = tt.extract %arg4, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %15 = tt.extract %8, 1 : tensor<16x16xf16> -> tensor<8x16xf16>
      %16 = tt.dot %15, %9, %14 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %17 = tt.glue %13, %16 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %18 = tt.extract %arg5, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %19 = tt.extract %8, 0 : tensor<16x16xf16> -> tensor<8x16xf16>
      %20 = tt.dot %19, %10, %18 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %21 = tt.extract %arg5, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %22 = tt.extract %8, 1 : tensor<16x16xf16> -> tensor<8x16xf16>
      %23 = tt.dot %22, %10, %21 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %24 = tt.glue %20, %23 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %25 = tt.advance %arg6, [%c0_i32, %c16_i32] : <tensor<16x16xf16>, 1>
      %26 = tt.advance %arg7, [%c16_i32, %c0_i32] : <tensor<16x16xf16>, 1>
      %27 = tt.advance %arg8, [%c16_i32, %c0_i32] : <tensor<16x16xf16>, 1>
      scf.yield %17, %24, %25, %26, %27 : tensor<16x16xf32>, tensor<16x16xf32>, !tt.ptr<tensor<16x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>
    }
    %4 = arith.truncf %3#0 : tensor<16x16xf32> to tensor<16x16xf16>
    %5 = arith.truncf %3#1 : tensor<16x16xf32> to tensor<16x16xf16>
    %6 = tt.make_tensor_ptr %arg2, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %7 = tt.make_tensor_ptr %arg2, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c16_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    tt.store %6, %4 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %7, %5 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.return
  }
}

