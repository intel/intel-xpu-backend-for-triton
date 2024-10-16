// RUN: triton-opt %s -split-input-file -tritonintelgpu-coalesce | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice1dim1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>
#slice2dim0 = #triton_gpu.slice<{dim = 0, parent = #blocked2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {

// CHECK: [[row_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: [[col_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: [[load_ptr:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32>, [[row_layout]]>
// CHECK: [[load_mask:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xi1, [[row_layout]]>
// CHECK: [[load_other:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xf32, [[row_layout]]>
// CHECK: [[load_val:%.*]] = tt.load [[load_ptr]], [[load_mask]], [[load_other]] : tensor<64x64x!tt.ptr<f32>, [[row_layout]]>
// CHECK: [[store_ptr:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32>, [[col_layout]]>
// CHECK: [[store_val:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xf32, [[col_layout]]>
// CHECK: [[store_mask:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xi1, [[col_layout]]>
// CHECK: tt.store [[store_ptr]], [[store_val]], [[store_mask]]
tt.func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                %arg1: i32 {tt.divisibility = 16 : i32},
                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                %arg3: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  %00 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1dim1>
  %01 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice2dim0>
  %1 = tt.expand_dims %00 {axis = 1 : i32} : tensor<64xi32, #slice1dim1> -> tensor<64x1xi32, #blocked1>
  %2 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked1>
  %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked1>
  %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %6 = tt.expand_dims %01 {axis = 0 : i32} : tensor<64xi32, #slice2dim0> -> tensor<1x64xi32, #blocked2>
  %7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %8 = tt.broadcast %6 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %9 = triton_gpu.convert_layout %8 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %10 = tt.addptr %7, %9 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %12 = tt.addptr %11, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %13 = tt.splat %arg3 : i32 -> tensor<1x64xi32, #blocked2>
  %14 = arith.muli %6, %13 : tensor<1x64xi32, #blocked2>
  %15 = tt.broadcast %12 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %16 = tt.broadcast %14 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %17 = triton_gpu.convert_layout %16 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %18 = tt.addptr %15, %17 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %19 = tt.load %10, %cst, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.store %18, %19, %cst : tensor<64x64x!tt.ptr<f32>, #blocked1>
  tt.return
}

}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {


// CHECK: [[NARROW_LAYOUT:#.*]] = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[WIDE_LAYOUT:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @load_tensors_two_types(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi "slt", %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %13 = arith.extf %12 : tensor<1024xf16, #blocked> to tensor<1024xf32, #blocked>
    %14 = arith.addf %9, %13 : tensor<1024xf32, #blocked>
    %15 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[WIDE_LAYOUT]]>
    tt.store %16, %14, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
}

}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-NOT: sizePerThread = [4]
// CHECK: #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-NOT: sizePerThread = [4]
tt.func public @load_tensors_two_types(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi "slt", %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    %13 = arith.extf %12 : tensor<1024xf16, #blocked> to tensor<1024xf32, #blocked>
    %14 = arith.addf %9, %13 : tensor<1024xf32, #blocked>
    %15 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %17 = arith.truncf %14 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    tt.store %16, %17, %6 : tensor<1024x!tt.ptr<f16>, #blocked>
    tt.return
}

}

// -----

// COM: Reproducer for issue #3866
// CHECK-LABEL: @test_3866
// CHECK: tt.load {{.*}} : !tt.ptr<tensor<64x16xf16>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @test_3866(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i64) {
    %0 = tt.make_tensor_ptr %arg0, [%arg2, %arg2], [%arg2, %arg2], [%arg1, %arg1] {order = array<i32: 1, 0>} : <tensor<64x16xf16>>
    %1 = tt.load %0 : !tt.ptr<tensor<64x16xf16>>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
#dot2 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK: [[BLOCKED_LAYOUT1:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK: [[BLOCKED_LAYOUT2:#.*]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK: @test_block_ptrs
  tt.func public @test_block_ptrs(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32, %arg20: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<8xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg19 : i32
    %3 = arith.remsi %1, %arg19 : i32
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.extsi %arg6 : i32 to i64
    %6 = arith.muli %4, %5 : i64
    %7 = arith.extsi %3 : i32 to i64
    %8 = arith.extsi %arg7 : i32 to i64
    %9 = arith.muli %7, %8 : i64
    %10 = arith.addi %6, %9 : i64
    %11 = tt.addptr %arg0, %10 : !tt.ptr<f8E5M2>, i64
    %12 = arith.muli %0, %c8_i32 : i32
    %13 = arith.extsi %arg20 : i32 to i64
    %14 = arith.extsi %arg8 : i32 to i64
    // CHECK: [[PTR1:%.*]] = tt.make_tensor_ptr {{.*}} : <tensor<8x64xf8E5M2, [[BLOCKED_LAYOUT1]]>
    %15 = tt.make_tensor_ptr %11, [%13, %c64_i64], [%14, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x64xf8E5M2, #dot1>>
    %16 = tt.addptr %arg1, %10 : !tt.ptr<f8E5M2>, i64
    %17 = arith.extsi %arg11 : i32 to i64
    // CHECK: [[PTR2:%.*]] = tt.make_tensor_ptr {{.*}} : <tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]>
    %18 = tt.make_tensor_ptr %16, [%c64_i64, %13], [%c1_i64, %17], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf8E5M2, #dot2>>
    %19 = tt.addptr %arg5, %10 : !tt.ptr<f8E5M2>, i64
    %20 = arith.extsi %arg17 : i32 to i64
    // CHECK: [[PTR3:%.*]] = tt.make_tensor_ptr {{.*}} : <tensor<8x64xf8E5M2, [[BLOCKED_LAYOUT1]]>
    %21 = tt.make_tensor_ptr %19, [%13, %c64_i64], [%20, %c1_i64], [%12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x64xf8E5M2, #blocked1>>
    %22 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #blocked>
    %23 = tt.splat %12 : i32 -> tensor<8xi32, #blocked>
    %24 = arith.addi %23, %22 : tensor<8xi32, #blocked>
    // CHECK: [[LOAD1:%.*]] = tt.load [[PTR1]] : !tt.ptr<tensor<8x64xf8E5M2, [[BLOCKED_LAYOUT1]]>
    // CHECK-NEXT: triton_gpu.convert_layout [[LOAD1]] : tensor<8x64xf8E5M2, [[BLOCKED_LAYOUT1]]> -> tensor<8x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %25 = tt.load %15 : !tt.ptr<tensor<8x64xf8E5M2, #dot1>>
    %26 = arith.addi %0, %c1_i32 : i32
    %27 = arith.muli %26, %c8_i32 : i32
    // CHECK: [[ADVANCE1:%.*]] = tt.advance [[PTR2]], {{.*}} : <tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]>>
    %28 = tt.advance %18, [%c0_i32, %12] : <tensor<64x16xf8E5M2, #dot2>>
    // CHECK: [[RES:%.*:2]] = scf.for {{.*}} iter_args(%arg22 = %cst_1, %arg23 = [[ADVANCE1]]) -> (tensor<8xf32, #blocked>, !tt.ptr<tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]>>)
    %29:2 = scf.for %arg21 = %12 to %27 step %c16_i32 iter_args(%arg22 = %cst_1, %arg23 = %28) -> (tensor<8xf32, #blocked>, !tt.ptr<tensor<64x16xf8E5M2, #dot2>>)  : i32 {
      // CHECK: [[LOAD2:%.*]] = tt.load %arg23 : !tt.ptr<tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]>>
      // CHECK-NEXT: triton_gpu.convert_layout [[LOAD2]] : tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]> -> tensor<64x16xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %36 = tt.load %arg23 : !tt.ptr<tensor<64x16xf8E5M2, #dot2>>
      %37 = tt.fp_to_fp %25 : tensor<8x64xf8E5M2, #dot1> -> tensor<8x64xf16, #dot1>
      %38 = tt.fp_to_fp %36 : tensor<64x16xf8E5M2, #dot2> -> tensor<64x16xf16, #dot2>
      %39 = tt.dot %37, %38, %cst, inputPrecision = tf32 : tensor<8x64xf16, #dot1> * tensor<64x16xf16, #dot2> -> tensor<8x16xf32, #dpas>
      %40 = triton_gpu.convert_layout %39 : tensor<8x16xf32, #dpas> -> tensor<8x16xf32, #blocked2>
      %41 = "tt.reduce"(%40) <{axis = 1 : i32}> ({
      ^bb0(%arg24: f32, %arg25: f32):
        %44 = arith.maxnumf %arg24, %arg25 : f32
        tt.reduce.return %44 : f32
      }) : (tensor<8x16xf32, #blocked2>) -> tensor<8xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %42 = triton_gpu.convert_layout %41 : tensor<8xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<8xf32, #blocked>
      // CHECK: [[ADVANCE2:%.*]] = tt.advance %arg23, {{.*}} : <tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]>>
      // CHECK-NEXT: scf.yield {{.*}}, [[ADVANCE2]] : tensor<8xf32, #blocked>, !tt.ptr<tensor<64x16xf8E5M2, [[BLOCKED_LAYOUT2]]>>
      %43 = tt.advance %arg23, [%c0_i32, %c16_i32] : <tensor<64x16xf8E5M2, #dot2>>
      scf.yield %42, %43 : tensor<8xf32, #blocked>, !tt.ptr<tensor<64x16xf8E5M2, #dot2>>
    } {tt.divisibility_arg1 = dense<16> : tensor<1xi32>}
    %30 = arith.addf %29#0, %cst_0 : tensor<8xf32, #blocked>
    %31 = arith.muli %1, %arg20 : i32
    %32 = tt.addptr %arg4, %31 : !tt.ptr<f32>, i32
    %33 = tt.splat %32 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>, #blocked>
    %34 = tt.addptr %33, %24 : tensor<8x!tt.ptr<f32>, #blocked>, tensor<8xi32, #blocked>
    tt.store %34, %30 : tensor<8x!tt.ptr<f32>, #blocked>
    %35 = tt.fp_to_fp %cst_2, rounding = rtne : tensor<8x64xf32, #blocked1> -> tensor<8x64xf8E5M2, #blocked1>
    tt.store %21, %35 : !tt.ptr<tensor<8x64xf8E5M2, #blocked1>>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#dot2 = #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: [[BLOCKED_LAYOUT:#.*]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK: @test_block_ptrs
  tt.func public @test_block_ptrs(%arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg19: i32) {
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #dpas}>>
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg19 : i32
    %3 = arith.remsi %1, %arg19 : i32
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.extsi %arg6 : i32 to i64
    %6 = arith.muli %4, %5 : i64
    %7 = arith.extsi %3 : i32 to i64
    %8 = arith.extsi %arg7 : i32 to i64
    %9 = arith.muli %7, %8 : i64
    %10 = arith.addi %6, %9 : i64
    %12 = arith.muli %0, %c64_i32 : i32
    %13 = arith.extsi %arg19 : i32 to i64
    %19 = tt.addptr %arg1, %10 : !tt.ptr<f8E5M2>, i64
    %20 = arith.extsi %arg11 : i32 to i64
    // CHECK: [[PTR1:%.*]] = tt.make_tensor_ptr {{.*}} : <tensor<64x32xf8E5M2, [[BLOCKED_LAYOUT]]>
    %21 = tt.make_tensor_ptr %19, [%c64_i64, %13], [%c1_i64, %20], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf8E5M2, #dot2>>
    // CHECK: [[RES:%.*]]:2 = scf.for {{.*}} iter_args(%arg6 = %cst, %arg7 = [[PTR1]]) -> (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x32xf8E5M2, [[BLOCKED_LAYOUT]]>>)
    %33:2 = scf.for %arg21 = %c0_i32 to %12 step %c32_i32 iter_args(%arg22 = %cst_1, %arg23 = %21) -> (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #dpas}>>, !tt.ptr<tensor<64x32xf8E5M2, #dot2>>)  : i32 {
      // CHECK: [[LOAD:%.*]] = tt.load %arg7 : !tt.ptr<tensor<64x32xf8E5M2, [[BLOCKED_LAYOUT]]>>
      // CHECK-NEXT: triton_gpu.convert_layout [[LOAD]] : tensor<64x32xf8E5M2, [[BLOCKED_LAYOUT]]> -> tensor<64x32xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      // CHECK-NEXT: scf.yield %arg6, %arg7 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x32xf8E5M2, #blocked>>
      %load = tt.load %arg23 : !tt.ptr<tensor<64x32xf8E5M2, #dot2>>
      scf.yield %arg22, %arg23 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #dpas}>>, !tt.ptr<tensor<64x32xf8E5M2, #dot2>>
    }
    // CHECK: scf.for {{.*}} iter_args(%arg6 = [[RES]]#0, %arg7 = [[RES]]#1) -> (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x32xf8E5M2, [[BLOCKED_LAYOUT]]>>)
    %34:2 = scf.for %arg21 = %c0_i32 to %12 step %c32_i32 iter_args(%arg22 = %33#0, %arg23 = %33#1) -> (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #dpas}>>, !tt.ptr<tensor<64x32xf8E5M2, #dot2>>) : i32 {
      // CHECK: scf.yield %arg6, %arg7 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<64x32xf8E5M2, [[BLOCKED_LAYOUT]]>>
      scf.yield %arg22, %arg23 : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #dpas}>>, !tt.ptr<tensor<64x32xf8E5M2, #dot2>>
    }
    tt.return
  }
}
