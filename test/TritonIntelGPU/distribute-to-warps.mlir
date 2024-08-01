// RUN: triton-opt %s -split-input-file -tritonintelgpu-distribute-to-warps | FileCheck %s

#blocked1 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blockedC = #triton_gpu.blocked<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blockedA = #triton_gpu.dot_op<{opIdx = 0, parent = #blockedC}>
#blockedB = #triton_gpu.dot_op<{opIdx = 1, parent = #blockedC}>

// CHECK-DAG: [[WARP1:#.*]] = #triton_intel_gpu.warp<{sizePerThread = [64, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
// CHECK-DAG: [[WARP2:#.*]] = #triton_intel_gpu.warp<{sizePerThread = [32, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
// COM: In the loop body:
// COM:   - thread block works on: 128x128xf32 = 128x32xf16 * 32x128xf16
// COM:   - each warp works on:    64x64xf32   =  64x32xf16 * 32x64xf16
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_with_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    // CHECK: @matmul_kernel_with_block_pointers_with_convertlayout
    // CHECK:      [[SUB_GROUP_ID:%.*]] = gpu.subgroup_id : index
    // CHECK:      [[WARP_ID:%.*]] = arith.index_cast [[SUB_GROUP_ID]] : index to i32
    // CHECK:      [[CST:%.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32, [[WARP1]]>
    // CHECK-DAG:  [[ARG3_EXT:%.*]] = arith.extsi %arg3 : i32 to i64
    // CHECK-DAG:  [[ARG5_EXT:%.*]] = arith.extsi %arg5 : i32 to i64
    // CHECK-DAG:  [[ARG6_EXT:%.*]] = arith.extsi %arg6 : i32 to i64
    // CHECK:      [[CST32:%.*]] = arith.constant 32 : i32
    // CHECK:      [[MUL:%.*]] = arith.muli [[WARP_ID]], [[CST32]] : i32
    // CHECK-NEXT: [[ADD:%.*]] = arith.addi [[MUL]], {{.*}} : i32
    // CHECK:      [[TPTR1:%.*]] = tt.make_tensor_ptr %arg0, [[[ARG3_EXT]], [[ARG5_EXT]]], [[[ARG6_EXT]], {{.*}}], [[[ADD]], {{.*}}]
    // CHECK-SAME:      {order = array<i32: 1, 0>} : <tensor<32x32xf16, [[WARP2]]>>
    // CHECK-DAG:  [[ARG4_EXT:%.*]] = arith.extsi %arg4 : i32 to i64
    // CHECK-DAG:  [[ARG7_EXT:%.*]] = arith.extsi %arg7 : i32 to i64
    // CHECK:      [[TPTR2:%.*]] = tt.make_tensor_ptr %arg1, [[[ARG5_EXT]], [[ARG4_EXT]]], [[[ARG7_EXT]], {{.*}}], [{{.*}}, {{.*}}]
    // CHECK-SAME:      {order = array<i32: 1, 0>} : <tensor<32x32xf16, [[WARP2]]>>
    // CHECK:      scf.for {{.*}} iter_args([[ARG10:%.*]] = [[CST]], [[ARG11:%.*]] = [[TPTR1]], [[ARG12:%.*]] = [[TPTR2]])
    // CHECK-DAG:    [[LOAD1:%.*]] = tt.load [[ARG11]] : !tt.ptr<tensor<32x32xf16, [[WARP2]]>>
    // CHECK-DAG:    [[LOAD2:%.*]] = tt.load [[ARG12]] : !tt.ptr<tensor<32x32xf16, [[WARP2]]>>
    // CHECK:        [[ALLOC1:%.*]] = triton_intel_gpu.alloc : <f16, 3>
    // CHECK:        [[PTR1:%.*]] = tt.make_tensor_ptr [[ALLOC1]], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x32xf16, [[WARP2]]>, 3>
    // CHECK:        tt.store [[PTR1]], [[LOAD1]] : !tt.ptr<tensor<32x32xf16, [[WARP2]]>, 3>
    // CHECK:        gpu.barrier
    // CHECK:        [[PTR2:%.*]] = tt.make_tensor_ptr [[ALLOC1]], {{.*}} {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>>, 3>
    // CHECK:        [[LOAD3:%.*]] = tt.load [[PTR2]] : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>>, 3>
    // CHECK:        [[ALLOC2:%.*]] = triton_intel_gpu.alloc : <f16, 3>
    // CHECK:        [[PTR3:%.*]] = tt.make_tensor_ptr [[ALLOC2]], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x32xf16, [[WARP2]]>, 3>
    // CHECK:        tt.store [[PTR3]], [[LOAD2]] : !tt.ptr<tensor<32x32xf16, [[WARP2]]>, 3>
    // CHECK:        gpu.barrier
    // CHECK:        [[PTR3:%.*]] = tt.make_tensor_ptr [[ALLOC2]], {{.*}} {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>>, 3>
    // CHECK:        [[LOAD4:%.*]] = tt.load [[PTR3]] : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>>, 3>
    // CHECK:        tt.dot [[LOAD3]], [[LOAD4]], [[ARG10]], inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>> -> tensor<64x64xf32, [[WARP1]]>
    // CHECK:        tt.advance [[ARG11]], [{{.*}}] : <tensor<32x32xf16, [[WARP2]]>>
    // CHECK:        tt.advance [[ARG12]], [{{.*}}] : <tensor<32x32xf16, [[WARP2]]>>
    // CHECK:      }
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
      %28 = tt.load %arg11 : !tt.ptr<tensor<128x32xf16, #blocked1>, 1>
      %29 = tt.load %arg12 : !tt.ptr<tensor<32x128xf16, #blocked2>, 1>
      %30 = triton_gpu.convert_layout %28 : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #blockedA>
      %31 = triton_gpu.convert_layout %29 : tensor<32x128xf16, #blocked2> -> tensor<32x128xf16, #blockedB>
      %32 = tt.dot %30, %31, %arg10 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #blockedA> * tensor<32x128xf16, #blockedB> -> tensor<128x128xf32, #blockedC>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<128x32xf16, #blocked1>, 1>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x128xf16, #blocked2>, 1>
      scf.yield %32, %33, %34 : tensor<128x128xf32, #blockedC>, !tt.ptr<tensor<128x32xf16, #blocked1>, 1>, !tt.ptr<tensor<32x128xf16, #blocked2>, 1>
    }
    tt.return
  }

  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    // CHECK: @matmul_kernel_with_block_pointers_without_convertlayout
    // CHECK:      [[SUB_GROUP_ID:%.*]] = gpu.subgroup_id : index
    // CHECK:      [[WARP_ID:%.*]] = arith.index_cast [[SUB_GROUP_ID]] : index to i32
    // CHECK:      [[CST:%.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32, [[WARP1]]>
    // CHECK:      [[CST32:%.*]] = arith.constant 32 : i32
    // CHECK-DAG:  [[ARG5_EXT:%.*]] = arith.extsi %arg5 : i32 to i64
    // CHECK-DAG:  [[ARG6_EXT:%.*]] = arith.extsi %arg6 : i32 to i64
    // CHECK:      [[CST2:%.*]] = arith.constant 2 : i32
    // CHECK:      [[DIV:%.*]] = arith.divsi [[WARP_ID]], [[CST2]] : i32
    // CHECK:      [[CST2_1:%.*]] = arith.constant 2 : i32
    // CHECK:      [[REM:%.*]] = arith.remsi [[DIV]], [[CST2_1]] : i32
    // CHECK:      [[CST64:%.*]] = arith.constant 64 : i32
    // CHECK:      [[MUL:%.*]] = arith.muli [[REM]], [[CST64]] : i32
    // CHECK-NEXT: [[ADD:%.*]] = arith.addi [[MUL]], {{.*}} : i32
    // CHECK:      [[TPTR1:%.*]] = tt.make_tensor_ptr %arg0, [[[ARG3_EXT]], [[ARG5_EXT]]], [[[ARG6_EXT]], {{.*}}], [[[ADD]], {{.*}}]
    // CHECK-SAME:      {order = array<i32: 1, 0>} : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>>>
    // CHECK-DAG:  [[ARG4_EXT:%.*]] = arith.extsi %arg4 : i32 to i64
    // CHECK-DAG:  [[ARG7_EXT:%.*]] = arith.extsi %arg7 : i32 to i64
    // CHECK:      [[TPTR2:%.*]] = tt.make_tensor_ptr %arg1, [[[ARG5_EXT]], [[ARG4_EXT]]], [[[ARG7_EXT]], {{.*}}], [{{.*}}, {{.*}}]
    // CHECK-SAME:    {order = array<i32: 1, 0>} : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>>>
    // CHECK:      scf.for {{.*}} iter_args([[ARG10:%.*]] = [[CST]], [[ARG11:%.*]] = [[TPTR1]], [[ARG12:%.*]] = [[TPTR2]])
    // CHECK-DAG:    [[LOAD1:%.*]] = tt.load [[ARG11]] : !tt.ptr<tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>>>
    // CHECK-DAG:    [[LOAD2:%.*]] = tt.load [[ARG12]] : !tt.ptr<tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>>>
    // CHECK:        tt.dot [[LOAD1]], [[LOAD2]], [[ARG10]], inputPrecision = tf32 : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>> -> tensor<64x64xf32, [[WARP1]]>
    // CHECK:        tt.advance [[ARG11]], [{{.*}}] : <tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[WARP1]]}>>>
    // CHECK:        tt.advance [[ARG12]], [{{.*}}] : <tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[WARP1]]}>>>
    // CHECK:      }
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
      %28 = tt.load %arg11 : !tt.ptr<tensor<128x32xf16, #blockedA>, 1>
      %29 = tt.load %arg12 : !tt.ptr<tensor<32x128xf16, #blockedB>, 1>
      %32 = tt.dot %28, %29, %arg10 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #blockedA> * tensor<32x128xf16, #blockedB> -> tensor<128x128xf32, #blockedC>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<128x32xf16, #blockedA>, 1>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x128xf16, #blockedB>, 1>
      scf.yield %32, %33, %34 : tensor<128x128xf32, #blockedC>, !tt.ptr<tensor<128x32xf16, #blockedA>, 1>, !tt.ptr<tensor<32x128xf16, #blockedB>, 1>
    }
    tt.return
  }
}

// -----

// COM: test for flash-attention related ops
#blocked = #triton_gpu.blocked<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], warpsPerCTA = [8, 1], order = [1, 0]}>

// CHECK: [[WARP:#.*]] = #triton_intel_gpu.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: f32, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>) {
    // CHECK: @_attn_fwd
    // CHECK: tt.splat {{.*}} : f32 -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = [[WARP]]}>>
    // CHECK: tt.splat {{.*}} : f32 -> tensor<16x64xf32, [[WARP]]>
    // CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}> ({
    // CHECK: }) : (tensor<16x64xf32, [[WARP]]>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = [[WARP]]}>>
    // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = [[WARP]]}>> -> tensor<16x1xf32, [[WARP]]>
    // CHECK: tt.broadcast {{.*}} : tensor<16x1xf32, [[WARP]]> -> tensor<16x64xf32, [[WARP]]>
    %0 = tt.splat %arg3 : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.splat %arg3 : f32 -> tensor<128x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %6 = arith.maxnumf %arg6, %arg7 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = arith.mulf %2, %0 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %5 = tt.broadcast %4 : tensor<128x1xf32, #blocked> -> tensor<128x64xf32, #blocked>
    tt.return
  }
}
