// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7625
// COM: There is an inefficient issue in the RemoveLayout optimization for the loop.
// COM: The structured loop return values and loop iteration arguments are duplicated during backward rematerialization,
// COM: which causes the loop body to be duplicated in the final result.a

// CHECK: #[[$ATTR_0:.+]] = #ttg.linear<{register = {{\[\[0, 1\], \[0, 2\], \[0, 4\], \[0, 8\]\],}} lane = {{\[\[1, 0\], \[2, 0\], \[4, 0\], \[8, 0\]\],}} warp = {{\[\[0, 16\], \[0, 32\]\],}}
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-LABEL: chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  tt.func public @chunk_gated_delta_rule_fwd_kernel_h_blockdim64(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.tensordesc<64x64xbf16>, %arg3: !tt.tensordesc<64x16xbf16>, %arg4: !tt.tensordesc<64x64xbf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #mma>
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    // CHECK:  scf.for %[[VAL_0:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args(%[[VAL_1:.*]] = {{.*}}) -> (tensor<16x64xf32, {{.*}}>)  : i32 {
    %0 = scf.for %arg5 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg6 = %cst) -> (tensor<16x64xf32, #blocked>)  : i32 {
      %2 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <16x64xbf16>
      %3 = arith.truncf %arg6 : tensor<16x64xf32, #blocked> to tensor<16x64xbf16, #blocked>
      tt.descriptor_store %2[%c0_i32, %c0_i32], %3 {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<16x64xbf16>, tensor<16x64xbf16, #blocked>
      %4 = arith.muli %arg5, %c64_i32 : i32
      // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<16x64xbf16>, tensor<16x64xbf16, #[[$ATTR_0]]>
      %5 = tt.descriptor_load %arg2[%4, %c0_i32] {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #blocked1>
      %6 = tt.trans %arg6 {order = array<i32: 1, 0>} : tensor<16x64xf32, #blocked> -> tensor<64x16xf32, #blocked2>
      %7 = arith.truncf %6 : tensor<64x16xf32, #blocked2> to tensor<64x16xbf16, #blocked2>
      %8 = ttg.convert_layout %5 : tensor<64x64xbf16, #blocked1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %9 = ttg.convert_layout %7 : tensor<64x16xbf16, #blocked2> -> tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %10 = tt.dot %8, %9, %cst_0, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x16xf32, #mma>
      %11 = ttg.convert_layout %10 : tensor<64x16xf32, #mma> -> tensor<64x16xf32, #blocked3>
      %12 = arith.truncf %11 : tensor<64x16xf32, #blocked3> to tensor<64x16xbf16, #blocked3>
      %13 = tt.descriptor_load %arg4[%4, %c0_i32] {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #blocked1>
      %14 = tt.trans %13 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #blocked1> -> tensor<64x64xbf16, #blocked4>
      %15 = ttg.convert_layout %14 : tensor<64x64xbf16, #blocked4> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %16 = ttg.convert_layout %12 : tensor<64x16xbf16, #blocked3> -> tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %17 = tt.dot %15, %16, %cst_0, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x16xf32, #mma>
      %18 = ttg.convert_layout %17 : tensor<64x16xf32, #mma> -> tensor<64x16xf32, #blocked2>
      %19 = tt.trans %18 {order = array<i32: 1, 0>} : tensor<64x16xf32, #blocked2> -> tensor<16x64xf32, #blocked>
      scf.yield %19 : tensor<16x64xf32, #blocked>
    }
    %1 = tt.make_tensor_descriptor %arg1, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <16x64xf32>
    // CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<16x64xf32>, tensor<16x64xf32, #[[$ATTR_0]]>
    tt.descriptor_store %1[%c0_i32, %c0_i32], %0 {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<16x64xf32>, tensor<16x64xf32, #blocked>
    tt.return
  }
}
