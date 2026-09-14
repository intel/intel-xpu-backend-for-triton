// RUN: triton-opt %s -tritonintelgpu-simd-reduce-locality | FileCheck %s

#linear = #ttg.linear<{register = [[8, 0], [0, 2], [0, 4], [0, 8]], lane = [[0, 1], [0, 16], [1, 0], [2, 0], [4, 0]], warp = [[16, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_arithmetic, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_rounded_divide_sqrt, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {

  // CHECK-LABEL: tt.func public @simd_reduce_locality_smoke
  // CHECK: %[[RED:.+]] = "tt.reduce"(%[[ARG:.+]]) <{axis = 1 : i32}> ({
  // CHECK: ^bb0(%[[LHS:.+]]: f32, %[[RHS:.+]]: f32):
  // CHECK:   %[[MASK0:.+]] = arith.cmpf ogt, %[[LHS]], %[[RHS]] : f32
  // CHECK:   %[[MASK1:.+]] = arith.cmpf une, %[[LHS]], %[[LHS]] : f32
  // CHECK:   %[[MASK2:.+]] = arith.ori %[[MASK0]], %[[MASK1]] : i1
  // CHECK:   %[[SEL:.+]] = arith.select %[[MASK2]], %[[LHS]], %[[RHS]] : f32
  // CHECK:   tt.reduce.return %[[SEL]] : f32
  tt.func public @simd_reduce_locality_smoke(%arg0: tensor<32x32xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>> attributes {noinline = false} {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%tmp6_45: f32, %tmp6_46: f32):
      %mask = arith.cmpf ogt, %tmp6_45, %tmp6_46 : f32
      %mask_47 = arith.cmpf une, %tmp6_45, %tmp6_45 : f32
      %mask_48 = arith.ori %mask, %mask_47 : i1
      %tmp6_49 = arith.select %mask_48, %tmp6_45, %tmp6_46 : f32
      tt.reduce.return %tmp6_49 : f32
    }) : (tensor<32x32xf32, #linear>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    tt.return %0 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  }
}
