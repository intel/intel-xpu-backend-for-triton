// RUN: triton-opt %s -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
// CHECK-LABEL: @triton_per_fused_exp_logsumexp_sub_view_3
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/5947
// COM: try to catch "Unexpected user" issue for TDESC
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.is_lts, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  tt.func public @triton_per_fused_exp_logsumexp_sub_view_3(%in_out_ptr0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}, %r0_numel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %tmp_desc = arith.constant 393216 : i64
    %tmp9 = arith.constant dense<0.000000e+00> : tensor<4x1xf32, #blocked>
    %tmp7 = arith.constant dense<0x7F800000> : tensor<4x1xf32, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %r0_offset = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %xoffset = tt.get_program_id x : i32
    %xoffset_0 = arith.muli %xoffset, %c4_i32 : i32
    %tmp_desc_1 = tt.make_tensor_ptr %in_out_ptr0, [%tmp_desc, %c128_i64], [%c128_i64, %c1_i64], [%r0_offset, %r0_offset] {order = array<i32: 1, 0>} : <tensor<4x128xf16, #blocked1>>
    %tmp0 = tt.advance %tmp_desc_1, [%xoffset_0, %r0_offset] : <tensor<4x128xf16, #blocked1>>
    %tmp0_2 = tt.load %tmp0 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<4x128xf16, #blocked1>>
    %tmp0_3 = ttg.convert_layout %tmp0_2 : tensor<4x128xf16, #blocked1> -> tensor<4x128xf16, #blocked2>
    %tmp0_4 = arith.extf %tmp0_3 : tensor<4x128xf16, #blocked2> to tensor<4x128xf32, #blocked2>
    %tmp4 = "tt.reduce"(%tmp0_4) <{axis = 1 : i32}> ({
    ^bb0(%tmp4_19: f32, %tmp4_20: f32):
      %mask = arith.cmpf ogt, %tmp4_19, %tmp4_20 : f32
      %mask_21 = arith.cmpf une, %tmp4_19, %tmp4_19 : f32
      %mask_22 = arith.ori %mask, %mask_21 : i1
      %tmp4_23 = arith.select %mask_22, %tmp4_19, %tmp4_20 : f32
      tt.reduce.return %tmp4_23 : f32
    }) : (tensor<4x128xf32, #blocked2>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %tmp4_5 = ttg.convert_layout %tmp4 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<4xf32, #blocked3>
    %tmp4_6 = ttg.convert_layout %tmp4_5 : tensor<4xf32, #blocked3> -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %tmp4_7 = tt.expand_dims %tmp4_6 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<4x1xf32, #blocked4>
    %tmp5 = ttg.convert_layout %tmp4_7 : tensor<4x1xf32, #blocked4> -> tensor<4x1xf32, #blocked>
    %tmp5_8 = math.absf %tmp5 : tensor<4x1xf32, #blocked>
    %tmp7_9 = arith.cmpf oeq, %tmp5_8, %tmp7 : tensor<4x1xf32, #blocked>
    %tmp9_10 = arith.select %tmp7_9, %tmp9, %tmp5 : tensor<4x1xi1, #blocked>, tensor<4x1xf32, #blocked>
    %tmp10 = tt.broadcast %tmp9_10 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
    %tmp10_11 = ttg.convert_layout %tmp10 : tensor<4x128xf32, #blocked> -> tensor<4x128xf32, #blocked2>
    %tmp10_12 = arith.subf %tmp0_4, %tmp10_11 : tensor<4x128xf32, #blocked2>
    %tmp11 = tt.extern_elementwise %tmp10_12 {libname = "", libpath = "", pure = true, symbol = "__imf_expf"} : (tensor<4x128xf32, #blocked2>) -> tensor<4x128xf32, #blocked2>
    %tmp14 = "tt.reduce"(%tmp11) <{axis = 1 : i32}> ({
    ^bb0(%tmp14_19: f32, %tmp14_20: f32):
      %tmp14_21 = arith.addf %tmp14_19, %tmp14_20 : f32
      tt.reduce.return %tmp14_21 : f32
    }) : (tensor<4x128xf32, #blocked2>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %tmp14_13 = ttg.convert_layout %tmp14 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<4xf32, #blocked3>
    %tmp14_14 = ttg.convert_layout %tmp14_13 : tensor<4xf32, #blocked3> -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %tmp14_15 = tt.expand_dims %tmp14_14 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<4x1xf32, #blocked4>
    %tmp15 = ttg.convert_layout %tmp14_15 : tensor<4x1xf32, #blocked4> -> tensor<4x1xf32, #blocked>
    %tmp15_16 = math.log %tmp15 : tensor<4x1xf32, #blocked>
    %tmp16 = arith.addf %tmp15_16, %tmp9_10 : tensor<4x1xf32, #blocked>
    %tmp18 = tt.broadcast %tmp16 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
    %tmp18_17 = ttg.convert_layout %tmp18 : tensor<4x128xf32, #blocked> -> tensor<4x128xf32, #blocked2>
    %tmp18_18 = arith.subf %tmp0_4, %tmp18_17 : tensor<4x128xf32, #blocked2>
    %tmp19 = tt.extern_elementwise %tmp18_18 {libname = "", libpath = "", pure = true, symbol = "__imf_expf"} : (tensor<4x128xf32, #blocked2>) -> tensor<4x128xf32, #blocked2>
    %0 = arith.truncf %tmp19 : tensor<4x128xf32, #blocked2> to tensor<4x128xf16, #blocked2>
    %1 = ttg.convert_layout %0 : tensor<4x128xf16, #blocked2> -> tensor<4x128xf16, #blocked1>
    tt.store %tmp0, %1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<4x128xf16, #blocked1>>
    tt.return
  }
}
