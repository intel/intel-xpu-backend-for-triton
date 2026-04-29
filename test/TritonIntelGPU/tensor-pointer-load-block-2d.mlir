// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// COM: Regression extracted from torchbench DCGAN on the LTS driver path:
// COM: tensor<1024x1xf32> with ttig.block_io = "row_major" and
// COM: threadsPerWarp = 32. Before the fix this hit an assertion in
// COM: LoadOpToBlockIOConversion; after the fix the pattern returns failure()
// COM: and the load is lowered via the generic scalar path.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.is_lts, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @triton_poi_fused_convolution_0
  tt.func public @triton_poi_fused_convolution_0(%in_ptr0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %ynumel: i32 {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) {
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<3> : tensor<1024x1xi32, #blocked>
    %cst_0 = arith.constant dense<4096> : tensor<1024x1xi32, #blocked>
    %cst_1 = arith.constant dense<12288> : tensor<1024x1xi32, #blocked>
    %c3_i32 = arith.constant 3 : i32
    %yoffset = tt.get_program_id y : i32
    %yoffset_2 = arith.muli %yoffset, %c1024_i32 : i32
    %yindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %yindex_3 = tt.expand_dims %yindex {axis = 1 : i32} : tensor<1024xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1024x1xi32, #blocked>
    %yindex_4 = tt.splat %yoffset_2 : i32 -> tensor<1024x1xi32, #blocked>
    %yindex_5 = arith.addi %yindex_4, %yindex_3 : tensor<1024x1xi32, #blocked>
    %xoffset = tt.get_program_id x : i32
    %y0 = arith.remui %yindex_5, %cst : tensor<1024x1xi32, #blocked>
    %y1 = arith.divui %yindex_5, %cst : tensor<1024x1xi32, #blocked>
    %tmp0 = arith.muli %yindex_5, %cst_0 : tensor<1024x1xi32, #blocked>
    %tmp0_6 = tt.splat %xoffset : i32 -> tensor<1024x1xi32, #blocked>
    %tmp0_7 = arith.addi %tmp0_6, %tmp0 : tensor<1024x1xi32, #blocked>
    %tmp0_8 = tt.splat %in_ptr0 : !tt.ptr<f32> -> tensor<1024x1x!tt.ptr<f32>, #blocked>
    %tmp0_9 = tt.addptr %tmp0_8, %tmp0_7 : tensor<1024x1x!tt.ptr<f32>, #blocked>, tensor<1024x1xi32, #blocked>
    %tmp0_10 = tt.load %tmp0_9 evictionPolicy = evict_last {ttig.block_io = "row_major"} : tensor<1024x1x!tt.ptr<f32>, #blocked>
    %0 = arith.muli %xoffset, %c3_i32 : i32
    %1 = tt.splat %0 : i32 -> tensor<1024x1xi32, #blocked>
    %2 = arith.addi %y0, %1 : tensor<1024x1xi32, #blocked>
    %3 = arith.muli %y1, %cst_1 : tensor<1024x1xi32, #blocked>
    %4 = arith.addi %2, %3 : tensor<1024x1xi32, #blocked>
    %5 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<1024x1x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<1024x1x!tt.ptr<f32>, #blocked>, tensor<1024x1xi32, #blocked>
    tt.store %6, %tmp0_10 : tensor<1024x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
