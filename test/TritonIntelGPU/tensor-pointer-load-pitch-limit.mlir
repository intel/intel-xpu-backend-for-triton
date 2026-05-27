// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: Regression test for issue #7022 (T5 training warmup failure).
// COM:
// COM: Inductor's hf_T5 / hf_T5_base softmax-backward kernel produces a
// COM: column-major tensor-of-pointers load whose column stride is 12_582_912
// COM: bf16 elements -> 25_165_824 byte pitch -> 0x1800000 (25 bits), exceeding
// COM: the 24-bit limit on the `triton_gen.2Dblockload` `base_pitch` operand
// COM: (verifier in TritonGENOps.cpp). Lowering must refuse to emit a 2D block
// COM: load in this case and fall back to the scalar/vectorized path; otherwise
// COM: PassManager::run fails in `make_llir`.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @col_pitch_exceeds_24bit_limit
  tt.func public @col_pitch_exceeds_24bit_limit(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %row = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row_exp = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %col = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col_exp = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %col_stride = arith.constant dense<12582912> : tensor<1x8xi32, #blocked>
    %col_off = arith.muli %col_exp, %col_stride : tensor<1x8xi32, #blocked>
    %row_bc = tt.broadcast %row_exp : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %col_bc = tt.broadcast %col_off : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %offsets = arith.addi %row_bc, %col_bc : tensor<8x8xi32, #blocked>
    %ptrs_splat = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x8x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptrs_splat, %offsets : tensor<8x8x!tt.ptr<bf16>, #blocked>, tensor<8x8xi32, #blocked>
    // CHECK-NOT: triton_gen.2Dblockload
    tt.load %ptrs {ttig.block_io = "column_major"} : tensor<8x8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// COM: Boundary case: column stride 8_388_608 bf16 elements -> 16_777_216 byte
// COM: pitch == (1 << 24) -- still within hardware limit, must lower to a
// COM: triton_gen.2Dblockload as before.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @col_pitch_at_24bit_limit
  tt.func public @col_pitch_at_24bit_limit(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %row = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row_exp = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %col = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col_exp = tt.expand_dims %col {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %col_stride = arith.constant dense<8388608> : tensor<1x8xi32, #blocked>
    %col_off = arith.muli %col_exp, %col_stride : tensor<1x8xi32, #blocked>
    %row_bc = tt.broadcast %row_exp : tensor<8x1xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %col_bc = tt.broadcast %col_off : tensor<1x8xi32, #blocked> -> tensor<8x8xi32, #blocked>
    %offsets = arith.addi %row_bc, %col_bc : tensor<8x8xi32, #blocked>
    %ptrs_splat = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x8x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptrs_splat, %offsets : tensor<8x8x!tt.ptr<bf16>, #blocked>, tensor<8x8xi32, #blocked>
    // CHECK: triton_gen.2Dblockload
    tt.load %ptrs {ttig.block_io = "column_major"} : tensor<8x8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
