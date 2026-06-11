// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=ttig.2d_block_load_from_ptr --implicit-check-not=ttig.2d_block_load

// COM: Descriptor load without ttig.block_io attribute should NOT be converted
// COM: to ttig.2d_block_load — it is lowered to tt.load as a fallback.
// COM: Verify the generated IR builds pointer tensor, mask, and uses tt.load.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_no_block_io
  tt.func @descriptor_load_no_block_io(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: ttig.extract_desc
    // CHECK: tt.splat
    // CHECK: tt.make_range
    // CHECK: tt.addptr
    // CHECK: arith.cmpi
    // CHECK: tt.load
    // CHECK-NOT: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Column-major descriptor load that cannot find MakeTensorDescOp (passed
// COM: as a function argument) should be lowered to tt.load with inner-2
// COM: dimensions permuted. The descriptor has shape [32, 64] (N=32, K=64) but
// COM: the result is [64, 32] (K, N) due to the column_major transpose.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @descriptor_load_column_major_fallback
  tt.func @descriptor_load_column_major_fallback(%desc: !tt.tensordesc<32x64xf16>) -> tensor<64x32xf16, #dot0> {
    %c0_i32 = arith.constant 0 : i32
    // column_major means result [64, 32] = [K, N] (inner dims swapped).
    // The fallback must permute shapes/strides/indices to match result order.
    // CHECK: ttig.extract_desc
    // CHECK: tt.make_range
    // CHECK: tt.load
    // CHECK-NOT: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<32x64xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Module without support_2d_block_io attribute should skip the pass entirely.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_2d_block_support
  tt.func @no_2d_block_support(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) -> tensor<64x32xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK: tt.descriptor_load
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Pointer-based load from a function arg has unknown stride, so it
// COM: cannot be converted (stride analysis returns -1). Stays as tt.load.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @ptr_load_unknown_stride
  tt.func @ptr_load_unknown_stride(%arg0: tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0> {
    // CHECK: tt.load
    %0 = tt.load %arg0 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Load without block_io attribute should NOT be converted.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @no_block_io_attr
  tt.func @no_block_io_attr(%arg0: tensor<64x32x!tt.ptr<f16>, #dot0>) -> tensor<64x32xf16, #dot0> {
    // CHECK: tt.load
    %0 = tt.load %arg0 : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %0 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Pointer load with invalid pitch (< 64 bytes). The stride of 16 f16
// COM: elements gives pitch = 32 bytes, which is below the HW minimum of 64.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @ptr_load_invalid_pitch
  tt.func @ptr_load_invalid_pitch(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: tt.load
    %5 = tt.load %4 {ttig.block_io = "row_major", ttig.block_io_stride = 16 : i64} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Masked pointer load where the mask has non-trivial axis info that
// COM: constrains the tile size below what is valid for 2D block I/O.
// COM: The mask is derived from a comparison with a runtime bound, giving
// COM: constancy=1 in the fast-change dimension, which makes the tile invalid.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_load_non_power_of_2_constancy
  tt.func @masked_load_non_power_of_2_constancy(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // Mask with non-trivial constancy: compare range against runtime bound.
    // The mask has constancy=1 along the fast-change dim because each element
    // is independently compared against %arg2.
    %5 = tt.splat %arg2 : i32 -> tensor<1x32xi32, #dot0>
    %6 = arith.cmpi slt, %1, %5 : tensor<1x32xi32, #dot0>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
    %9 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #dot0>
    %10 = arith.cmpi slt, %8, %9 : tensor<64x1xi32, #dot0>
    %11 = tt.broadcast %6 : tensor<1x32xi1, #dot0> -> tensor<64x32xi1, #dot0>
    %12 = tt.broadcast %10 : tensor<64x1xi1, #dot0> -> tensor<64x32xi1, #dot0>
    %mask = arith.andi %11, %12 : tensor<64x32xi1, #dot0>
    // CHECK: tt.load
    %13 = tt.load %4, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %13 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Broadcast load (stride=0) where the tile width is incompatible with
// COM: threadsPerWarp for the row replication logic. tileWidth=1 with
// COM: threadsPerWarp=32 does not satisfy tileWidth*2 == threadsPerWarp.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @broadcast_load_incompatible_tile_width
  tt.func @broadcast_load_incompatible_tile_width(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32) {
    %cst = arith.constant dense<4096> : tensor<1x1024xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<1024xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<1x1024xi32, #blocked>
    %4 = tt.splat %0 : i32 -> tensor<1x1024xi32, #blocked>
    %5 = arith.addi %4, %3 : tensor<1x1024xi32, #blocked>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x1024x!tt.ptr<f32>, #blocked>
    %7 = tt.addptr %6, %5 : tensor<1x1024x!tt.ptr<f32>, #blocked>, tensor<1x1024xi32, #blocked>
    // CHECK: tt.load
    %8 = tt.load %7 {ttig.block_io = "column_major"} : tensor<1x1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: Regression test for issue #7022 (T5 training warmup failure).
// COM: Inductor's hf_T5 / hf_T5_base softmax-backward kernel produces a
// COM: column-major tensor-of-pointers load whose column stride is
// COM: 12_582_912 bf16 elements -> 25_165_824 byte pitch -> 0x1800000 (25
// COM: bits), exceeding the 24-bit limit on the 2D block IO message
// COM: descriptor base_pitch field (verifier in TritonGENOps.cpp). The pass
// COM: must refuse to convert this load; otherwise the downstream
// COM: triton_gen.2Dblockload verifier rejects the lowering and PassManager
// COM: fails in `make_llir`.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @col_pitch_exceeds_24bit_limit
  tt.func @col_pitch_exceeds_24bit_limit(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
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
    // CHECK: tt.load
    tt.load %ptrs {ttig.block_io = "column_major"} : tensor<8x8x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
