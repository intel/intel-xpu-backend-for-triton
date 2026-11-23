// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_WIDE_BLOCK_STORE=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefix=OFF

// A 16-bit SG16 store with a second 16-column block folds it into a 32-wide
// (64 B) row. Each message's payload is row-major [r0 blk0, r0 blk1, r1 blk0,
// ...]: registers 0, 8, 1, 9 for these bases [r1, r2, r4, c16, r8, r16].

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_tdesc
  // CHECK:         %[[E0:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(f16
  // CHECK:         %[[E1:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(f16
  // CHECK:         %[[E8:.*]] = llvm.extractvalue %{{.*}}[8] : !llvm.struct<(f16
  // CHECK:         llvm.insertelement %[[E0]], %{{.*}} : vector<16xf16>
  // CHECK-NOT:     llvm.insertelement
  // CHECK:         llvm.insertelement %[[E8]], %{{.*}} : vector<16xf16>
  // CHECK-NOT:     llvm.insertelement
  // CHECK:         llvm.insertelement %[[E1]], %{{.*}} : vector<16xf16>
  // CHECK:         %[[P:.*]] = llvm.bitcast %{{.*}} : vector<16xf16> to vector<16xi16>
  // CHECK:         triton_gen.sub_group_bitcast_shuffle
  // Row 1: payload elements 2 and 3 go through the shuffle and back in place.
  // CHECK:         %[[I2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    %[[X2:.*]] = llvm.extractelement %[[P]][%[[I2]] : i32] : vector<16xi16>
  // CHECK-NEXT:    %[[J0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT:    %[[A0:.*]] = llvm.insertelement %[[X2]], %{{.*}}[%[[J0]] : i32] : vector<2xi16>
  // CHECK-NEXT:    %[[I3:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT:    %[[X3:.*]] = llvm.extractelement %[[P]][%[[I3]] : i32] : vector<16xi16>
  // CHECK-NEXT:    %[[J1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    %[[A1:.*]] = llvm.insertelement %[[X3]], %[[A0]][%[[J1]] : i32] : vector<2xi16>
  // CHECK-NEXT:    %[[S:.*]] = triton_gen.sub_group_bitcast_shuffle %[[A1]] : vector<2xi16> -> vector<1xi32>
  // CHECK-NEXT:    %[[U:.*]] = llvm.bitcast %[[S]] : vector<1xi32> to vector<2xi16>
  // CHECK-NEXT:    %[[K0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NEXT:    %[[Y0:.*]] = llvm.extractelement %[[U]][%[[K0]] : i32] : vector<2xi16>
  // CHECK-NEXT:    %[[L2:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    %[[R2:.*]] = llvm.insertelement %[[Y0]], %{{.*}}[%[[L2]] : i32] : vector<16xi16>
  // CHECK-NEXT:    %[[K1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    %[[Y1:.*]] = llvm.extractelement %[[U]][%[[K1]] : i32] : vector<2xi16>
  // CHECK-NEXT:    %[[L3:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT:    llvm.insertelement %[[Y1]], %[[R2]][%[[L3]] : i32] : vector<16xi16>
  // CHECK-COUNT-6: triton_gen.sub_group_bitcast_shuffle %{{.*}} : vector<2xi16> -> vector<1xi32>
  // The message stores the last insert of the repack.
  // CHECK:         %[[I15:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK-NEXT:    %[[REPACKED:.*]] = llvm.insertelement %{{.*}}, %{{.*}}[%[[I15]] : i32] : vector<16xi16>
  // CHECK:         triton_gen.2Dblockstore {{.*}}, %[[REPACKED]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<16xi16>)
  // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}} tile_width = 32
  // CHECK-NOT:     triton_gen.2Dblockstore
  // OFF-LABEL:     llvm.func spir_kernelcc @fold_tdesc
  // OFF-NOT:       triton_gen.sub_group_bitcast_shuffle
  // OFF-COUNT-8:   triton_gen.2Dblockstore {{.*}} tile_width = 16
  tt.func public @fold_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %val: tensor<32x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----

// Pointer store: bases [r1, r2, r4, c16, r8] fold into 2 messages per lane.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 2]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_ptr
  // CHECK:         triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-2: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}
  // OFF-LABEL:     llvm.func spir_kernelcc @fold_ptr
  // OFF-NOT:       triton_gen.sub_group_bitcast_shuffle
  // OFF-COUNT-4:   triton_gen.2Dblockstore {{.*}} tile_width = 16
  tt.func public @fold_ptr(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %val: tensor<64x64xf16, #dpas>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dpas}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dpas}>> -> tensor<64x1xi32, #dpas>
    %2 = arith.constant dense<64> : tensor<64x1xi32, #dpas>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #dpas>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dpas}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dpas}>> -> tensor<1x64xi32, #dpas>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #dpas> -> tensor<64x64xi32, #dpas>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dpas> -> tensor<64x64xi32, #dpas>
    %8 = arith.addi %6, %7 : tensor<64x64xi32, #dpas>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #dpas>
    %ptrs = tt.addptr %9, %8 : tensor<64x64x!tt.ptr<f16>, #dpas>, tensor<64x64xi32, #dpas>
    tt.store %ptrs, %val {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<f16>, #dpas>
    tt.return
  }
}

// -----

// Four column blocks (bases add c32): only the first folds, so 8 messages.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_four_blocks_tdesc
  // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} tile_width = 32, tile_height = 8
  // CHECK-NOT:     triton_gen.2Dblockstore
  tt.func public @fold_four_blocks_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %val: tensor<32x64xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x64xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x64xf16, #dpas>, tensor<32x64xf16, #dpas>
    tt.return
  }
}

// -----

// Pairs of f16 packed into d32 already fill a 64 B row, so the store keeps its
// 16-wide d32 tiles.

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_packed_tdesc
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-2: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8
  tt.func public @no_fold_packed_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %val: tensor<8x64xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <8x64xf16, #blocked>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<8x64xf16, #blocked>, tensor<8x64xf16, #blocked>
    tt.return
  }
}

// -----

// SG32 DPAS C: lanes span 16 columns and 2 rows, so the rows stay 16 wide.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [1, 1], repCluster = [2, 2]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_sg32_dpas_tdesc
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} tile_width = 16, tile_height = 8
  tt.func public @no_fold_sg32_dpas_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %val: tensor<16x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <16x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<16x32xf16, #dpas>, tensor<16x32xf16, #dpas>
    tt.return
  }
}

// -----

// A column mask constant over 16 columns only keeps the 16-column block out of
// the v-blocks, so a 32-wide message would span a mask change.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [2, 2]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_mask16_ptr
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} tile_width = 16, tile_height = 8
  tt.func public @no_fold_mask16_ptr(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %val: tensor<64x64xf16, #dpas>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dpas}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dpas}>> -> tensor<64x1xi32, #dpas>
    %2 = arith.constant dense<64> : tensor<64x1xi32, #dpas>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #dpas>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dpas}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dpas}>> -> tensor<1x64xi32, #dpas>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #dpas> -> tensor<64x64xi32, #dpas>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dpas> -> tensor<64x64xi32, #dpas>
    %8 = arith.addi %6, %7 : tensor<64x64xi32, #dpas>
    %9 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #dpas>
    %ptrs = tt.addptr %9, %8 : tensor<64x64x!tt.ptr<f16>, #dpas>, tensor<64x64xi32, #dpas>
    %c16 = arith.constant dense<16> : tensor<1x64xi32, #dpas>
    %c3 = arith.constant dense<3> : tensor<1x64xi32, #dpas>
    %grp = arith.divui %5, %c16 : tensor<1x64xi32, #dpas>
    %m = arith.cmpi slt, %grp, %c3 : tensor<1x64xi32, #dpas>
    %mask = tt.broadcast %m : tensor<1x64xi1, #dpas> -> tensor<64x64xi1, #dpas>
    tt.store %ptrs, %val, %mask {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<f16>, #dpas>
    tt.return
  }
}

// -----

// A column index known divisible by 16 only would let a 32-wide row straddle
// column 0 after base-alignment compensation, so the store keeps 16-wide rows.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_straddle_tdesc
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} tile_width = 16, tile_height = 8
  tt.func public @no_fold_straddle_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %col: i32 {tt.divisibility = 16 : i32}, %val: tensor<32x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %col], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }

  // A column index known divisible by 32 cannot straddle column 0, so it folds.
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_aligned_col_tdesc
  // CHECK:         triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} tile_width = 32, tile_height = 8
  tt.func public @fold_aligned_col_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %col: i32 {tt.divisibility = 32 : i32}, %val: tensor<32x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %col], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----

// The LTS driver lacks the bitcast-shuffle intrinsic, so LTS keeps 16-wide rows.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.is_lts} {
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_lts_tdesc
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} tile_width = 16, tile_height = 8
  tt.func public @no_fold_lts_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %val: tensor<32x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----

// Math between the stored value and its dot, load, constant or argument keeps
// 16-wide rows: IGC vectorizes epilogue math from the narrow payload only.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // SCF reaches this pass as CF, IV first: the walk follows the accumulator's
  // branch operands to the zero init and the dot, so the loop result folds.
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_loop_dot_tdesc
  // CHECK:       ^bb1(%{{[0-9]+}}: i32, %[[ACC:[0-9]+]]: !llvm.struct<(f32
  // CHECK:         llvm.br ^bb1(
  // CHECK-NEXT:  ^bb3:
  // CHECK-NEXT:    llvm.extractvalue %[[ACC]][0]
  // CHECK:         triton_gen.sub_group_bitcast_shuffle
  // CHECK:         triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}
  // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}} tile_width = 32
  // CHECK-NOT:     triton_gen.2Dblockstore
  tt.func public @fold_loop_dot_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %a: tensor<32x16xf16, #dot0>, %b: tensor<16x32xf16, #dot1>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    cf.br ^loop(%c0_i32, %zero : i32, tensor<32x32xf32, #dpas>)
  ^loop(%i: i32, %acc: tensor<32x32xf32, #dpas>):
    %more = arith.cmpi slt, %i, %arg2 : i32
    cf.cond_br %more, ^body, ^exit
  ^body:
    %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<32x16xf16, #dot0> * tensor<16x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    %next = arith.addi %i, %c1_i32 : i32
    cf.br ^loop(%next, %d : i32, tensor<32x32xf32, #dpas>)
  ^exit:
    %t = arith.truncf %acc : tensor<32x32xf32, #dpas> to tensor<32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %t {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }

  // A loop-carried sum is math too; only the walk through the branches finds it.
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_loop_add_tdesc
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} tile_width = 16, tile_height = 8
  tt.func public @no_fold_loop_add_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %a: tensor<32x16xf16, #dot0>, %b: tensor<16x32xf16, #dot1>, %more: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    cf.br ^loop(%zero : tensor<32x32xf32, #dpas>)
  ^loop(%acc: tensor<32x32xf32, #dpas>):
    cf.cond_br %more, ^body, ^exit
  ^body:
    %d = tt.dot %a, %b, %zero, inputPrecision = tf32 : tensor<32x16xf16, #dot0> * tensor<16x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    %s = arith.addf %acc, %d : tensor<32x32xf32, #dpas>
    cf.br ^loop(%s : tensor<32x32xf32, #dpas>)
  ^exit:
    %t = arith.truncf %acc : tensor<32x32xf32, #dpas> to tensor<32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %t {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }

  // tt.store checks its value too; ttig.block_io_stride stands in for the pitch.
  // CHECK-LABEL: llvm.func spir_kernelcc @no_fold_mul_ptr
  // CHECK-NOT:     triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} tile_width = 16, tile_height = 8
  tt.func public @no_fold_mul_ptr(%ptrs: tensor<32x32x!tt.ptr<f16>, #dpas>, %x: tensor<32x32xf16, #dpas>) {
    %v = arith.mulf %x, %x : tensor<32x32xf16, #dpas>
    tt.store %ptrs, %v {ttig.block_io = "row_major", ttig.block_io_stride = 32 : i64} : tensor<32x32x!tt.ptr<f16>, #dpas>
    tt.return
  }

  // tt.fp_to_fp converts like arith.truncf, so a converted dot folds.
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_fp_to_fp_dot_tdesc
  // CHECK:         triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} tile_width = 32, tile_height = 8
  tt.func public @fold_fp_to_fp_dot_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %a: tensor<32x16xf16, #dot0>, %b: tensor<16x32xf16, #dot1>, %c: tensor<32x32xf32, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    %d = tt.dot %a, %b, %c, inputPrecision = tf32 : tensor<32x16xf16, #dot0> * tensor<16x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    %t = tt.fp_to_fp %d, rounding = rtne : tensor<32x32xf32, #dpas> -> tensor<32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %t {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }

  // A loaded tile stored back unchanged folds.
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_load_tdesc
  // CHECK:         triton_gen.sub_group_bitcast_shuffle
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} tile_width = 32, tile_height = 8
  tt.func public @fold_load_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    %v = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<32x32xf16, #dpas> -> tensor<32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %v {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }
}

// -----

// With ttig.store_l1_write_back, a widened 64 B row is a whole line and keeps
// the default policy; the kill switch's 32 B rows are written back through L1.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_l1wb_tdesc
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}
  // CHECK-NOT:     triton_gen.2Dblockstore
  // OFF-LABEL:     llvm.func spir_kernelcc @fold_l1wb_tdesc
  // OFF-COUNT-8:   triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = L1WB_L3WB}
  // OFF-NOT:       triton_gen.2Dblockstore
  tt.func public @fold_l1wb_tdesc(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %val: tensor<32x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %val {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }

  // tt.store picks the same policy; ttig.block_io_stride stands in for the pitch.
  // CHECK-LABEL: llvm.func spir_kernelcc @fold_l1wb_ptr
  // CHECK:         triton_gen.2Dblockstore {{.*}} tile_width = 32, {{.*}} cache_control = Default
  // OFF-LABEL:     llvm.func spir_kernelcc @fold_l1wb_ptr
  // OFF:           triton_gen.2Dblockstore {{.*}} tile_width = 16, {{.*}} cache_control = L1WB_L3WB
  tt.func public @fold_l1wb_ptr(%ptrs: tensor<32x32x!tt.ptr<f16>, #dpas>, %val: tensor<32x32xf16, #dpas>) {
    tt.store %ptrs, %val {ttig.block_io = "row_major", ttig.block_io_stride = 32 : i64} : tensor<32x32x!tt.ptr<f16>, #dpas>
    tt.return
  }
}
