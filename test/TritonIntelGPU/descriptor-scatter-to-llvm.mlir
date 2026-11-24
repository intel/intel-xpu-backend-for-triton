// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// ttig.descriptor_scatter stores each row run that getScatterBlockWriteRuns finds with one
// sub-group block write under a uniform warp, row and window predicate, and stores a run the
// window rejects per element. Other layouts and unaligned descriptors store per lane.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#x = #ttg.slice<{dim = 1, parent = #dpas}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#xb = #ttg.slice<{dim = 1, parent = #blocked}>
#strided = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#xs = #ttg.slice<{dim = 1, parent = #strided}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_predicated_io} {
  // fast_bf16: each row a thread holds is one run of 2 registers 16 columns apart (N = 8).
  // CHECK-LABEL: llvm.func spir_kernelcc @fast_bf16
  // CHECK-DAG:   %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC:.*]][0] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
  // CHECK-DAG:   %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
  // CHECK-DAG:   %[[STRIDE0:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
  // CHECK-DAG:   %[[BASE:.*]] = llvm.extractvalue %[[DESC]][4] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
  // CHECK-DAG:   %[[XOFF1:.*]] = llvm.extractvalue %arg1[1] :
  // CHECK-DAG:   %[[E0:.*]] = llvm.extractvalue %arg2[0] :
  // CHECK-DAG:   %[[E8:.*]] = llvm.extractvalue %arg2[8] :
  // CHECK-DAG:   %[[X:.*]] = llvm.call spir_funccc @_Z17sub_group_shuffleij(
  // CHECK-DAG:   %[[X64:.*]] = llvm.sext %[[X]] : i32 to i64
  // CHECK-DAG:   %[[SHAPE0_32:.*]] = llvm.trunc %[[SHAPE0]] : i64 to i32
  // CHECK-DAG:   %[[SHAPE0_64:.*]] = llvm.zext %[[SHAPE0_32]] : i32 to i64
  // CHECK-DAG:   %[[ROW_OK:.*]] = llvm.icmp "ult" %[[X64]], %[[SHAPE0_64]] : i64
  // CHECK-DAG:   %[[ROW_OFF:.*]] = llvm.mul %[[X64]], %[[STRIDE0]] : i64
  // CHECK-DAG:   %[[LANE:.*]] = llvm.urem %{{.*}}, %{{.*}} : i32
  // CHECK-DAG:   %[[LANE64:.*]] = llvm.zext %[[LANE]] : i32 to i64
  // CHECK-DAG:   %[[WIDTH:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-DAG:   %[[LIMIT:.*]] = llvm.sub %[[SHAPE1]], %[[WIDTH]] : i64
  // CHECK-DAG:   %[[LE:.*]] = llvm.icmp "sle" %[[C0:[^,]*]], %[[LIMIT]] : i64
  // CHECK-DAG:   %[[C0]] = llvm.add %[[Y64:[^,]*]], %{{.*}} : i64
  // CHECK-DAG:   %[[GE:.*]] = llvm.icmp "sge" %[[C0]], %{{.*}} : i64
  // CHECK-DAG:   %[[WIN:.*]] = llvm.and %[[GE]], %[[LE]] : i1
  // CHECK:       %[[GUARD:.*]] = llvm.and %[[ROW_OK]], %[[WIN]] : i1
  // CHECK:       llvm.cond_br %[[GUARD]], ^
  // CHECK:       %[[OFF:.*]] = llvm.add %[[ROW_OFF]], %[[C0]] : i64
  // CHECK:       %[[ADDR:.*]] = llvm.getelementptr %[[BASE]][%[[OFF]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, bf16
  // CHECK-DAG:   %[[W0:.*]] = llvm.bitcast %[[E0]] : bf16 to i16
  // CHECK-DAG:   %[[I0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:       %[[V0:.*]] = llvm.insertelement %[[W0]], %{{.*}}[%[[I0]] : i32] : vector<2xi16>
  // CHECK-DAG:   %[[W8:.*]] = llvm.bitcast %[[E8]] : bf16 to i16
  // CHECK-DAG:   %[[I1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:       %[[DATA:.*]] = llvm.insertelement %[[W8]], %[[V0]][%[[I1]] : i32] : vector<2xi16>
  // CHECK:       triton_gen.sub_group_block_write %[[ADDR]], %[[DATA]] : !llvm.ptr<1>, vector<2xi16>
  // CHECK:       %[[NOT_WIN:.*]] = llvm.xor %[[WIN]], %{{.*}} : i1
  // CHECK:       %[[TAIL:.*]] = llvm.and %[[ROW_OK]], %[[NOT_WIN]] : i1
  // CHECK:       llvm.cond_br %[[TAIL]], ^
  // CHECK:       %[[LANE_COL:.*]] = llvm.add %[[C0]], %[[LANE64]] : i64
  // CHECK:       %[[STEP:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK:       %[[COL1:.*]] = llvm.add %[[LANE_COL]], %[[STEP]] : i64
  // CHECK:       %[[GE1:.*]] = llvm.icmp "sge" %[[COL1]], %{{.*}} : i64
  // CHECK:       %[[LT1:.*]] = llvm.icmp "slt" %[[COL1]], %[[SHAPE1]] : i64
  // CHECK:       %[[OK1:.*]] = llvm.and %[[GE1]], %[[LT1]] : i1
  // CHECK:       triton_gen.predicated_store %{{.*}}, %{{.*}}, %[[OK1]]
  // CHECK:       llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[XOFF1]],
  // CHECK:       llvm.add %[[Y64]], %{{.*}} : i64
  // CHECK-COUNT-7: triton_gen.sub_group_block_write %{{.*}}, %{{.*}} : !llvm.ptr<1>, vector<2xi16>
  // CHECK-NOT:   triton_gen.sub_group_block_write
  tt.func public @fast_bf16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<16xi32, #x>, %arg2: tensor<16x64xbf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x64xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x64xbf16>, tensor<16xi32, #x>, i32, tensor<16x64xbf16, #dpas>
    tt.return
  }

  // fast_f32: a blocked layout whose lanes are adjacent columns: 2 rows of 4 registers (N = 2).
  // CHECK-LABEL: llvm.func spir_kernelcc @fast_f32
  // CHECK-COUNT-2: triton_gen.sub_group_block_write %{{.*}}, %{{.*}} : !llvm.ptr<1>, vector<4xi32>
  // CHECK-NOT:   triton_gen.sub_group_block_write
  tt.func public @fast_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: tensor<8xi32, #xb>, %arg2: tensor<8x64xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <1x64xf32>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x64xf32>, tensor<8xi32, #xb>, i32, tensor<8x64xf32, #blocked>
    tt.return
  }

  // window_signed: a negative y must fail the window, so y is sign-extended.
  // CHECK-LABEL: llvm.func spir_kernelcc @window_signed
  // CHECK-NOT:   llvm.zext %arg2
  // CHECK:       %[[Y64:.*]] = llvm.sext %arg2 : i32 to i64
  // CHECK:       %[[C0:.*]] = llvm.add %[[Y64]], %{{.*}} : i64
  // CHECK:       llvm.icmp "sle" %[[C0]], %{{.*}} : i64
  // CHECK-NOT:   llvm.zext %arg2
  tt.func public @window_signed(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<16xi32, #x>, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<16x64xbf16, #dpas>) {
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x64xbf16>
    ttig.descriptor_scatter %desc[%arg1, %arg2], %arg3 : !tt.tensordesc<1x64xbf16>, tensor<16xi32, #x>, i32, tensor<16x64xbf16, #dpas>
    tt.return
  }

  // warp_replicated: an 8x16 tile repeats registers 8-15 and every warp: one warp writes 8 runs.
  // CHECK-LABEL: llvm.func spir_kernelcc @warp_replicated
  // CHECK:       %[[SG:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
  // CHECK:       %[[SG64:.*]] = llvm.zext %[[SG]] : i32 to i64
  // CHECK:       %[[WID:.*]] = llvm.trunc %[[SG64]] : i64 to i32
  // CHECK:       %[[MASKED:.*]] = llvm.and %[[WID]], %{{.*}} : i32
  // CHECK:       llvm.icmp "eq" %[[MASKED]], %{{.*}} : i32
  // CHECK-COUNT-8: triton_gen.sub_group_block_write %{{.*}}, %{{.*}} : !llvm.ptr<1>, i16
  // CHECK-NOT:   triton_gen.sub_group_block_write
  tt.func public @warp_replicated(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<8xi32, #x>, %arg2: tensor<8x16xbf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x16xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x16xbf16>, tensor<8xi32, #x>, i32, tensor<8x16xbf16, #dpas>
    tt.return
  }

  // fast_requires_unit_stride: fast_bf16 with stride1 = 2, which leaves one contiguous element.
  // CHECK-LABEL: llvm.func spir_kernelcc @fast_requires_unit_stride
  // CHECK:       %[[STRIDE1:.*]] = llvm.extractvalue %{{.*}}[3] : !llvm.struct<(i64, i64, i64, i64, ptr<1>)>
  // CHECK:       llvm.mul %{{.*}}, %[[STRIDE1]] : i64
  // CHECK:       triton_gen.predicated_store
  // CHECK-NOT:   triton_gen.sub_group_block_write
  tt.func public @fast_requires_unit_stride(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<16xi32, #x>, %arg2: tensor<16x64xbf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i64 = arith.constant 2 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c2_i64] : <bf16>, <1x64xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x64xbf16>, tensor<16xi32, #x>, i32, tensor<16x64xbf16, #dpas>
    tt.return
  }

  // scalar_arm_blocked: lanes 2 columns apart, and lanes 8-15 repeat lanes 0-7.
  // CHECK-LABEL: llvm.func spir_kernelcc @scalar_arm_blocked
  // CHECK:       %[[NOT_DUP:.*]] = llvm.icmp "eq" %{{.*}}, %{{.*}} : i32
  // CHECK:       %[[ROW_OK:.*]] = llvm.icmp "ult" %{{.*}}, %{{.*}} : i64
  // CHECK:       %[[GE:.*]] = llvm.icmp "sge" %[[COL:[^,]*]], %{{.*}} : i64
  // CHECK:       %[[LT:.*]] = llvm.icmp "slt" %[[COL]], %{{.*}} : i64
  // CHECK:       %[[COL_OK:.*]] = llvm.and %[[GE]], %[[LT]] : i1
  // CHECK:       %[[IN:.*]] = llvm.and %[[ROW_OK]], %[[COL_OK]] : i1
  // CHECK:       %[[PRED:.*]] = llvm.and %[[IN]], %[[NOT_DUP]] : i1
  // CHECK:       triton_gen.predicated_store %{{.*}}, %{{.*}}, %[[PRED]] {cache_control = Default} : (!llvm.ptr<1>, i32, i1) -> ()
  // CHECK-NOT:   triton_gen.sub_group_block_write
  tt.func public @scalar_arm_blocked(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<4xi32, #xs>, %arg2: tensor<4x16xbf16, #strided>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x16xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x16xbf16>, tensor<4xi32, #xs>, i32, tensor<4x16xbf16, #strided>
    tt.return
  }

  // scalar_unaligned_y: warp_replicated with an unaligned y: 8 stores per lane, not 16.
  // CHECK-LABEL: llvm.func spir_kernelcc @scalar_unaligned_y
  // CHECK:       llvm.sext %arg2 : i32 to i64
  // CHECK-NOT:   {{llvm.zext %arg2|sub_group_block_write}}
  // CHECK-COUNT-8: triton_gen.predicated_store
  // CHECK-NOT:   {{llvm.zext %arg2|sub_group_block_write|predicated_store}}
  tt.func public @scalar_unaligned_y(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<8xi32, #x>, %arg2: i32, %arg3: tensor<8x16xbf16, #dpas>) {
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x16xbf16>
    ttig.descriptor_scatter %desc[%arg1, %arg2], %arg3 : !tt.tensordesc<1x16xbf16>, tensor<8xi32, #x>, i32, tensor<8x16xbf16, #dpas>
    tt.return
  }
}

// -----

// no_predicated_io: fast_bf16 without ttig.support_predicated_io.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#x = #ttg.slice<{dim = 1, parent = #dpas}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @no_predicated_io(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<16xi32, #x>, %arg2: tensor<16x64xbf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x64xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x64xbf16>, tensor<16xi32, #x>, i32, tensor<16x64xbf16, #dpas>
    tt.return
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @no_predicated_io
// CHECK:       triton_gen.sub_group_block_write
// CHECK:       llvm.store
// CHECK-NOT:   triton_gen.predicated_store

// -----

// scalar_arm_tpw32: with 32 lanes, lane 16 starts the next row.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [2, 2], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#x = #ttg.slice<{dim = 1, parent = #dpas}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  tt.func public @scalar_arm_tpw32(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<32xi32, #x>, %arg2: tensor<32x64xbf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x64xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x64xbf16>, tensor<32xi32, #x>, i32, tensor<32x64xbf16, #dpas>
    tt.return
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @scalar_arm_tpw32
// CHECK-NOT:   triton_gen.sub_group_block_write
// CHECK:       triton_gen.predicated_store {{.*}} : (!llvm.ptr<1>, i16, i1) -> ()
// CHECK-NOT:   triton_gen.sub_group_block_write

// -----

// two_runs_per_row: fast_bf16 with 128 columns, so each row's second run starts at column 64.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#x = #ttg.slice<{dim = 1, parent = #dpas}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_predicated_io} {
  tt.func public @two_runs_per_row(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<16xi32, #x>, %arg2: tensor<16x128xbf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x128xbf16>
    ttig.descriptor_scatter %desc[%arg1, %c0_i32], %arg2 : !tt.tensordesc<1x128xbf16>, tensor<16xi32, #x>, i32, tensor<16x128xbf16, #dpas>
    tt.return
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @two_runs_per_row
// CHECK:       %[[COL:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:       %[[DIM1:.*]] = llvm.xor %[[COL]], %{{.*}} : i32
// CHECK:       llvm.zext %[[DIM1]] : i32 to i64
