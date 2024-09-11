// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm
// RUN: TRITON_INTEL_ENABLE_FAST_PREFETCH=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefix=FAST

// CHECK-DAG: llvm.func spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_4r16x2cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {passthrough = ["nounwind", ["memory", "1"]]}
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_with_prefetch(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
    // CHECK-LABEL: @matmul_with_prefetch
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // FAST: [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // FAST: [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // FAST: [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
    // FAST: [[C4:%.*]] = llvm.mlir.constant(4 : i32) : i32
    // FAST: [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // FAST: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[C16]], [[C32]], [[C4]], [[C1]], {{.*}}, {{.*}}, {{.*}}) {{.*}} : (i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> ()
    // CHECK: %[[ROW_MAJOR_BLOCK_PTR:.*]] = llvm.insertvalue %arg0, {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[VAL_17:.*]] = llvm.call spir_funccc @_Z16get_sub_group_idv()
    // CHECK: %[[VAL_18:.*]] = llvm.sext %[[VAL_17]] : i32 to i64
    // CHECK: %[[VAL_19:.*]] = llvm.trunc %[[VAL_18]] : i64 to i32
    // CHECK: %[[VAL_20:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[VAL_21:.*]] = llvm.urem %[[VAL_19]], %[[VAL_20]]  : i32
    // CHECK: %[[VAL_22:.*]] = llvm.udiv %[[VAL_19]], %[[VAL_20]]  : i32
    // CHECK: %[[VAL_23:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: %[[VAL_24:.*]] = llvm.urem %[[VAL_22]], %[[VAL_23]]  : i32
    // CHECK: %[[VAL_25:.*]] = llvm.udiv %[[VAL_22]], %[[VAL_23]]  : i32
    // CHECK: %[[ROW_MAJOR_OFFSET_Y:.*]] = llvm.extractvalue %[[ROW_MAJOR_BLOCK_PTR]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_MAJOR_OFFSET_X:.*]] = llvm.extractvalue %[[ROW_MAJOR_BLOCK_PTR]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_MAJOR_HEIGHT_:.*]] = llvm.extractvalue %[[ROW_MAJOR_BLOCK_PTR]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_MAJOR_WIDTH_:.*]] = llvm.extractvalue %[[ROW_MAJOR_BLOCK_PTR]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_MAJOR_ROW_STRIDE_:.*]] = llvm.extractvalue %[[ROW_MAJOR_BLOCK_PTR]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_MAJOR_BASE:.*]] = llvm.extractvalue %[[ROW_MAJOR_BLOCK_PTR]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[VAL_34:.*]] = llvm.mul %[[ROW_MAJOR_WIDTH_]], {{.*}} : i64
    // CHECK: %[[ROW_MAJOR_WIDTH:.*]] = llvm.trunc %[[VAL_34]] : i64 to i32
    // CHECK: %[[ROW_MAJOR_HEIGHT:.*]] = llvm.trunc %[[ROW_MAJOR_HEIGHT_]] : i64 to i32
    // CHECK: %[[ROW_MAJOR_ROW_STRIDE:.*]] = llvm.mul %[[ROW_MAJOR_ROW_STRIDE_]], {{.*}} : i64
    // CHECK: %[[ROW_MAJOR_STRIDE:.*]] = llvm.trunc %[[ROW_MAJOR_ROW_STRIDE]] : i64 to i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_X_:.*]] = llvm.add {{.*}}, %[[ROW_MAJOR_OFFSET_X]] : i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_Y_:.*]] = llvm.add {{.*}}, %[[ROW_MAJOR_OFFSET_Y]] : i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_Y:.*]] = llvm.trunc %[[COLUMN_MAJOR_WARP_OFF_Y_]] : i32 to i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_X:.*]] = llvm.trunc %[[COLUMN_MAJOR_WARP_OFF_X_]] : i32 to i32
    // CHECK: %[[VAL_56:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[VAL_57:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[VAL_59:.*]] = llvm.insertelement %[[COLUMN_MAJOR_WARP_OFF_X]],  {{.*}}{{\[}}%[[VAL_57]] : i32] : vector<2xi32>
    // CHECK: %[[ROW_MAJOR_COORD:.*]] = llvm.insertelement %[[COLUMN_MAJOR_WARP_OFF_Y]],  {{.*}}{{\[}}%[[VAL_56]] : i32] : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_4r16x2cPU3AS1viiiDv2_i(%[[ROW_MAJOR_BASE]], %[[ROW_MAJOR_WIDTH]], %[[ROW_MAJOR_HEIGHT]], %[[ROW_MAJOR_STRIDE]], %[[ROW_MAJOR_COORD]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>) -> ()
    %rowMajorPtr = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16>>
    triton_intel_gpu.prefetch %rowMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16>>

    // COM: The memory layout is same for the column major memory and row major memory. The prefetch function should be the same.

    // FAST: [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // FAST: [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // FAST: [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
    // FAST: [[C4:%.*]] = llvm.mlir.constant(4 : i32) : i32
    // FAST: [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // FAST: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[C16]], [[C32]], [[C4]], [[C1]], {{.*}}, {{.*}}, {{.*}}) {{.*}} : (i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> ()
    // CHECK: %[[COLUMN_MAJOR_BLOCK_PTR:.*]] = llvm.insertvalue %arg1, {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[COLUMN_MAJOR_OFFSET_Y:.*]] = llvm.extractvalue %[[COLUMN_MAJOR_BLOCK_PTR]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[COLUMN_MAJOR_OFFSET_X:.*]] = llvm.extractvalue %[[COLUMN_MAJOR_BLOCK_PTR]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[COLUMN_MAJOR_HEIGHT_:.*]] = llvm.extractvalue %[[COLUMN_MAJOR_BLOCK_PTR]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[COLUMN_MAJOR_WIDTH:.*]] = llvm.extractvalue %[[COLUMN_MAJOR_BLOCK_PTR]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[COLUMN_MAJOR_COL_STRIDE:.*]] = llvm.extractvalue %[[COLUMN_MAJOR_BLOCK_PTR]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[COLUMN_MAJOR_BASE:.*]] = llvm.extractvalue %[[COLUMN_MAJOR_BLOCK_PTR]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[VAL_86:.*]] = llvm.mul %[[COLUMN_MAJOR_HEIGHT_]], {{.*}} : i64
    // CHECK: %[[COLUMN_MAJOR_HEIGHT:.*]] = llvm.trunc %[[VAL_86]] : i64 to i32
    // CHECK: %[[COLUMN_MAJOR_WIDTH_:.*]] = llvm.trunc %[[COLUMN_MAJOR_WIDTH]] : i64 to i32
    // CHECK: %[[VAL_90:.*]] = llvm.mul %[[COLUMN_MAJOR_COL_STRIDE]], {{.*}} : i64
    // CHECK: %[[COLUMN_MAJOR_STRIDE:.*]] = llvm.trunc %[[VAL_90]] : i64 to i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_X_:.*]] = llvm.add {{.*}}, %[[COLUMN_MAJOR_OFFSET_X]] : i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_Y_:.*]] = llvm.add {{.*}}, %[[COLUMN_MAJOR_OFFSET_Y]] : i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_Y:.*]] = llvm.trunc %[[COLUMN_MAJOR_WARP_OFF_Y_]] : i32 to i32
    // CHECK: %[[COLUMN_MAJOR_WARP_OFF_X:.*]] = llvm.trunc %[[COLUMN_MAJOR_WARP_OFF_X_]] : i32 to i32
    // CHECK: %[[VAL_108:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[VAL_109:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.insertelement %[[COLUMN_MAJOR_WARP_OFF_X]], {{.*}}{{\[}}%[[VAL_109]] : i32] : vector<2xi32>
    // CHECK: %[[COLUMN_MAJOR_COORD:.*]] = llvm.insertelement %[[COLUMN_MAJOR_WARP_OFF_Y]], {{.*}}{{\[}}%[[VAL_108]] : i32] : vector<2xi32>
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_4r16x2cPU3AS1viiiDv2_i(%[[COLUMN_MAJOR_BASE]], %[[COLUMN_MAJOR_HEIGHT]], %[[COLUMN_MAJOR_WIDTH_]], %[[COLUMN_MAJOR_STRIDE]], %[[COLUMN_MAJOR_COORD]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>) -> ()
    %columnMajorPtr = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%c1_i64, %arg6], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<16x32xf16>>
    triton_intel_gpu.prefetch %columnMajorPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false, triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<16x32xf16>>

    // CHECK-NOT: block_prefetch
    %nonContiguousPtr = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg6, %arg6], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    triton_intel_gpu.prefetch %nonContiguousPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x32xf16>>
    tt.return
  }
}
