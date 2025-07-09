// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>
    // CHECK: %[[offsetBaseY:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[offsetBaseX:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[baseHeight:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[baseWidth:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[rowStride:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[colStride:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[base:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[offsetBaseY_0:.*]] = llvm.trunc %[[offsetBaseY]] : i32 to i32
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(f16, f16, {{.*}})>
    // CHECK-COUNT-1: llvm.call spir_funccc @_Z12get_local_idj
    // CHECK: %[[WARP_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // COM: Skip the register, lane, warp and block to the offset computation which should be covered by the LL tests.
    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY_0]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY_0]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY_0]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY_0]], {{.*}} : i32
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %13, %cst {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dpas_layout_2d_store_rep_cluster_4_2(
// CHECK-SAME:      %[[base:.*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[width:.*]]: i64, %[[height:.*]]: i64, %[[rowStride:.*]]: i64, %[[PTR_1:.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, triton_gen.max_work_group_size = array<i32: 16, 1, 1>} {
  tt.func public @dpas_layout_2d_store_rep_cluster_4_2(%base: !tt.ptr<f16>, %width: i64, %height: i64, %rowStride: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
    // CHECK:           %[[CST_FP16_0:.*]] = llvm.bitcast %[[VAL_5]] : f16 to f16
    // CHECK:           %[[VAL_71:.*]] = llvm.insertvalue %[[CST_FP16_0]], {{.*}}[63]

    // COM: The block pointer.
    // CHECK:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_74:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_75:.*]] = llvm.insertvalue %[[CST_0]], %[[VAL_74]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_76:.*]] = llvm.insertvalue %[[CST_0]], %[[VAL_75]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_77:.*]] = llvm.insertvalue %[[width]], %[[VAL_76]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_78:.*]] = llvm.insertvalue %[[height]], %[[VAL_77]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_79:.*]] = llvm.insertvalue %[[rowStride]], %[[VAL_78]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_80:.*]] = llvm.insertvalue %[[CST_1]], %[[VAL_79]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_PTR:.*]] = llvm.insertvalue %[[base]], %[[VAL_80]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SCALAR_BYTES:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[OFF_0:.*]] = llvm.extractvalue %[[BLOCK_PTR]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFF_1:.*]] = llvm.extractvalue %[[BLOCK_PTR]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE_PTR:.*]] = llvm.extractvalue %[[BLOCK_PTR]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[WIDTH:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_IN_BYTES:.*]] = llvm.mul %[[WIDTH]], %[[SCALAR_BYTES]] : i32
    // CHECK:           %[[ROW_STRIDE:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE]], %[[SCALAR_BYTES]] : i32
    // CHECK:           %[[OFFSET_1:.*]] = llvm.trunc %[[OFF_1]] : i32 to i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.trunc %[[OFF_0]] : i32 to i32
    %13 = tt.make_tensor_ptr %base, [%width, %height], [%rowStride, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dpas>>

    // COM: The shape of DPAS layout replica is [4, 2]
    // COM: The replica order are [0, 1]
    // COM:                       [2, 3]
    // COM:                       [4, 5]
    // COM:                       [6, 7]

    // COM: replica [0, 0]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_183:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_184:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_185:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_186:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_186]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_186]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [0, 1]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_207:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_208:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_209:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_210:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_211:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_211]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_210]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [1, 0]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_232:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_233:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_234:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_235:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_236:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_235]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_236]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)


    // COM: replica [1, 1]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_257:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_258:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_259:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_260:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_261:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_262:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_262]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_261]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [2, 0]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_283:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_284:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_285:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_286:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_287:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_286]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_287]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)


    // COM: replica [2, 1]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_308:.*]] = llvm.mlir.constant(40 : i32) : i32
    // CHECK:           %[[VAL_309:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_310:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_311:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_312:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_313]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_312]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)


    // COM: replica [3, 0]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_334:.*]] = llvm.mlir.constant(48 : i32) : i32
    // CHECK:           %[[VAL_335:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_336:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_337:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_338:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_337]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_338]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)


    // COM: replica [3, 1]
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_359:.*]] = llvm.mlir.constant(56 : i32) : i32
    // CHECK:           %[[VAL_360:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_361:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_362:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_363:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_364:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], %[[VAL_364]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], %[[VAL_363]] : i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    tt.store %13, %cst {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dpas>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @block_store_no_boundary_check(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>
    // CHECK: %[[OFF_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[OFF_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[colStride:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[base:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

    // CHECK:           %[[HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[WIDTH_IN_BYTES:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK:           %[[ROW_STRIDE:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE]], {{.*}} : i32
    // CHECK:           %[[OFFSET_1:.*]] = llvm.trunc %[[OFF_1]] : i32 to i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.trunc %[[OFF_0]] : i32 to i32

    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(f16, f16, {{.*}})>
    // CHECK-COUNT-1: llvm.call spir_funccc @_Z12get_local_idj
    // CHECK: %[[WARP_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // COM: Skip the register, lane, warp and block to the offset computation which should be covered by the LL tests.
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], {{.*}} : i32
    // CHECK:           %[[BASE_WITH_OFF:.*]] = llvm.getelementptr %[[base]]{{\[}}%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    // CHECK:           %[[CST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], {{.*}} : i32
    // CHECK:           %[[OFFSET_Y_IN_BYTES:.*]] = llvm.mul %[[OFFSET_Y]], %[[ROW_STRIDE_IN_BYTES]] : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_WITH_OFF]]{{\[}}%[[OFFSET_Y_IN_BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK:           %[[CST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.2Dblockstore %[[BASE]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], {{.*}}, %[[CST_0_1]], %[[CST_0_2]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], {{.*}} : i32
    // CHECK:           %[[BASE_WITH_OFF:.*]] = llvm.getelementptr %[[base]]{{\[}}%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    // CHECK:           %[[CST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], {{.*}} : i32
    // CHECK:           %[[OFFSET_Y_IN_BYTES:.*]] = llvm.mul %[[OFFSET_Y]], %[[ROW_STRIDE_IN_BYTES]] : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_WITH_OFF]]{{\[}}%[[OFFSET_Y_IN_BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK:           %[[CST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.2Dblockstore %[[BASE]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], {{.*}}, %[[CST_0_1]], %[[CST_0_2]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], {{.*}} : i32
    // CHECK:           %[[BASE_WITH_OFF:.*]] = llvm.getelementptr %[[base]]{{\[}}%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    // CHECK:           %[[CST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], {{.*}} : i32
    // CHECK:           %[[OFFSET_Y_IN_BYTES:.*]] = llvm.mul %[[OFFSET_Y]], %[[ROW_STRIDE_IN_BYTES]] : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_WITH_OFF]]{{\[}}%[[OFFSET_Y_IN_BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK:           %[[CST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.2Dblockstore %[[BASE]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], {{.*}}, %[[CST_0_1]], %[[CST_0_2]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], {{.*}} : i32
    // CHECK:           %[[BASE_WITH_OFF:.*]] = llvm.getelementptr %[[base]]{{\[}}%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    // CHECK:           %[[CST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], {{.*}} : i32
    // CHECK:           %[[OFFSET_Y_IN_BYTES:.*]] = llvm.mul %[[OFFSET_Y]], %[[ROW_STRIDE_IN_BYTES]] : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_WITH_OFF]]{{\[}}%[[OFFSET_Y_IN_BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK:           %[[CST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.2Dblockstore %[[BASE]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], {{.*}}, %[[CST_0_1]], %[[CST_0_2]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %13, %cst {ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @block_store_deduplicate_no_boundary_check(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #dpas>>
    // CHECK: %[[OFF_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[OFF_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[colStride:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[base:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

    // CHECK:           %[[HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[WIDTH:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_IN_BYTES:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK:           %[[ROW_STRIDE:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE]], {{.*}} : i32
    // CHECK:           %[[OFFSET_1:.*]] = llvm.trunc %[[OFF_1]] : i32 to i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.trunc %[[OFF_0]] : i32 to i32

    // CHECK-COUNT-16: llvm.extractvalue {{.*}} : !llvm.struct<(f16, f16, {{.*}})>
    // CHECK-COUNT-1: llvm.call spir_funccc @_Z12get_local_idj
    // CHECK: %[[WARP_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // COM: Skip the register, lane, warp and block to the offset computation which should be covered by the LL tests.
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], {{.*}} : i32
    // CHECK:           %[[BASE_WITH_OFF:.*]] = llvm.getelementptr %[[base]]{{\[}}%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    // CHECK:           %[[CST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], {{.*}} : i32
    // CHECK:           %[[OFFSET_Y_IN_BYTES:.*]] = llvm.mul %[[OFFSET_Y]], %[[ROW_STRIDE_IN_BYTES]] : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_WITH_OFF]]{{\[}}%[[OFFSET_Y_IN_BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK:           %[[CST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[DEDUP_OFFSET:.*]] = llvm.select{{.*}}, %[[CST_0_2]], %[[HEIGHT]] : i1, i32
    // CHECK: triton_gen.2Dblockstore %[[BASE]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], {{.*}}, %[[CST_0_1]], %[[DEDUP_OFFSET]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[OFFSET_X:.*]] = llvm.add %[[OFFSET_1]], {{.*}} : i32
    // CHECK:           %[[BASE_WITH_OFF:.*]] = llvm.getelementptr %[[base]]{{\[}}%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    // CHECK:           %[[CST_0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFFSET_0]], {{.*}} : i32
    // CHECK:           %[[OFFSET_Y_IN_BYTES:.*]] = llvm.mul %[[OFFSET_Y]], %[[ROW_STRIDE_IN_BYTES]] : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_WITH_OFF]]{{\[}}%[[OFFSET_Y_IN_BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK:           %[[CST_0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[DEDUP_OFFSET:.*]] = llvm.select{{.*}}, %[[CST_0_2]], %[[HEIGHT]] : i1, i32
    // CHECK: triton_gen.2Dblockstore %[[BASE]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], {{.*}}, %[[CST_0_1]], %[[DEDUP_OFFSET]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %13, %cst {ttig.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #dpas>>
    tt.return
  }
}
