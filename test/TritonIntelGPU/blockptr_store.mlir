// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c32_i32 = arith.constant 32 : i32
    %c-64_i32 = arith.constant -64 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %3 = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %6 = tt.make_tensor_ptr %arg1, [%arg3, %arg4], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    %7 = tt.advance %3, [%c64_i32, %c-32_i32] : <tensor<64x32xf16, #dot0>>
    %8 = tt.advance %7, [%c-64_i32, %c32_i32] : <tensor<64x32xf16, #dot0>>
    %9 = tt.load %8 {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %10 = tt.load %6 {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %11 = tt.dot %9, %10, %cst, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %12 = arith.truncf %11#0 : tensor<64x64xf32, #dpas> to tensor<64x64xf16, #dpas>
    %13 = tt.make_tensor_ptr %arg2, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>
    // The next two lines is used to start checking constant related to the BlockStore.
    // CHECK-COUNT-3: llvm.call spir_funccc @_Z16get_sub_group_id
    // CHECK-COUNT-39: llvm.extractvalue
    // Next constant must be equal to warpsPerCTA[0]
    // CHECK: %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[VAL_0:.*]] = llvm.urem %{{[0-9]+}}, %[[CST_4]] : i32
    // Next constant must be equal to warpsPerCTA[1]
    // CHECK: %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[VAL_1:.*]] = llvm.urem %{{[0-9]+}}, %[[CST_2]] : i32
    // Next constant must is elemsPerInstr[0]
    // CHECK: %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK: llvm.mul %[[VAL_0]], %[[CST_8]] : i32
    // Next constant must is elemsPerInstr[1]
    // CHECK: %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.mul %[[VAL_1]], %[[CST_16]] : i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %13, %12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dpas_layout_2d_store_rep_cluster_4_2(
// CHECK-SAME:      %[[base:.*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[width:.*]]: i64, %[[height:.*]]: i64, %[[rowStride:.*]]: i64) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>} {
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
    // CHECK:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[SUB_GROUP_ID_N:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[CST_1]]  : i32
    // CHECK:           %[[SUB_GROUP_ID_M_:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[CST_1]]  : i32
    // CHECK:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[SUB_GROUP_ID_M:.*]] = llvm.urem %[[SUB_GROUP_ID_M_]], %[[CST_1]]  : i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_PTR]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_PTR]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE_PTR:.*]] = llvm.extractvalue %[[BLOCK_PTR]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    %13 = tt.make_tensor_ptr %base, [%width, %height], [%rowStride, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dpas>>

    // COM: The decomposed values of the tensor with DPAS layout.
    // CHECK:           %[[VAL_97:.*]] = llvm.extractvalue %[[VAL_71]][0]
    // CHECK:           %[[VAL_98:.*]] = llvm.extractvalue %[[VAL_71]][1]
    // CHECK:           %[[VAL_99:.*]] = llvm.extractvalue %[[VAL_71]][2]
    // CHECK:           %[[VAL_100:.*]] = llvm.extractvalue %[[VAL_71]][3]
    // CHECK:           %[[VAL_101:.*]] = llvm.extractvalue %[[VAL_71]][4]
    // CHECK:           %[[VAL_102:.*]] = llvm.extractvalue %[[VAL_71]][5]
    // CHECK:           %[[VAL_103:.*]] = llvm.extractvalue %[[VAL_71]][6]
    // CHECK:           %[[VAL_104:.*]] = llvm.extractvalue %[[VAL_71]][7]
    // CHECK:           %[[VAL_105:.*]] = llvm.extractvalue %[[VAL_71]][8]
    // CHECK:           %[[VAL_106:.*]] = llvm.extractvalue %[[VAL_71]][9]
    // CHECK:           %[[VAL_107:.*]] = llvm.extractvalue %[[VAL_71]][10]
    // CHECK:           %[[VAL_108:.*]] = llvm.extractvalue %[[VAL_71]][11]
    // CHECK:           %[[VAL_109:.*]] = llvm.extractvalue %[[VAL_71]][12]
    // CHECK:           %[[VAL_110:.*]] = llvm.extractvalue %[[VAL_71]][13]
    // CHECK:           %[[VAL_111:.*]] = llvm.extractvalue %[[VAL_71]][14]
    // CHECK:           %[[VAL_112:.*]] = llvm.extractvalue %[[VAL_71]][15]
    // CHECK:           %[[VAL_113:.*]] = llvm.extractvalue %[[VAL_71]][16]
    // CHECK:           %[[VAL_114:.*]] = llvm.extractvalue %[[VAL_71]][17]
    // CHECK:           %[[VAL_115:.*]] = llvm.extractvalue %[[VAL_71]][18]
    // CHECK:           %[[VAL_116:.*]] = llvm.extractvalue %[[VAL_71]][19]
    // CHECK:           %[[VAL_117:.*]] = llvm.extractvalue %[[VAL_71]][20]
    // CHECK:           %[[VAL_118:.*]] = llvm.extractvalue %[[VAL_71]][21]
    // CHECK:           %[[VAL_119:.*]] = llvm.extractvalue %[[VAL_71]][22]
    // CHECK:           %[[VAL_120:.*]] = llvm.extractvalue %[[VAL_71]][23]
    // CHECK:           %[[VAL_121:.*]] = llvm.extractvalue %[[VAL_71]][24]
    // CHECK:           %[[VAL_122:.*]] = llvm.extractvalue %[[VAL_71]][25]
    // CHECK:           %[[VAL_123:.*]] = llvm.extractvalue %[[VAL_71]][26]
    // CHECK:           %[[VAL_124:.*]] = llvm.extractvalue %[[VAL_71]][27]
    // CHECK:           %[[VAL_125:.*]] = llvm.extractvalue %[[VAL_71]][28]
    // CHECK:           %[[VAL_126:.*]] = llvm.extractvalue %[[VAL_71]][29]
    // CHECK:           %[[VAL_127:.*]] = llvm.extractvalue %[[VAL_71]][30]
    // CHECK:           %[[VAL_128:.*]] = llvm.extractvalue %[[VAL_71]][31]
    // CHECK:           %[[VAL_129:.*]] = llvm.extractvalue %[[VAL_71]][32]
    // CHECK:           %[[VAL_130:.*]] = llvm.extractvalue %[[VAL_71]][33]
    // CHECK:           %[[VAL_131:.*]] = llvm.extractvalue %[[VAL_71]][34]
    // CHECK:           %[[VAL_132:.*]] = llvm.extractvalue %[[VAL_71]][35]
    // CHECK:           %[[VAL_133:.*]] = llvm.extractvalue %[[VAL_71]][36]
    // CHECK:           %[[VAL_134:.*]] = llvm.extractvalue %[[VAL_71]][37]
    // CHECK:           %[[VAL_135:.*]] = llvm.extractvalue %[[VAL_71]][38]
    // CHECK:           %[[VAL_136:.*]] = llvm.extractvalue %[[VAL_71]][39]
    // CHECK:           %[[VAL_137:.*]] = llvm.extractvalue %[[VAL_71]][40]
    // CHECK:           %[[VAL_138:.*]] = llvm.extractvalue %[[VAL_71]][41]
    // CHECK:           %[[VAL_139:.*]] = llvm.extractvalue %[[VAL_71]][42]
    // CHECK:           %[[VAL_140:.*]] = llvm.extractvalue %[[VAL_71]][43]
    // CHECK:           %[[VAL_141:.*]] = llvm.extractvalue %[[VAL_71]][44]
    // CHECK:           %[[VAL_142:.*]] = llvm.extractvalue %[[VAL_71]][45]
    // CHECK:           %[[VAL_143:.*]] = llvm.extractvalue %[[VAL_71]][46]
    // CHECK:           %[[VAL_144:.*]] = llvm.extractvalue %[[VAL_71]][47]
    // CHECK:           %[[VAL_145:.*]] = llvm.extractvalue %[[VAL_71]][48]
    // CHECK:           %[[VAL_146:.*]] = llvm.extractvalue %[[VAL_71]][49]
    // CHECK:           %[[VAL_147:.*]] = llvm.extractvalue %[[VAL_71]][50]
    // CHECK:           %[[VAL_148:.*]] = llvm.extractvalue %[[VAL_71]][51]
    // CHECK:           %[[VAL_149:.*]] = llvm.extractvalue %[[VAL_71]][52]
    // CHECK:           %[[VAL_150:.*]] = llvm.extractvalue %[[VAL_71]][53]
    // CHECK:           %[[VAL_151:.*]] = llvm.extractvalue %[[VAL_71]][54]
    // CHECK:           %[[VAL_152:.*]] = llvm.extractvalue %[[VAL_71]][55]
    // CHECK:           %[[VAL_153:.*]] = llvm.extractvalue %[[VAL_71]][56]
    // CHECK:           %[[VAL_154:.*]] = llvm.extractvalue %[[VAL_71]][57]
    // CHECK:           %[[VAL_155:.*]] = llvm.extractvalue %[[VAL_71]][58]
    // CHECK:           %[[VAL_156:.*]] = llvm.extractvalue %[[VAL_71]][59]
    // CHECK:           %[[VAL_157:.*]] = llvm.extractvalue %[[VAL_71]][60]
    // CHECK:           %[[VAL_158:.*]] = llvm.extractvalue %[[VAL_71]][61]
    // CHECK:           %[[VAL_159:.*]] = llvm.extractvalue %[[VAL_71]][62]
    // CHECK:           %[[VAL_160:.*]] = llvm.extractvalue %[[VAL_71]][63]

    // CHECK:           %[[HEIGHT_i32:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ROW_STRIDE_i32:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[baseWidth:.*]] = llvm.mul %[[HEIGHT_i32]], %[[CST_2]] : i32
    // CHECK:           %[[basePitch:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_166:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[SUB_GROUP_ID_M]], %[[VAL_166]]  : i32
    // CHECK:           %[[VAL_168:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[innerDimWarpId:.*]] = llvm.urem %[[SUB_GROUP_ID_N]], %[[VAL_168]]  : i32
    // CHECK:           %[[VAL_170:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[dimWarpId0:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_170]] : i32
    // CHECK:           %[[VAL_172:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[dimWarpId1:.*]] = llvm.mul %[[innerDimWarpId]], %[[VAL_172]] : i32
    // CHECK:           %[[warpId0Offset:.*]] = llvm.add %[[dimWarpId0]], %[[OFFSET_0]] : i32
    // CHECK:           %[[warpId1Offset:.*]] = llvm.add %[[dimWarpId1]], %[[OFFSET_1]] : i32
    // CHECK:           %[[VAL_176:.*]] = llvm.mlir.constant(0 : i32) : i32


    // COM: The shape of DPAS layout replica is [4, 2]
    // COM: The replica order are [0, 1]
    // COM:                       [2, 3]
    // COM:                       [4, 5]
    // COM:                       [6, 7]

    // COM: replica [0, 0]
    // CHECK:           %[[offsetY:.*]] = llvm.add %[[warpId0Offset]], %[[VAL_176]] : i32
    // CHECK:           %[[VAL_178:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_178]] : i32
    // CHECK:           %[[VAL_180:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_182:.*]] = llvm.insertelement %[[VAL_97]], %[[VAL_180]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_184:.*]] = llvm.insertelement %[[VAL_98]], %[[VAL_182]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_186:.*]] = llvm.insertelement %[[VAL_99]], %[[VAL_184]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_188:.*]] = llvm.insertelement %[[VAL_100]], %[[VAL_186]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_190:.*]] = llvm.insertelement %[[VAL_101]], %[[VAL_188]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_192:.*]] = llvm.insertelement %[[VAL_102]], %[[VAL_190]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_194:.*]] = llvm.insertelement %[[VAL_103]], %[[VAL_192]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_196:.*]] = llvm.insertelement %[[VAL_104]], %[[VAL_194]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_197:.*]] = llvm.bitcast %[[VAL_196]] : vector<8xf16> to vector<8xi16>
    // CHECK:           %[[VAL_198:.*]] = llvm.trunc %[[offsetY]] : i32 to i32
    // CHECK:           %[[VAL_199:.*]] = llvm.trunc %[[offsetX]] : i32 to i32
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], %[[VAL_199]], %[[VAL_198]], %[[VAL_197]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [0, 1]
    // CHECK:           %[[VAL_207:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_208:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_207]] : i32
    // CHECK:           %[[VAL_209:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_211:.*]] = llvm.insertelement %[[VAL_105]], %[[VAL_209]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_213:.*]] = llvm.insertelement %[[VAL_106]], %[[VAL_211]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_215:.*]] = llvm.insertelement %[[VAL_107]], %[[VAL_213]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_217:.*]] = llvm.insertelement %[[VAL_108]], %[[VAL_215]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_219:.*]] = llvm.insertelement %[[VAL_109]], %[[VAL_217]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_221:.*]] = llvm.insertelement %[[VAL_110]], %[[VAL_219]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_223:.*]] = llvm.insertelement %[[VAL_111]], %[[VAL_221]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_225:.*]] = llvm.insertelement %[[VAL_112]], %[[VAL_223]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_226:.*]] = llvm.bitcast %[[VAL_225]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_226]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [1, 0]
    // CHECK:           %[[VAL_236:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_237:.*]] = llvm.add %[[warpId0Offset]], %[[VAL_236]] : i32
    // CHECK:           %[[VAL_238:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_239:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_238]] : i32
    // CHECK:           %[[VAL_240:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_242:.*]] = llvm.insertelement %[[VAL_113]], %[[VAL_240]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_244:.*]] = llvm.insertelement %[[VAL_114]], %[[VAL_242]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_246:.*]] = llvm.insertelement %[[VAL_115]], %[[VAL_244]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_248:.*]] = llvm.insertelement %[[VAL_116]], %[[VAL_246]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_250:.*]] = llvm.insertelement %[[VAL_117]], %[[VAL_248]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_252:.*]] = llvm.insertelement %[[VAL_118]], %[[VAL_250]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_254:.*]] = llvm.insertelement %[[VAL_119]], %[[VAL_252]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_256:.*]] = llvm.insertelement %[[VAL_120]], %[[VAL_254]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_257:.*]] = llvm.bitcast %[[VAL_256]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_257]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [1, 1]
    // CHECK:           %[[VAL_267:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_268:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_267]] : i32
    // CHECK:           %[[VAL_269:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_271:.*]] = llvm.insertelement %[[VAL_121]], %[[VAL_269]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_273:.*]] = llvm.insertelement %[[VAL_122]], %[[VAL_271]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_275:.*]] = llvm.insertelement %[[VAL_123]], %[[VAL_273]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_277:.*]] = llvm.insertelement %[[VAL_124]], %[[VAL_275]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_279:.*]] = llvm.insertelement %[[VAL_125]], %[[VAL_277]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_281:.*]] = llvm.insertelement %[[VAL_126]], %[[VAL_279]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_283:.*]] = llvm.insertelement %[[VAL_127]], %[[VAL_281]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_285:.*]] = llvm.insertelement %[[VAL_128]], %[[VAL_283]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_286:.*]] = llvm.bitcast %[[VAL_285]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_286]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [2, 0]
    // CHECK:           %[[VAL_296:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.add %[[warpId0Offset]], %[[VAL_296]] : i32
    // CHECK:           %[[VAL_298:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_299:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_298]] : i32
    // CHECK:           %[[VAL_300:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_302:.*]] = llvm.insertelement %[[VAL_129]], %[[VAL_300]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_304:.*]] = llvm.insertelement %[[VAL_130]], %[[VAL_302]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_306:.*]] = llvm.insertelement %[[VAL_131]], %[[VAL_304]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_308:.*]] = llvm.insertelement %[[VAL_132]], %[[VAL_306]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_310:.*]] = llvm.insertelement %[[VAL_133]], %[[VAL_308]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_312:.*]] = llvm.insertelement %[[VAL_134]], %[[VAL_310]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_314:.*]] = llvm.insertelement %[[VAL_135]], %[[VAL_312]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_316:.*]] = llvm.insertelement %[[VAL_136]], %[[VAL_314]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_317:.*]] = llvm.bitcast %[[VAL_316]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_317]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [2, 1]
    // CHECK:           %[[VAL_327:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_328:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_327]] : i32
    // CHECK:           %[[VAL_329:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_331:.*]] = llvm.insertelement %[[VAL_137]], %[[VAL_329]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_333:.*]] = llvm.insertelement %[[VAL_138]], %[[VAL_331]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_335:.*]] = llvm.insertelement %[[VAL_139]], %[[VAL_333]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_337:.*]] = llvm.insertelement %[[VAL_140]], %[[VAL_335]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_339:.*]] = llvm.insertelement %[[VAL_141]], %[[VAL_337]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_341:.*]] = llvm.insertelement %[[VAL_142]], %[[VAL_339]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_343:.*]] = llvm.insertelement %[[VAL_143]], %[[VAL_341]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_345:.*]] = llvm.insertelement %[[VAL_144]], %[[VAL_343]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_346:.*]] = llvm.bitcast %[[VAL_345]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_346]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [3, 0]
    // CHECK:           %[[VAL_356:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_357:.*]] = llvm.add %[[warpId0Offset]], %[[VAL_356]] : i32
    // CHECK:           %[[VAL_358:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_359:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_358]] : i32
    // CHECK:           %[[VAL_360:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_362:.*]] = llvm.insertelement %[[VAL_145]], %[[VAL_360]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_364:.*]] = llvm.insertelement %[[VAL_146]], %[[VAL_362]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_366:.*]] = llvm.insertelement %[[VAL_147]], %[[VAL_364]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_368:.*]] = llvm.insertelement %[[VAL_148]], %[[VAL_366]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_370:.*]] = llvm.insertelement %[[VAL_149]], %[[VAL_368]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_372:.*]] = llvm.insertelement %[[VAL_150]], %[[VAL_370]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_374:.*]] = llvm.insertelement %[[VAL_151]], %[[VAL_372]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_376:.*]] = llvm.insertelement %[[VAL_152]], %[[VAL_374]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_377:.*]] = llvm.bitcast %[[VAL_376]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_377]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // COM: replica [3, 1]
    // CHECK:           %[[VAL_387:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_388:.*]] = llvm.add %[[warpId1Offset]], %[[VAL_387]] : i32
    // CHECK:           %[[VAL_389:.*]] = llvm.mlir.undef : vector<8xf16>
    // CHECK:           %[[VAL_391:.*]] = llvm.insertelement %[[VAL_153]], %[[VAL_389]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_393:.*]] = llvm.insertelement %[[VAL_154]], %[[VAL_391]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_395:.*]] = llvm.insertelement %[[VAL_155]], %[[VAL_393]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_397:.*]] = llvm.insertelement %[[VAL_156]], %[[VAL_395]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_399:.*]] = llvm.insertelement %[[VAL_157]], %[[VAL_397]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_401:.*]] = llvm.insertelement %[[VAL_158]], %[[VAL_399]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_403:.*]] = llvm.insertelement %[[VAL_159]], %[[VAL_401]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_405:.*]] = llvm.insertelement %[[VAL_160]], %[[VAL_403]]{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           %[[VAL_406:.*]] = llvm.bitcast %[[VAL_405]] : vector<8xf16> to vector<8xi16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %{{.*}}, %[[WIDTH_i32]], %[[basePitch]], {{.*}}, %[[VAL_406]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    tt.store %13, %cst {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xf16, #dpas>>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @boundary_check
  tt.func public @boundary_check(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #blocked>
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #blocked>>
      // CHECK: llvm.call spir_funccc @_Z12get_local_idj
      // CHECK-NOT: llvm.icmp "slt"
      // CHECK: %[[threadID:.*]] = llvm.call spir_funccc @_Z12get_local_idj
      // CHECK: %[[VAL_583:.*]] = llvm.trunc %[[threadID]] : i64 to i32
      // CHECK: %[[VAL_584:.*]] = llvm.mlir.constant(16 : i32) : i32
      // CHECK: %[[VAL_586:.*]] = llvm.udiv %[[VAL_583]], %[[VAL_584]] : i32
      // CHECK: %[[VAL_587:.*]] = llvm.mlir.constant(3 : i32) : i32
      // CHECK: %[[VAL_588:.*]] = llvm.and %[[VAL_586]], %[[VAL_587]] : i32
      // CHECK: %[[threadPred:.*]] = llvm.icmp "eq" %[[VAL_588]], {{.*}} : i32
      // CHECK-COUNT-32: llvm.cond_br %[[threadPred]]
      tt.store %0, %cst : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-16: llvm.icmp "slt"
      // CHECK: %[[threadPred_0:.*]] = llvm.icmp "eq"
      // CHECK-COUNT-32: llvm.and %[[threadPred_0]], {{.*}} : i1
      tt.store %0, %cst {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-16: llvm.icmp "slt"
      // CHECK: %[[threadPred_1:.*]] = llvm.icmp "eq"
      // CHECK-COUNT-32: llvm.and %[[threadPred_1]], {{.*}} : i1
      tt.store %0, %cst {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-32: llvm.icmp "slt"
      // CHECK: %[[threadPred_2:.*]] = llvm.icmp "eq"
      // CHECK-COUNT-32: llvm.and %[[threadPred_2]], {{.*}} : i1
      tt.store %0, %cst {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      tt.return
  }
}
