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
