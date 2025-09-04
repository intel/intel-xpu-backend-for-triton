// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm  --check-prefixes=ALL-LAYOUT

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {ttig.support_sg_2d_block,  "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @dot_a_layout
  tt.func public @dot_a_layout(%arg0: !tt.ptr<i8>, %col_stride: i64) {
      %cst = arith.constant dense<0> : tensor<256x64xi8, #dot_a>
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<256x64xi8, #dot_a>>
      // ALL-LAYOUT:           %[[OFF_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[OFF_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[BASE_PTR:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

      // ALL-LAYOUT:           %[[HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32

      // ALL-LAYOUT:           %[[OFFSET:.*]] = llvm.add %[[OFF_0]], {{.*}} : i32
      // ALL-LAYOUT:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
      // ALL-LAYOUT:           %[[OFFSET_X:.*]] = llvm.mlir.constant(0 : i32) : i32
      // ALL-LAYOUT:           %[[OFFSET_Y:.*]] = llvm.select {{.*}}, %[[OFFSET]], %[[HEIGHT]] : i1, i32
      // ALL-LAYOUT:           llvm.mlir.undef : vector<4xi8>
      // ALL-LAYOUT-COUNT-4:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<4xi8>
      // ALL-LAYOUT: triton_gen.2Dblockstore {{.*}}, %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 8, tile_width = 8, tile_height = 8, v_blocks = 1, cache_control = Default}
      tt.store %0, %cst {ttig.block_io = "row_major", boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<256x64xi8, #dot_a>>
      // ALL-LAYOUT-COUNT-63: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 8, tile_width = 8, tile_height = 8, v_blocks = 1, cache_control = Default}

      tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 1}>
module attributes {ttig.support_sg_2d_block,  "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @dot_b_layout
  tt.func public @dot_b_layout(%arg0: !tt.ptr<i8>, %col_stride: i64) {
      %cst = arith.constant dense<0> : tensor<256x64xi8, #dot_b>
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<256x64xi8, #dot_b>>
      // ALL-LAYOUT:           %[[OFF_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[OFF_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[BASE_PTR:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

      // ALL-LAYOUT:           %[[HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32

      // ALL-LAYOUT:           %[[OFFSET:.*]] = llvm.add %[[OFF_0]], {{.*}} : i32
      // ALL-LAYOUT:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
      // ALL-LAYOUT:           %[[OFFSET_X:.*]] = llvm.mlir.constant(0 : i32) : i32
      // ALL-LAYOUT:           %[[OFFSET_Y:.*]] = llvm.select {{.*}}, %[[OFFSET]], %[[HEIGHT]] : i1, i32
      // ALL-LAYOUT:           llvm.mlir.undef : vector<8xi8>
      // ALL-LAYOUT-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xi8>
      // ALL-LAYOUT: triton_gen.2Dblockstore {{.*}}, %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 8, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
      tt.store %0, %cst {ttig.block_io = "row_major", boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<256x64xi8, #dot_b>>
      // ALL-LAYOUT-COUNT-63: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 8, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

      tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 8, 2], order = [2, 1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {ttig.support_sg_2d_block,  "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @slice_layout
  tt.func public @slice_layout(%arg0: !tt.ptr<i8>, %col_stride: i64) {
      %cst = arith.constant dense<0> : tensor<256x64xi8, #slice>
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<256x64xi8, #slice>>
      // ALL-LAYOUT:           %[[OFF_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[OFF_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[BASE_PTR:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

      // ALL-LAYOUT:           %[[HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32

      // ALL-LAYOUT:           %[[OFFSET:.*]] = llvm.add %[[OFF_0]], {{.*}} : i32
      // ALL-LAYOUT:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
      // ALL-LAYOUT:           %[[OFFSET_X:.*]] = llvm.mlir.constant(0 : i32) : i32
      // ALL-LAYOUT:           %[[OFFSET_Y:.*]] = llvm.select {{.*}}, %[[OFFSET]], %[[HEIGHT]] : i1, i32
      // ALL-LAYOUT:           llvm.mlir.undef : vector<16xi8>
      // ALL-LAYOUT-COUNT-16:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<16xi8>
      // ALL-LAYOUT: triton_gen.2Dblockstore {{.*}}, %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}
      tt.store %0, %cst {ttig.block_io = "row_major", boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<256x64xi8, #slice>>
      // ALL-LAYOUT-COUNT-31: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}

      tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 2], order = [1, 0]}>
module attributes {ttig.support_sg_2d_block,  "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @block_layout
  tt.func public @block_layout(%arg0: !tt.ptr<i8>, %col_stride: i64) {
      %cst = arith.constant dense<0> : tensor<256x64xi8, #blocked>
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<256x64xi8, #blocked>>
      // ALL-LAYOUT:           %[[OFF_0:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[OFF_1:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
      // ALL-LAYOUT:           %[[BASE_PTR:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

      // ALL-LAYOUT:           %[[HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32

      // ALL-LAYOUT:           %[[OFFSET:.*]] = llvm.add %[[OFF_0]], {{.*}} : i32
      // ALL-LAYOUT:           %[[BASE:.*]] = llvm.getelementptr %[[BASE_PTR]]{{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
      // ALL-LAYOUT:           %[[OFFSET_X:.*]] = llvm.mlir.constant(0 : i32) : i32
      // ALL-LAYOUT:           %[[OFFSET_Y:.*]] = llvm.select {{.*}}, %[[OFFSET]], %[[HEIGHT]] : i1, i32
      // ALL-LAYOUT:           llvm.mlir.undef : vector<8xi8>
      // ALL-LAYOUT-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xi8>
      // ALL-LAYOUT: triton_gen.2Dblockstore {{.*}}, %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 4, v_blocks = 1, cache_control = Default}
      tt.store %0, %cst {ttig.block_io = "row_major", boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<256x64xi8, #blocked>>
      // ALL-LAYOUT-COUNT-7: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 4, v_blocks = 1, cache_control = Default}

      tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_no_scf_with_advance_kernel(%base: !tt.ptr<f16>, %width: i64, %height: i64, %rowStride: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.make_tensor_ptr %base, [%width, %height], [%rowStride, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>
    // CHECK: %[[WARP_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK: %[[offsetBaseY:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[offsetBaseX:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[baseHeight:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[baseWidth:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[rowStride:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[colStride:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[base:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(f16, f16, {{.*}})>
    // COM: Skip the register, lane, warp and block to the offset computation which should be covered by the LL tests.
    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY]], {{.*}} : i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY]], {{.*}} : i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY]], {{.*}} : i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseY]], {{.*}} : i32
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: triton_gen.2Dblockstore {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[OFFSET_X]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %0, %cst {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @no_boundary_check(%base: !tt.ptr<f16>, %width: i64, %height: i64, %rowStride: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.make_tensor_ptr %base, [%width, %height], [%rowStride, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>

    // CHECK: %[[WARP_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32

    // CHECK: %[[offsetBaseY:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[offsetBaseX:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[baseHeight:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[baseWidth:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[rowStride:.*]] = llvm.extractvalue {{.*}}[4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[colStride:.*]] = llvm.extractvalue {{.*}}[5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK: %[[base:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

    // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[rowStride_i32:.*]] = llvm.trunc %[[rowStride]] : i64 to i32
    // CHECK: %[[PITCH:.*]] = llvm.mul %[[rowStride_i32]], %[[C2]]
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(f16, f16, {{.*}})>

    // COM: Skip the register, lane, warp and block to the offset computation which should be covered by the LL tests.
    // CHECK: %[[OFFSET_X:.*]] = llvm.add %[[offsetBaseX]], {{.*}} : i32
    // CHECK: %[[OFFSET_Y:.*]] = llvm.add %[[offsetBaseY]], {{.*}} : i32

    // COM: When boundary check is absent:
    // CHECK: %[[baseWidth:.*]] = llvm.mlir.constant(64 : i32)
    // CHECK: %[[base1:.*]] = llvm.getelementptr %[[base]][%[[OFFSET_X]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f16
    // CHECK: %[[OFFSET_X:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[baseHeight:.*]] = llvm.mlir.constant(8 : i32)
    // CHECK: %[[OFF:.*]] = llvm.mul %[[OFFSET_Y]], %[[PITCH]] : i32
    // CHECK: %[[base:.*]] = llvm.getelementptr %[[base1]][%[[OFF]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    // CHECK: %[[OFFSET_Y:.*]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-7: llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: %[[VAL0:.*]] = llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK: %[[VAL:.*]] = llvm.bitcast %[[VAL0]] : vector<8xf16> to vector<8xi16>

    // CHECK: triton_gen.2Dblockstore %[[base]], %[[baseWidth]], %[[baseHeight]], %[[PITCH]], %[[OFFSET_X]], %[[OFFSET_Y]], %[[VAL]] {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}

    tt.store %0, %cst {ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dpas_layout_2d_store_rep_cluster_4_2(
// CHECK-SAME:      %[[base:.*]]: !llvm.ptr<1>, %[[width:.*]]: i64, %[[height:.*]]: i64, %[[rowStride:.*]]: i64, %[[PTR_1:.*]]: !llvm.ptr<1>,
// CHECK-SAME:      %[[PTR_2:.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 16, 1, 1>} {
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
    // CHECK:           %[[OFF_0:.*]] = llvm.extractvalue %[[BLOCK_PTR]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFF_1:.*]] = llvm.extractvalue %[[BLOCK_PTR]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_PTR]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE_PTR:.*]] = llvm.extractvalue %[[BLOCK_PTR]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SCALAR_BYTES:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[WIDTH:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ROW_STRIDE:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_IN_BYTES:.*]] = llvm.mul %[[WIDTH]], %[[SCALAR_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE]], %[[SCALAR_BYTES]] : i32
    %13 = tt.make_tensor_ptr %base, [%width, %height], [%rowStride, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dpas>>

    // COM: The shape of DPAS layout replica is [4, 2]
    // COM: The replica order are [0, 1]
    // COM:                       [2, 3]
    // COM:                       [4, 5]
    // COM:                       [6, 7]

    // COM: replica [0, 0]
    // CHECK:          llvm.call spir_funccc @_Z12get_local_idj
    // CHECK-COUNT-4:   %[[VAL_164:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_168:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_169:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_170:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_171:.*]] = llvm.shl {{.*}}, %[[VAL_170]] : i32
    // CHECK:           %[[VAL_172:.*]] = llvm.or %[[VAL_169]], %[[VAL_171]] : i32
    // CHECK:           %[[VAL_173:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_174:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_175:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_176:.*]] = llvm.or disjoint %[[VAL_174]], %[[VAL_175]] : i32
    // CHECK:           %[[VAL_177:.*]] = llvm.xor %[[VAL_168]], %[[VAL_176]] : i32
    // CHECK:           %[[VAL_178:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_179:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_180:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_181:.*]] = llvm.or disjoint %[[VAL_179]], %[[VAL_180]] : i32
    // CHECK:           %[[VAL_182:.*]] = llvm.xor %[[VAL_168]], %[[VAL_181]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_182]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_177]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [0, 1]
    // CHECK:           llvm.mlir.constant(8 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_208:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_209:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_210:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_211:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_212:.*]] = llvm.shl {{.*}}, %[[VAL_211]] : i32
    // CHECK:           %[[VAL_213:.*]] = llvm.or %[[VAL_210]], %[[VAL_212]] : i32
    // CHECK:           %[[VAL_214:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_215:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_216:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.or disjoint %[[VAL_215]], %[[VAL_216]] : i32
    // CHECK:           %[[VAL_218:.*]] = llvm.xor %[[VAL_208]], %[[VAL_217]] : i32
    // CHECK:           %[[VAL_219:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_220:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_221:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_222:.*]] = llvm.or disjoint %[[VAL_220]], %[[VAL_221]] : i32
    // CHECK:           %[[VAL_223:.*]] = llvm.xor %[[VAL_209]], %[[VAL_222]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_223]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_218]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [1, 0]
    // CHECK:           llvm.mlir.constant(16 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_249:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_250:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_251:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_252:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_253:.*]] = llvm.shl {{.*}}, %[[VAL_252]] : i32
    // CHECK:           %[[VAL_254:.*]] = llvm.or %[[VAL_251]], %[[VAL_253]] : i32
    // CHECK:           %[[VAL_255:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_256:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_257:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_258:.*]] = llvm.or disjoint %[[VAL_256]], %[[VAL_257]] : i32
    // CHECK:           %[[VAL_259:.*]] = llvm.xor %[[VAL_250]], %[[VAL_258]] : i32
    // CHECK:           %[[VAL_260:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_261:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_262:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_263:.*]] = llvm.or disjoint %[[VAL_261]], %[[VAL_262]] : i32
    // CHECK:           %[[VAL_264:.*]] = llvm.xor %[[VAL_249]], %[[VAL_263]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_264]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_259]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [1, 1]
    // CHECK:           llvm.mlir.constant(24 : i32) : i32
    // CHECK-COUNT-3:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_291:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_292:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_293:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_294:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_295:.*]] = llvm.shl {{.*}}, %[[VAL_294]] : i32
    // CHECK:           %[[VAL_296:.*]] = llvm.or %[[VAL_293]], %[[VAL_295]] : i32
    // CHECK:           %[[VAL_297:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_298:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_299:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_300:.*]] = llvm.or disjoint %[[VAL_298]], %[[VAL_299]] : i32
    // CHECK:           %[[VAL_301:.*]] = llvm.xor %[[VAL_291]], %[[VAL_300]] : i32
    // CHECK:           %[[VAL_302:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_303:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_304:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_305:.*]] = llvm.or disjoint %[[VAL_303]], %[[VAL_304]] : i32
    // CHECK:           %[[VAL_306:.*]] = llvm.xor %[[VAL_292]], %[[VAL_305]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_306]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_301]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [2, 0]
    // CHECK:           llvm.mlir.constant(32 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_332:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_333:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_334:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_335:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_336:.*]] = llvm.shl {{.*}}, %[[VAL_335]] : i32
    // CHECK:           %[[VAL_337:.*]] = llvm.or %[[VAL_334]], %[[VAL_336]] : i32
    // CHECK:           %[[VAL_338:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_339:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_340:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_341:.*]] = llvm.or disjoint %[[VAL_339]], %[[VAL_340]] : i32
    // CHECK:           %[[VAL_342:.*]] = llvm.xor %[[VAL_333]], %[[VAL_341]] : i32
    // CHECK:           %[[VAL_343:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_344:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_345:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_346:.*]] = llvm.or disjoint %[[VAL_344]], %[[VAL_345]] : i32
    // CHECK:           %[[VAL_347:.*]] = llvm.xor %[[VAL_332]], %[[VAL_346]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_347]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_342]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [2, 1]
    // CHECK:           llvm.mlir.constant(40 : i32) : i32
    // CHECK-COUNT-3:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_374:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_375:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_376:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_377:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_378:.*]] = llvm.shl {{.*}}, %[[VAL_377]] : i32
    // CHECK:           %[[VAL_379:.*]] = llvm.or %[[VAL_376]], %[[VAL_378]] : i32
    // CHECK:           %[[VAL_380:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_381:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_382:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_383:.*]] = llvm.or disjoint %[[VAL_381]], %[[VAL_382]] : i32
    // CHECK:           %[[VAL_384:.*]] = llvm.xor %[[VAL_374]], %[[VAL_383]] : i32
    // CHECK:           %[[VAL_385:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_386:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_387:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_388:.*]] = llvm.or disjoint %[[VAL_386]], %[[VAL_387]] : i32
    // CHECK:           %[[VAL_389:.*]] = llvm.xor %[[VAL_375]], %[[VAL_388]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_389]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_384]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [3, 0]
    // CHECK:           llvm.mlir.constant(48 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_415:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_416:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_417:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_418:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_419:.*]] = llvm.shl {{.*}}, %[[VAL_418]] : i32
    // CHECK:           %[[VAL_420:.*]] = llvm.or %[[VAL_417]], %[[VAL_419]] : i32
    // CHECK:           %[[VAL_421:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_422:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_423:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_424:.*]] = llvm.or disjoint %[[VAL_422]], %[[VAL_423]] : i32
    // CHECK:           %[[VAL_425:.*]] = llvm.xor %[[VAL_416]], %[[VAL_424]] : i32
    // CHECK:           %[[VAL_426:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_427:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_428:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_429:.*]] = llvm.or disjoint %[[VAL_427]], %[[VAL_428]] : i32
    // CHECK:           %[[VAL_430:.*]] = llvm.xor %[[VAL_415]], %[[VAL_429]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_430]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_425]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [3, 1]
    // CHECK:           llvm.mlir.constant(56 : i32) : i32
    // CHECK-COUNT-3:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_457:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_458:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_459:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_460:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_461:.*]] = llvm.shl {{.*}}, %[[VAL_460]] : i32
    // CHECK:           %[[VAL_462:.*]] = llvm.or %[[VAL_459]], %[[VAL_461]] : i32
    // CHECK:           %[[VAL_463:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_464:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_465:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_466:.*]] = llvm.or disjoint %[[VAL_464]], %[[VAL_465]] : i32
    // CHECK:           %[[VAL_467:.*]] = llvm.xor %[[VAL_457]], %[[VAL_466]] : i32
    // CHECK:           %[[VAL_468:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_469:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_470:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_471:.*]] = llvm.or disjoint %[[VAL_469]], %[[VAL_470]] : i32
    // CHECK:           %[[VAL_472:.*]] = llvm.xor %[[VAL_458]], %[[VAL_471]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_472]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_467]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    tt.store %13, %cst {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dpas>>
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
      // CHECK: %[[THREAD_ID:.*]] = llvm.call spir_funccc @_Z12get_local_idj
      // CHECK: %[[THREAD_ID_32:.*]] = llvm.trunc %[[THREAD_ID]] : i64 to i32
      // CHECK-DAG: %[[CST_127:.*]] = llvm.mlir.constant(127 : i32) : i32
      // CHECK-DAG: %[[RTID:.*]] = llvm.and %[[THREAD_ID_32:.*]], %[[CST_127]] : i32
      // CHECK-DAG: %[[VAL_584:.*]] = llvm.mlir.constant(16 : i32) : i32
      // CHECK: %[[VAL_586:.*]] = llvm.udiv %[[RTID]], %[[VAL_584]] : i32
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
