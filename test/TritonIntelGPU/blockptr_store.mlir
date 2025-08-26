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
    // CHECK-COUNT-5:   %[[VAL_185:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-COUNT-2:   %[[VAL_186:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_188:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_189:.*]] = llvm.xor %[[VAL_185]], %[[VAL_188]] : i32
    // CHECK:           %[[VAL_190:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_191:.*]] = llvm.xor %[[VAL_185]], %[[VAL_190]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_191]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_189]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [0, 1]
    // CHECK:           llvm.mlir.constant(8 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_210:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_211:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_212:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_213:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_215:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_216:.*]] = llvm.xor %[[VAL_210]], %[[VAL_215]] : i32
    // CHECK:           %[[VAL_217:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_218:.*]] = llvm.xor %[[VAL_211]], %[[VAL_217]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_218]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_216]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [1, 0]
    // CHECK:           llvm.mlir.constant(16 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_235:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_236:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_237:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_238:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_239:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_240:.*]] = llvm.xor %[[VAL_236]], %[[VAL_239]] : i32
    // CHECK:           %[[VAL_241:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_242:.*]] = llvm.xor %[[VAL_235]], %[[VAL_241]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_242]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_240]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [1, 1]
    // CHECK:           llvm.mlir.constant(24 : i32) : i32
    // CHECK-COUNT-3:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_261:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK:           %[[VAL_262:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_263:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_264:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_265:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_266:.*]] = llvm.xor %[[VAL_261]], %[[VAL_265]] : i32
    // CHECK:           %[[VAL_267:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_268:.*]] = llvm.xor %[[VAL_262]], %[[VAL_267]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_268]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_266]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [2, 0]
    // CHECK:           llvm.mlir.constant(32 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_286:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_287:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_288:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_289:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_290:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_291:.*]] = llvm.xor %[[VAL_287]], %[[VAL_290]] : i32
    // CHECK:           %[[VAL_292:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_293:.*]] = llvm.xor %[[VAL_286]], %[[VAL_292]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_293]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_291]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [2, 1]
    // CHECK:           llvm.mlir.constant(40 : i32) : i32
    // CHECK-COUNT-3:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_312:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_313:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_314:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_315:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_316:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_317:.*]] = llvm.xor %[[VAL_312]], %[[VAL_316]] : i32
    // CHECK:           %[[VAL_318:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_319:.*]] = llvm.xor %[[VAL_313]], %[[VAL_318]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_319]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_317]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [3, 0]
    // CHECK:           llvm.mlir.constant(48 : i32) : i32
    // CHECK-COUNT-2:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_337:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_338:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_339:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_340:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_341:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_342:.*]] = llvm.xor %[[VAL_338]], %[[VAL_341]] : i32
    // CHECK:           %[[VAL_343:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_344:.*]] = llvm.xor %[[VAL_337]], %[[VAL_343]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_344]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_342]] : i32
    // CHECK:           %[[NUM_PACKED_VALS:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[OFFSET_X:.*]] = llvm.udiv %[[ADD]], %[[NUM_PACKED_VALS]] : i32
    // CHECK:           llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8:   llvm.insertelement %{{[0-9]+}}, %{{[0-9]+}}{{\[}}{{.*}} : i32] : vector<8xf16>
    // CHECK:           triton_gen.2Dblockstore %[[BASE_PTR]], %[[WIDTH_IN_BYTES]], %[[HEIGHT]], %[[ROW_STRIDE_IN_BYTES]], %[[OFFSET_X]], %[[OFFSET_Y]], {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)

    // COM: replica [3, 1]
    // CHECK:           llvm.mlir.constant(56 : i32) : i32
    // CHECK-COUNT-3:   llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_363:.*]] = llvm.mlir.constant(24 : i32) : i32
    // CHECK:           %[[VAL_364:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK:           %[[VAL_365:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_366:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_367:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_368:.*]] = llvm.xor %[[VAL_363]], %[[VAL_367]] : i32
    // CHECK:           %[[VAL_369:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_370:.*]] = llvm.xor %[[VAL_364]], %[[VAL_369]] : i32
    // CHECK:           %[[ADD:.*]] = llvm.add %[[OFF_1]], %[[VAL_370]] : i32
    // CHECK:           %[[OFFSET_Y:.*]] = llvm.add %[[OFF_0]], %[[VAL_368]] : i32
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
