// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Basic 16x16 transpose test

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_f16(
  // CHECK-SAME:                                      , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_f16(%arg0: tensor<16x16xf16, #blocked>) -> tensor<16x16xf16, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f16 to i16
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.call spir_funccc @_Z32intel_sub_group_block_write_us16PU3AS3tDv16_t(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<16xi16>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi16>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i16 to f16
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    tt.return %0 : tensor<16x16xf16, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_bf16(
  // CHECK-SAME:                                     , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_bf16(%arg0: tensor<16x16xbf16, #blocked>) -> tensor<16x16xbf16, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : bf16 to i16
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.call spir_funccc @_Z32intel_sub_group_block_write_us16PU3AS3tDv16_t(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<16xi16>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi16>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i16 to bf16
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xbf16, #blocked> -> tensor<16x16xbf16, #blocked1>
    tt.return %0 : tensor<16x16xbf16, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_f32(
  // CHECK-SAME:                                    , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_f32(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_60:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_60]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.return %0 : tensor<16x16xf32, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i8(
  // CHECK-SAME:                                   , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_i8(%arg0: tensor<16x16xi8, #blocked>) -> tensor<16x16xi8, #blocked1> {
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.call spir_funccc @_Z32intel_sub_group_block_write_uc16PU3AS3hDv16_h(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<16xi8>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi8>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xi8, #blocked> -> tensor<16x16xi8, #blocked1>
    tt.return %0 : tensor<16x16xi8, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i64(
  // CHECK-SAME:                                    , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_i64(%arg0: tensor<16x16xi64, #blocked>) -> tensor<16x16xi64, #blocked1> {
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ul8PU3AS3mDv8_m(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi64>) -> ()
    // CHECK:           %[[VAL_60:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi64>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ul8PU3AS3mDv8_m(%[[VAL_60]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi64>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi64>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xi64, #blocked> -> tensor<16x16xi64, #blocked1>
    tt.return %0 : tensor<16x16xi64, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_ptr(
  // CHECK-SAME:                                    , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_ptr(%arg0: tensor<16x16x!tt.ptr<f32>, #blocked>) -> tensor<16x16x!tt.ptr<f32>, #blocked1> {
    // CHECK-COUNT-16:  llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ul8PU3AS3mDv8_m(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi64>) -> ()
    // CHECK:           %[[VAL_60:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi64>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ul8PU3AS3mDv8_m(%[[VAL_60]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi64>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi64>
    // CHECK-COUNT-16:  llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    tt.return %0 : tensor<16x16x!tt.ptr<f32>, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i1(
  // CHECK-SAME:                                   , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test_i1(%arg0: tensor<16x16xi1, #blocked>) -> tensor<16x16xi1, #blocked1> {
    // CHECK-COUNT-16:  llvm.zext %{{.*}} : i1 to i8
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.call spir_funccc @_Z32intel_sub_group_block_write_uc16PU3AS3hDv16_h(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<16xi8>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi8>
    // CHECK-COUNT-16:  llvm.trunc %{{.*}} : i8 to i1
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xi1, #blocked> -> tensor<16x16xi1, #blocked1>
    tt.return %0 : tensor<16x16xi1, #blocked1>
  }
}

// -----

// Test with two sub-groups in the first dimension.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #blocked1>
    tt.return %0 : tensor<32x16xf32, #blocked1>
  }
}

// -----

// Test with two sub-groups in the second dimension.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 2], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<16x32xf32, #blocked>) -> tensor<16x32xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #blocked1>
    tt.return %0 : tensor<16x32xf32, #blocked1>
  }
}

// -----

// Test with four sub-groups in each dimension.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 4], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 16 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.return %0 : tensor<64x64xf32, #blocked1>
  }
}

// -----

// Test with four sub-groups in each dimension and an additional dimension.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1, 1], threadsPerWarp = [1, 16, 1], warpsPerCTA = [4, 4, 1], order = [0, 1, 2]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [4, 4, 1], order = [0, 1, 2]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 16 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<64x64x1xf32, #blocked>) -> tensor<64x64x1xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<64x64x1xf32, #blocked> -> tensor<64x64x1xf32, #blocked1>
    tt.return %0 : tensor<64x64x1xf32, #blocked1>
  }
}
// -----

// Test with four sub-groups in each dimension and sliced layout.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1, 1], threadsPerWarp = [1, 16, 1], warpsPerCTA = [4, 4, 1], order = [0, 1, 2]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 4], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 16 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<64x64xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>>) -> tensor<64x64xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<64x64xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>> -> tensor<64x64xf32, #blocked1>
    tt.return %0 : tensor<64x64xf32, #blocked1>
  }
}

// -----

// Test with one sub-group and double-sliced layout.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [1, 2, 3, 4, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<16x16x1xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<16x16x1xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16x1xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #blocked}>}>> -> tensor<16x16x1xf32, #blocked1>
    tt.return %0 : tensor<16x16x1xf32, #blocked1>
  }
}

// -----

// Test with four sub-groups in each dimension and double-sliced layout.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [4, 1, 1, 4, 1], order = [1, 2, 3, 4, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [4, 1, 4], order = [1, 2, 0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 16 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>
  tt.func @test(%arg0: tensor<64x16x4xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<64x16x4xf32, #blocked1> {
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(
    %0 = triton_gpu.convert_layout %arg0 : tensor<64x16x4xf32, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 4, parent = #blocked}>}>> -> tensor<64x16x4xf32, #blocked1>
    tt.return %0 : tensor<64x16x4xf32, #blocked1>
  }
}

// -----

// Test transposition with 32 elements per work-item.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>)
  tt.func @test(%arg0: tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_60:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_60]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_60]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_61]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_62:.*]] = llvm.getelementptr inbounds %[[VAL_61]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_62]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK:           %[[VAL_78:.*]] = llvm.getelementptr inbounds %[[VAL_77]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #blocked1>
    tt.return %0 : tensor<32x16xf32, #blocked1>
  }
}

// -----

// Test transposition with 32 elements per work-item with a different layout.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>)
  tt.func @test(%arg0: tensor<16x32xf32, #blocked>) -> tensor<16x32xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_60:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_60]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_60]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_61]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_62:.*]] = llvm.getelementptr inbounds %[[VAL_61]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_62]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK:           %[[VAL_78:.*]] = llvm.getelementptr inbounds %[[VAL_77]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #blocked1>
    tt.return %0 : tensor<16x32xf32, #blocked1>
  }
}

// -----

// Test transposition with 32 elements per work-item and two warps in each dimension.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  // CHECK-SAME:                                , %[[VAL_1:.*]]: !llvm.ptr<3>)
  tt.func @test(%arg0: tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[BASE:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_54:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id()
    // CHECK:           %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i32 to i64
    // CHECK:           %[[VAL_56:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK:           %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i32 to i64
    // CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK-DAG:       %[[VAL_20:.*]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK:           %[[VAL_58:.*]] = llvm.mul %[[VAL_19]], %[[VAL_55]] : i64
    // CHECK:           %[[VAL_59:.*]] = llvm.getelementptr inbounds %[[BASE]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_59]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_60:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_60]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_60]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_61]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_62:.*]] = llvm.getelementptr inbounds %[[VAL_61]]{{\[}}16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<8xi32>
    // CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_62]]
    // CHECK-SAME:          (!llvm.ptr<3>, vector<8xi32>) -> ()
    // CHECK:           %[[VAL_76:.*]] = llvm.mul %[[VAL_20]], %[[VAL_57]] : i64
    // CHECK:           %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_59]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_77]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK:           %[[VAL_78:.*]] = llvm.getelementptr inbounds %[[VAL_77]][16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    tt.return %0 : tensor<32x64xf32, #blocked1>
  }
}

// -----

// Test no barriers are inserted when back to back transpositions are performed.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @test_back_to_back
  // CHECK-NOT: barrier
  tt.func @test_back_to_back(%arg0: tensor<32x64xf32, #blocked>, %arg1: tensor<32x64xf32, #blocked>) -> (tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    %1 = triton_gpu.convert_layout %arg1 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    tt.return %0, %1 : tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>
  }
}
