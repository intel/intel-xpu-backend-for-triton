// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s

// Basic 16x16 transpose test

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_f16(
  tt.func @test_f16(%arg0: tensor<16x16xf16, #blocked>) -> tensor<16x16xf16, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f16 to i16
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_usPU3AS3tt(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_usPU3AS3tt(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_usPU3AS3tt(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi16>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i16 to f16
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    tt.return %0 : tensor<16x16xf16, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_bf16(
  tt.func @test_bf16(%arg0: tensor<16x16xbf16, #blocked>) -> tensor<16x16xbf16, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : bf16 to i16
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_usPU3AS3tt(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_usPU3AS3tt(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_usPU3AS3tt(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i16
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi16>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i16 to bf16
    %0 = ttg.convert_layout %arg0 : tensor<16x16xbf16, #blocked> -> tensor<16x16xbf16, #blocked1>
    tt.return %0 : tensor<16x16xbf16, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_f32(
  tt.func @test_f32(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.return %0 : tensor<16x16xf32, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i8(
  tt.func @test_i8(%arg0: tensor<16x16xi8, #blocked>) -> tensor<16x16xi8, #blocked1> {
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_ucPU3AS3hh(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_ucPU3AS3hh(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_ucPU3AS3hh(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi8>
    %0 = ttg.convert_layout %arg0 : tensor<16x16xi8, #blocked> -> tensor<16x16xi8, #blocked1>
    tt.return %0 : tensor<16x16xi8, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i64(
  tt.func @test_i64(%arg0: tensor<16x16xi64, #blocked>) -> tensor<16x16xi64, #blocked1> {
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_ulPU3AS3mm(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_ulPU3AS3mm(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_ulPU3AS3mm(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi64>
    %0 = ttg.convert_layout %arg0 : tensor<16x16xi64, #blocked> -> tensor<16x16xi64, #blocked1>
    tt.return %0 : tensor<16x16xi64, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_ptr(
  tt.func @test_ptr(%arg0: tensor<16x16x!tt.ptr<f32>, #blocked>) -> tensor<16x16x!tt.ptr<f32>, #blocked1> {
    // CHECK-COUNT-16:  llvm.ptrtoint %{{.*}} : !llvm.ptr<1> to i64
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_ulPU3AS3mm(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_ulPU3AS3mm(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_ulPU3AS3mm(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi64>
    // CHECK-COUNT-16:  llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
    %0 = ttg.convert_layout %arg0 : tensor<16x16x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    tt.return %0 : tensor<16x16x!tt.ptr<f32>, #blocked1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i1(
  tt.func @test_i1(%arg0: tensor<16x16xi1, #blocked>) -> tensor<16x16xi1, #blocked1> {
    // CHECK-COUNT-16:  llvm.zext %{{.*}} : i1 to i8
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_ucPU3AS3hh(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_ucPU3AS3hh(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_ucPU3AS3hh(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi8>
    // CHECK-COUNT-16:  llvm.trunc %{{.*}} : i8 to i1
    %0 = ttg.convert_layout %arg0 : tensor<16x16xi1, #blocked> -> tensor<16x16xi1, #blocked1>
    tt.return %0 : tensor<16x16xi1, #blocked1>
  }
}

// -----

// Test with two sub-groups in the first dimension.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #blocked1>
    tt.return %0 : tensor<32x16xf32, #blocked1>
  }
}

// -----

// Test with two sub-groups in the second dimension.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<16x32xf32, #blocked>) -> tensor<16x32xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #blocked1>
    tt.return %0 : tensor<16x32xf32, #blocked1>
  }
}

// -----

// Test with four sub-groups in each dimension.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.return %0 : tensor<64x64xf32, #blocked1>
  }
}

// -----

// Test with four sub-groups in each dimension and an additional dimension.

#blocked = #ttg.blocked<{sizePerThread = [16, 1, 1], threadsPerWarp = [1, 16, 1], warpsPerCTA = [4, 4, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [4, 4, 1], order = [0, 1, 2]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<64x64x1xf32, #blocked>) -> tensor<64x64x1xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<64x64x1xf32, #blocked> -> tensor<64x64x1xf32, #blocked1>
    tt.return %0 : tensor<64x64x1xf32, #blocked1>
  }
}
// -----

// Test with four sub-groups in each dimension and sliced layout.

#blocked = #ttg.blocked<{sizePerThread = [16, 1, 1], threadsPerWarp = [1, 16, 1], warpsPerCTA = [4, 4, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<64x64xf32, #ttg.slice<{dim = 2, parent = #blocked}>>) -> tensor<64x64xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<64x64xf32, #blocked1>
    tt.return %0 : tensor<64x64xf32, #blocked1>
  }
}

// -----

// Test with one sub-group and double-sliced layout.

#blocked = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [1, 2, 3, 4, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<16x16x1xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<16x16x1xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<16x16x1xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<16x16x1xf32, #blocked1>
    tt.return %0 : tensor<16x16x1xf32, #blocked1>
  }
}

// -----

// Test with four sub-groups in each dimension and double-sliced layout.

#blocked = #ttg.blocked<{sizePerThread = [16, 1, 1, 1, 1], threadsPerWarp = [1, 16, 1, 1, 1], warpsPerCTA = [4, 1, 1, 4, 1], order = [1, 2, 3, 4, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16, 1], threadsPerWarp = [16, 1, 1], warpsPerCTA = [4, 1, 4], order = [1, 2, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<64x16x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>>) -> tensor<64x16x4xf32, #blocked1> {
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(272 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 15 more stores:
    // CHECK-COUNT-15:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<64x16x4xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 4, parent = #blocked}>}>> -> tensor<64x16x4xf32, #blocked1>
    tt.return %0 : tensor<64x16x4xf32, #blocked1>
  }
}

// -----

// Test transposition with 32 elements per work-item.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<32x16xf32, #blocked>) -> tensor<32x16xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // COM: Offset is double as before as we have double the rows.
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(544 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 31 more stores:
    // CHECK-COUNT-31:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #blocked1>
    tt.return %0 : tensor<32x16xf32, #blocked1>
  }
}

// -----

// Test transposition with 32 elements per work-item with a different layout.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<16x32xf32, #blocked>) -> tensor<16x32xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // COM: Offset is double as before as we have double the rows.
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(544 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 31 more stores:
    // CHECK-COUNT-31:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #blocked1>
    tt.return %0 : tensor<16x32xf32, #blocked1>
  }
}

// -----

// Test transposition with 32 elements per work-item and two warps in each dimension.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test(
  tt.func @test(%arg0: tensor<32x64xf32, #blocked>) -> tensor<32x64xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // COM: Offset is double as before as we have double the rows.
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(544 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 31 more stores:
    // CHECK-COUNT-31:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    tt.return %0 : tensor<32x64xf32, #blocked1>
  }
}

// -----

// Test no barriers are inserted when back to back transpositions are performed.

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @test_back_to_back
  // CHECK-NOT: barrier
  tt.func @test_back_to_back(%arg0: tensor<32x64xf32, #blocked>, %arg1: tensor<32x64xf32, #blocked>) -> (tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>) {
    %0 = ttg.convert_layout %arg0 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    tt.return %0, %1 : tensor<32x64xf32, #blocked1>, tensor<32x64xf32, #blocked1>
  }
}

// -----

// Test transposition with sub-group size 32.

#blocked = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_32(
  tt.func @test_32(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // COM: Offset changes with increased number of columns:
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(1056 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[33]
    // COM: Check there are 31 more stores:
    // CHECK-COUNT-31:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(33 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK:           %[[VAL_62:.*]] = llvm.getelementptr inbounds %[[VAL_61]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<16xi32>
    // CHECK:           llvm.load %[[VAL_62]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK:           %[[VAL_63:.*]] = llvm.getelementptr inbounds %[[VAL_62]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<16xi32>
    // CHECK:           %[[VAL_64:.*]] = llvm.getelementptr inbounds %[[VAL_63]][1024] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.return %0 : tensor<64x64xf32, #blocked1>
  }
}

// -----

// Test transposition with two contiguous rows.

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_2_cont(
  tt.func @test_2_cont(%arg0: tensor<64x64xf32, #blocked>) -> tensor<64x64xf32, #blocked1> {
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : f32 to i32
    // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[SMEM_0:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[SMEM_0]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK:           %[[VAL_36:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_37:.*]] = llvm.zext %[[VAL_36]] : i32 to i64
    // CHECK:           %[[VAL_38:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[VAL_39:.*]] = llvm.zext %[[VAL_38]] : i32 to i64
    // COM: Offset changes with increased number of columns:
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(544 : i64) : i64
    // CHECK:           %[[VAL_41:.*]] = llvm.mul %[[VAL_37]], %[[VAL_40]] : i64
    // CHECK:           %[[VAL_42:.*]] = llvm.getelementptr inbounds %[[VAL_35]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(%[[VAL_42]]
    // COM: Check offset:
    // CHECK:           llvm.getelementptr inbounds %{{.*}}[17]
    // COM: Check there are 31 more stores:
    // CHECK-COUNT-31:  llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK-NOT:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS3jj(
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(34 : i64) : i64
    // CHECK:           %[[VAL_60:.*]] = llvm.mul %[[VAL_39]], %[[VAL_59]] : i64
    // CHECK:           %[[VAL_61:.*]] = llvm.getelementptr inbounds %[[VAL_42]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_61]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK:           %[[VAL_62:.*]] = llvm.getelementptr inbounds %[[VAL_61]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, vector<16xi32>
    // CHECK:           %[[VAL_64:.*]] = llvm.getelementptr inbounds %[[VAL_62]][1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
    // CHECK:           llvm.load %[[VAL_64]] : !llvm.ptr<3> -> vector<16xi32>
    // CHECK-COUNT-32:  llvm.bitcast %{{.*}} : i32 to f32
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.return %0 : tensor<64x64xf32, #blocked1>
  }
}

// -----

// Test no barrier is introduced between transpositions with two contiguous rows.

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [1, 16], warpsPerCTA = [4, 2], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_2_cont_back_2_back(
  tt.func @test_2_cont_back_2_back(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xf32, #blocked>) -> (tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>) {
    // CHECK-NOT: barrier
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    tt.return %0, %1 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>
  }
}
