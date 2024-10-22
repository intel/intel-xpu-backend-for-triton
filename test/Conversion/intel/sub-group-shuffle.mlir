// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Basic 16x16 shuffle test

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#sliced = #triton_gpu.slice<{dim = 1, parent = #blocked}>
#sliced1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_f16(
  // CHECK-SAME:                                      %[[VAL_0:.*]]: !llvm.struct<(f16)>,
  // CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f16)>
  // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_4]])
  // CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_7]])
  // CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_10]])
  // CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_13]])
  // CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_16]])
  // CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_19]])
  // CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_22]])
  // CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_25]])
  // CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_28]])
  // CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(9 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_31]])
  // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(10 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_34]])
  // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(11 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_37]])
  // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(12 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_40]])
  // CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_43]])
  // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(14 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_46]])
  // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shuffleDhj(%[[VAL_2]], %[[VAL_49]])
  tt.func @test_f16(%arg0: tensor<16xf16, #sliced>) -> tensor<16xf16, #sliced1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16xf16, #sliced> -> tensor<16xf16, #sliced1>
    tt.return %0 : tensor<16xf16, #sliced1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_bf16(
  // CHECK-SAME:                                       %[[VAL_0:.*]]: !llvm.struct<(bf16)>,
  // CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(bf16)>
  // CHECK:           %[[VAL_2:.*]] = llvm.bitcast %[[VAL_1]] : bf16 to i16
  // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_4]])
  // CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_7]])
  // CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_10]])
  // CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_13]])
  // CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_16]])
  // CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_19]])
  // CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_22]])
  // CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_25]])
  // CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_28]])
  // CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(9 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_31]])
  // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(10 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_34]])
  // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(11 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_37]])
  // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(12 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_40]])
  // CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_43]])
  // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(14 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_46]])
  // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflesj(%[[VAL_2]], %[[VAL_49]])
  // CHECK-COUNT-16:  llvm.bitcast %{{.*}} : i16 to bf16
  tt.func @test_bf16(%arg0: tensor<16xbf16, #sliced>) -> tensor<16xbf16, #sliced1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16xbf16, #sliced> -> tensor<16xbf16, #sliced1>
    tt.return %0 : tensor<16xbf16, #sliced1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_i1(
  // CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.struct<(i1)>,
  // CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(i1)>
  // CHECK:           %[[VAL_2:.*]] = llvm.zext %[[VAL_1]] : i1 to i8
  // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_4]])
  // CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_7]])
  // CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_10]])
  // CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_13]])
  // CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_16]])
  // CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_19]])
  // CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_22]])
  // CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_25]])
  // CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_28]])
  // CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(9 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_31]])
  // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(10 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_34]])
  // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(11 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_37]])
  // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(12 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_40]])
  // CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_43]])
  // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(14 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_46]])
  // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[VAL_2]], %[[VAL_49]])
  // CHECK-COUNT-16:  llvm.trunc %{{.*}} : i8 to i1
  tt.func @test_i1(%arg0: tensor<16xi1, #sliced>) -> tensor<16xi1, #sliced1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16xi1, #sliced> -> tensor<16xi1, #sliced1>
    tt.return %0 : tensor<16xi1, #sliced1>
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @test_ptr(
  // CHECK-SAME:                                      %[[VAL_0:.*]]: !llvm.struct<(ptr<1>)>,
  // CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(ptr<1>)>
  // CHECK:           %[[VAL_2:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr<1> to i64
  // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_4]])
  // CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_7]])
  // CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_10]])
  // CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_13]])
  // CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_16]])
  // CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_19]])
  // CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_22]])
  // CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_25]])
  // CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_28]])
  // CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(9 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_31]])
  // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(10 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_34]])
  // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(11 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_37]])
  // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(12 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_40]])
  // CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_43]])
  // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(14 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_46]])
  // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflelj(%[[VAL_2]], %[[VAL_49]])
  // CHECK-COUNT-16:  llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
  tt.func @test_ptr(%arg0: tensor<16x!tt.ptr<f32>, #sliced>) -> tensor<16x!tt.ptr<f32>, #sliced1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x!tt.ptr<f32>, #sliced> -> tensor<16x!tt.ptr<f32>, #sliced1>
    tt.return %0 : tensor<16x!tt.ptr<f32>, #sliced1>
  }
}

// -----

// Sub-group size 32 variant.

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#sliced = #triton_gpu.slice<{dim = 1, parent = #blocked}>
#sliced1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.target = "xpu", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @test_f32(
  // CHECK-SAME:                                       %[[VAL_0:.*]]: !llvm.struct<(f32)>,
  // CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f32)>
  // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_4]])
  // CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_7]])
  // CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_10]])
  // CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_13]])
  // CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_16]])
  // CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(5 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_19]])
  // CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_22]])
  // CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_25]])
  // CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_28]])
  // CHECK:           %[[VAL_31:.*]] = llvm.mlir.constant(9 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_31]])
  // CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(10 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_34]])
  // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(11 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_37]])
  // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(12 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_40]])
  // CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_43]])
  // CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(14 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_46]])
  // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(15 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_49]])
  // CHECK:           %[[VAL_52:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_52]])
  // CHECK:           %[[VAL_55:.*]] = llvm.mlir.constant(17 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_55]])
  // CHECK:           %[[VAL_58:.*]] = llvm.mlir.constant(18 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_58]])
  // CHECK:           %[[VAL_61:.*]] = llvm.mlir.constant(19 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_61]])
  // CHECK:           %[[VAL_64:.*]] = llvm.mlir.constant(20 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_64]])
  // CHECK:           %[[VAL_67:.*]] = llvm.mlir.constant(21 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_67]])
  // CHECK:           %[[VAL_70:.*]] = llvm.mlir.constant(22 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_70]])
  // CHECK:           %[[VAL_73:.*]] = llvm.mlir.constant(23 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_73]])
  // CHECK:           %[[VAL_76:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_76]])
  // CHECK:           %[[VAL_79:.*]] = llvm.mlir.constant(25 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_79]])
  // CHECK:           %[[VAL_82:.*]] = llvm.mlir.constant(26 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_82]])
  // CHECK:           %[[VAL_85:.*]] = llvm.mlir.constant(27 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_85]])
  // CHECK:           %[[VAL_88:.*]] = llvm.mlir.constant(28 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_88]])
  // CHECK:           %[[VAL_91:.*]] = llvm.mlir.constant(29 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_91]])
  // CHECK:           %[[VAL_94:.*]] = llvm.mlir.constant(30 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_94]])
  // CHECK:           %[[VAL_97:.*]] = llvm.mlir.constant(31 : i32) : i32
  // CHECK:           llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_2]], %[[VAL_97]])
  tt.func @test_f32(%arg0: tensor<32xf32, #sliced>) -> tensor<32xf32, #sliced1> {
    %0 = triton_gpu.convert_layout %arg0 : tensor<32xf32, #sliced> -> tensor<32xf32, #sliced1>
    tt.return %0 : tensor<32xf32, #sliced1>
  }
}
