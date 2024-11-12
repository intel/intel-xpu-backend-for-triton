// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Basic sub-group unbroadcast.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_basic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  tt.func @test_basic(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16xf32, #blocked1> {
// CHECK:           %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_0]][4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_0]][5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_7:.*]] = llvm.extractvalue %[[VAL_0]][6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_8:.*]] = llvm.extractvalue %[[VAL_0]][7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_9:.*]] = llvm.extractvalue %[[VAL_0]][8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_10:.*]] = llvm.extractvalue %[[VAL_0]][9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_11:.*]] = llvm.extractvalue %[[VAL_0]][10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_12:.*]] = llvm.extractvalue %[[VAL_0]][11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_13:.*]] = llvm.extractvalue %[[VAL_0]][12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_0]][13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_15:.*]] = llvm.extractvalue %[[VAL_0]][14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_0]][15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK:           %[[VAL_17:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
// CHECK:           %[[VAL_18:.*]] = llvm.zext %[[VAL_17]] : i32 to i64
// CHECK:           %[[VAL_19:.*]] = llvm.trunc %[[VAL_18]] : i64 to i32
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_21:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_23:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_22]] : i32
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_26]] : i32
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_29:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_28]] : i32
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_31:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_30]] : i32
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_32]] : i32
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_35:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_34]] : i32
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_37:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_36]] : i32
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:           %[[VAL_39:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_38]] : i32
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_41:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_40]] : i32
// CHECK:           %[[VAL_42:.*]] = llvm.mlir.constant(11 : i32) : i32
// CHECK:           %[[VAL_43:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_42]] : i32
// CHECK:           %[[VAL_44:.*]] = llvm.mlir.constant(12 : i32) : i32
// CHECK:           %[[VAL_45:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_44]] : i32
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_47:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_46]] : i32
// CHECK:           %[[VAL_48:.*]] = llvm.mlir.constant(14 : i32) : i32
// CHECK:           %[[VAL_49:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_48]] : i32
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_51:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_50]] : i32
// CHECK:           %[[VAL_52:.*]] = llvm.mlir.poison : f32
// CHECK:           %[[VAL_53:.*]] = llvm.select %[[VAL_21]], %[[VAL_1]], %[[VAL_52]] : i1, f32
// CHECK:           %[[VAL_54:.*]] = llvm.select %[[VAL_23]], %[[VAL_2]], %[[VAL_53]] : i1, f32
// CHECK:           %[[VAL_55:.*]] = llvm.select %[[VAL_25]], %[[VAL_3]], %[[VAL_54]] : i1, f32
// CHECK:           %[[VAL_56:.*]] = llvm.select %[[VAL_27]], %[[VAL_4]], %[[VAL_55]] : i1, f32
// CHECK:           %[[VAL_57:.*]] = llvm.select %[[VAL_29]], %[[VAL_5]], %[[VAL_56]] : i1, f32
// CHECK:           %[[VAL_58:.*]] = llvm.select %[[VAL_31]], %[[VAL_6]], %[[VAL_57]] : i1, f32
// CHECK:           %[[VAL_59:.*]] = llvm.select %[[VAL_33]], %[[VAL_7]], %[[VAL_58]] : i1, f32
// CHECK:           %[[VAL_60:.*]] = llvm.select %[[VAL_35]], %[[VAL_8]], %[[VAL_59]] : i1, f32
// CHECK:           %[[VAL_61:.*]] = llvm.select %[[VAL_37]], %[[VAL_9]], %[[VAL_60]] : i1, f32
// CHECK:           %[[VAL_62:.*]] = llvm.select %[[VAL_39]], %[[VAL_10]], %[[VAL_61]] : i1, f32
// CHECK:           %[[VAL_63:.*]] = llvm.select %[[VAL_41]], %[[VAL_11]], %[[VAL_62]] : i1, f32
// CHECK:           %[[VAL_64:.*]] = llvm.select %[[VAL_43]], %[[VAL_12]], %[[VAL_63]] : i1, f32
// CHECK:           %[[VAL_65:.*]] = llvm.select %[[VAL_45]], %[[VAL_13]], %[[VAL_64]] : i1, f32
// CHECK:           %[[VAL_66:.*]] = llvm.select %[[VAL_47]], %[[VAL_14]], %[[VAL_65]] : i1, f32
// CHECK:           %[[VAL_67:.*]] = llvm.select %[[VAL_49]], %[[VAL_15]], %[[VAL_66]] : i1, f32
// CHECK:           %[[VAL_68:.*]] = llvm.select %[[VAL_51]], %[[VAL_16]], %[[VAL_67]] : i1, f32
// CHECK:           %[[VAL_69:.*]] = llvm.mlir.undef : !llvm.struct<(f32)>
// CHECK:           %[[VAL_70:.*]] = llvm.insertvalue %[[VAL_68]], %[[VAL_69]][0] : !llvm.struct<(f32)>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf32, #blocked1>
// CHECK:           llvm.return %[[VAL_70]] : !llvm.struct<(f32)>
    tt.return %0 : tensor<16xf32, #blocked1>
  }
}

// -----

// Sub-group unbroadcast with two elements per thread.

#blocked = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_two_els
  tt.func @test_two_els(%arg0: tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xf32, #blocked1> {
// CHECK:           %[[VAL_33:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
// CHECK:           %[[VAL_34:.*]] = llvm.zext %[[VAL_33]] : i32 to i64
// CHECK:           %[[VAL_35:.*]] = llvm.trunc %[[VAL_34]] : i64 to i32
// CHECK-COUNT-16:  llvm.icmp "eq" %[[VAL_35]], %{{.*}} : i32
// CHECK:           llvm.select %{{.*}}, %0, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %2, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %4, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %6, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %8, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %10, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %12, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %14, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %16, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %18, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %20, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %22, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %24, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %26, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %28, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %30, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %1, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %3, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %5, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %7, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %9, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %11, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %13, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %15, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %17, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %19, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %21, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %23, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %25, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %27, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %29, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %31, %{{.*}} : i1, f32
// COM: We return a struct with two values: go from 32 elements per thread to just 2.
// CHECK:           %[[VAL_101:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// CHECK:           %[[VAL_102:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_101]][0] : !llvm.struct<(f32, f32)>
// CHECK:           %[[VAL_103:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_102]][1] : !llvm.struct<(f32, f32)>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
// CHECK:           llvm.return %[[VAL_103]] : !llvm.struct<(f32, f32)>
    tt.return %0 : tensor<32xf32, #blocked1>
  }
}

// -----

// Sub-group unbroadcast with four elements per thread and 4 warps.

#blocked = #triton_gpu.blocked<{sizePerThread = [64, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_four_els_four_warps(
  tt.func @test_four_els_four_warps(%arg0: tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<256xf32, #blocked1> {
// CHECK:           %[[VAL_33:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
// CHECK:           %[[VAL_34:.*]] = llvm.zext %[[VAL_33]] : i32 to i64
// CHECK:           %[[VAL_35:.*]] = llvm.trunc %[[VAL_34]] : i64 to i32
// CHECK-COUNT-16:  llvm.icmp "eq" %[[VAL_35]], %{{.*}} : i32
// CHECK-COUNT-64:  llvm.select
// COM: We return a struct with four values: go from 64 elements per thread to just 4.
// CHECK:           %[[VAL_165:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_166:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_165]][0] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_167:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_166]][1] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_168:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_167]][2] : !llvm.struct<(f32, f32, f32, f32)>
// CHECK:           %[[VAL_169:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_168]][3] : !llvm.struct<(f32, f32, f32, f32)>
    %0 = triton_gpu.convert_layout %arg0 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<256xf32, #blocked1>
// CHECK:           llvm.return %[[VAL_169]] : !llvm.struct<(f32, f32, f32, f32)>
    tt.return %0 : tensor<256xf32, #blocked1>
  }
}

// -----

// Sub-group unbroadcast with two elements per thread, but repeated layout.

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_two_els_with_repeat(
  tt.func @test_two_els_with_repeat(%arg0: tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xf32, #blocked1> {
// CHECK:           %[[VAL_33:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return} : () -> i32
// CHECK:           %[[VAL_34:.*]] = llvm.zext %[[VAL_33]] : i32 to i64
// CHECK:           %[[VAL_35:.*]] = llvm.trunc %[[VAL_34]] : i64 to i32
// CHECK-COUNT-16:  llvm.icmp "eq" %[[VAL_35]], %{{.*}} : i32
// COM: Check select order differs from above as we have a single element per thread.
// CHECK:           llvm.select %{{.*}}, %0, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %1, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %2, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %3, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %4, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %5, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %6, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %7, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %8, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %9, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %10, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %11, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %12, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %13, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %14, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %15, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %1, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %2, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %3, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %4, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %5, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %6, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %7, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %8, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %9, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %10, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %11, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %12, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %13, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %14, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %15, %{{.*}} : i1, f32
// CHECK:           llvm.select %{{.*}}, %16, %{{.*}} : i1, f32
// COM: We return a struct with two values: go from 32 elements per thread to just 2.
// CHECK:           %[[VAL_165:.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// CHECK:           %[[VAL_166:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_165]][0] : !llvm.struct<(f32, f32)>
// CHECK:           %[[VAL_167:.*]] = llvm.insertvalue %{{.*}}, %[[VAL_166]][1] : !llvm.struct<(f32, f32)>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
// CHECK:           llvm.return %[[VAL_167]] : !llvm.struct<(f32, f32)>
    tt.return %0 : tensor<32xf32, #blocked1>
  }
}
