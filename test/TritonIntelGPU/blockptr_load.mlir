// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,LARGE-BLOCK-SIZE-TRANS-B
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm=one_matrix_per_load_for_bt=1 | FileCheck %s --implicit-check-not=llvm.inline_asm --check-prefixes=CHECK,SMALL-BLOCK-SIZE-TRANS-B

// CHECK-DAG: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-8: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi({{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %0 = ttg.convert_layout %D : tensor<64x64xf32, #dpas> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_add_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg8: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg5, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-8: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi({{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %ptrX = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf32, #dpas>>
    // CHECK-COUNT-4: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    %X = tt.load %ptrX {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x64xf32, #dpas>>
    // CHECK-COUNT-32: llvm.fadd {{.*}}, {{.*}}
    %0 = arith.addf %D, %X : tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_add_transpose_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg8: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg5, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1>>
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-8: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi({{.*}}) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x64xf16, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x64xf16, #dot1> -> tensor<64x64xf32, #dpas>
    %ptrX = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg8, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf32, #dpas>>
    // CHECK-NOT: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    %X = tt.load %ptrX {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x64xf32, #dpas>>
    %0 = arith.addf %D, %X : tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_no_scf_with_advance_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf32, #dot0>>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf32, #dot1>>
    // CHECK-COUNT-2: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-1: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK-COUNT-4: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS0_i({{.*}}) {{.*}} : (i32, vector<4xf32>, vector<8xf32>, vector<8xf32>, i32) -> vector<8xf32>
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf32, #dot0>>
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x64xf32, #dot1>>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<64x32xf32, #dot0> * tensor<32x64xf32, #dot1> -> tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_a_2d_load(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64) attributes {intel_reqd_sub_group_size = 16 : i32, triton_gen.max_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @dot_op_a_2d_load(%arg0: !tt.ptr<f16>, %arg2: i64, %arg4: i64, %arg5: i64, %arg7: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_7:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_7]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_8]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_10]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_11]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_12]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_13]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_18]] : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_22:.*]] = llvm.urem %[[VAL_20]], %[[VAL_21]] : i32
    // CHECK:           %[[VAL_23:.*]] = llvm.udiv %[[VAL_20]], %[[VAL_21]] : i32
    // CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[VAL_22]], %[[VAL_24]] : i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i32:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[HEIGHT_i32:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ELEM_SIZE_IN_BYTES:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_38]] : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.add %[[VAL_39]], %[[VAL_37]] : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX_:.*]] = llvm.add %[[VAL_41]], %[[OFFSET_1]] : i32
    // CHECK:           %[[offsetY_:.*]] = llvm.add %[[VAL_40]], %[[OFFSET_0]] : i32
    // CHECK:           %[[VAL_44:.*]] = llvm.trunc %[[offsetY_]] : i32 to i32
    // CHECK:           %[[VAL_45:.*]] = llvm.trunc %[[offsetX_]] : i32 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.mul %[[HEIGHT_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[VAL_48:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK:           %[[DEST:.*]] = llvm.alloca %[[VAL_48]] x i16 : (i32) -> !llvm.ptr
    // CHECK:           %[[VAL_51:.*]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<1> to i64
    // CHECK:           %[[VAL_50:.*]] = llvm.mlir.constant(63 : i64) : i64
    // CHECK:           %[[VAL_52:.*]] = llvm.and %[[VAL_51]], %[[VAL_50]] : i64
    // CHECK:           %[[VAL_53:.*]] = llvm.trunc %[[VAL_52]] : i64 to i32
    // CHECK:           %[[HEIGHT_ALIGNED:.*]] = llvm.add %[[HEIGHT]], %[[VAL_53]] : i32
    // CHECK:           %[[VAL_55:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_56:.*]] = llvm.udiv %[[VAL_53]], %[[VAL_55]] : i32
    // CHECK:           %[[VAL_57:.*]] = llvm.add %[[VAL_45]], %[[VAL_56]] : i32
    // CHECK:           %[[VAL_58:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_60:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK:           %[[VAL_61:.*]] = llvm.insertelement %[[VAL_57]], %[[VAL_60]]{{\[}}%[[VAL_59]] : i32] : vector<2xi32>
    // CHECK:           %[[VAL_62:.*]] = llvm.insertelement %[[VAL_44]], %[[VAL_61]]{{\[}}%[[VAL_58]] : i32] : vector<2xi32>
    %ptrA = tt.make_tensor_ptr %arg0, [%arg2, %arg4], [%arg5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot0>>
    // CHECK-DAG:       [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:       [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:       [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:       [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileHeight]], [[TileWidth]], [[VBlocks]], %[[BASE]], %[[HEIGHT_ALIGNED]], %[[WIDTH_i32]], %[[ROW_STRIDE_IN_BYTES]], %[[VAL_62]], %[[DEST]])
    %A = tt.load %ptrA {boundaryCheck = array<i32: 1>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dot0>>
    %B = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot1>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

// CHECK-DAG:         llvm.func spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @dot_op_b_2d_load(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64) attributes {intel_reqd_sub_group_size = 16 : i32, triton_gen.max_work_group_size = array<i32: 128, 1, 1>} {
  tt.func public @dot_op_b_2d_load(%arg1: !tt.ptr<f16>, %arg3: i64, %arg4: i64, %arg7: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:           %[[VAL_6:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_6]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_7]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_8]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_9]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_10]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_11]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BLOCK_POINTER:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_12]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[SUB_GROUP_ID_RAW:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK:           %[[SUB_GROUP_ID_EXT:.*]] = llvm.zext %[[SUB_GROUP_ID_RAW]] : i32 to i64
    // CHECK:           %[[SUB_GROUP_ID:.*]] = llvm.trunc %[[SUB_GROUP_ID_EXT]] : i64 to i32
    // CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_18:.*]] = llvm.urem %[[SUB_GROUP_ID]], %[[VAL_17]] : i32
    // CHECK:           %[[VAL_19:.*]] = llvm.udiv %[[SUB_GROUP_ID]], %[[VAL_17]] : i32
    // CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_21:.*]] = llvm.urem %[[VAL_19]], %[[VAL_20]] : i32
    // CHECK:           %[[VAL_22:.*]] = llvm.udiv %[[VAL_19]], %[[VAL_20]] : i32
    // CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[outerDimWarpId:.*]] = llvm.urem %[[VAL_18]], %[[VAL_23]] : i32
    // CHECK:           %[[OFFSET_0:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_1:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][4] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[COL_STRIDE_i64:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][5] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[BASE:.*]] = llvm.extractvalue %[[BLOCK_POINTER]][6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[ROW_STRIDE_i32:.*]] = llvm.trunc %[[ROW_STRIDE_i64]] : i64 to i32
    // CHECK:           %[[HEIGHT_i32:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ELEM_SIZE_IN_BYTES:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.mul %[[outerDimWarpId]], %[[VAL_37]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.add %[[VAL_38]], %[[VAL_36]] : i32
    // CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[offsetX_:.*]] = llvm.add %[[VAL_39]], %[[OFFSET_1]] : i32
    // CHECK:           %[[offsetY_:.*]] = llvm.add %[[VAL_40]], %[[OFFSET_0]] : i32
    // CHECK:           %[[VAL_43:.*]] = llvm.trunc %[[offsetY_]] : i32 to i32
    // CHECK:           %[[VAL_44:.*]] = llvm.trunc %[[offsetX_]] : i32 to i32
    // CHECK:           %[[ROW_STRIDE_IN_BYTES:.*]] = llvm.mul %[[ROW_STRIDE_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[HEIGHT:.*]] = llvm.mul %[[HEIGHT_i32]], %[[ELEM_SIZE_IN_BYTES]] : i32
    // CHECK:           %[[VAL_47:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK:           %[[DEST:.*]] = llvm.alloca %[[VAL_47]] x i32 : (i32) -> !llvm.ptr
    // CHECK:           %[[VAL_50:.*]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<1> to i64
    // CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(63 : i64) : i64
    // CHECK:           %[[VAL_51:.*]] = llvm.and %[[VAL_50]], %[[VAL_49]] : i64
    // CHECK:           %[[VAL_52:.*]] = llvm.trunc %[[VAL_51]] : i64 to i32
    // CHECK:           %[[HEIGHT_ALIGNED:.*]] = llvm.add %[[HEIGHT]], %[[VAL_52]] : i32
    // CHECK:           %[[VAL_54:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[VAL_55:.*]] = llvm.udiv %[[VAL_52]], %[[VAL_54]] : i32
    // CHECK:           %[[VAL_56:.*]] = llvm.add %[[VAL_44]], %[[VAL_55]] : i32
    // CHECK:           %[[VAL_57:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:           %[[VAL_58:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:           %[[VAL_59:.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK:           %[[VAL_60:.*]] = llvm.insertelement %[[VAL_56]], %[[VAL_59]]{{\[}}%[[VAL_58]] : i32] : vector<2xi32>
    // CHECK:           %[[VAL_61:.*]] = llvm.insertelement %[[VAL_43]], %[[VAL_60]]{{\[}}%[[VAL_57]] : i32] : vector<2xi32>
    %ptrB = tt.make_tensor_ptr %arg1, [%arg4, %arg3], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot1>>
    // CHECK-DAG:       [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:       [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:       [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:       [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileHeight]], [[TileWidth]], [[VBlocks]], %[[BASE]], %[[HEIGHT_ALIGNED]], %[[WIDTH_i32]], %[[ROW_STRIDE_IN_BYTES]], %[[VAL_61]], %[[DEST]])
    %B = tt.load %ptrB {boundaryCheck = array<i32: 0>, padding = 1 : i32, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x32xf16, #dot1>>
    %A = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dot0>
    %C = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %D = tt.dot %A, %B, %C, inputPrecision = tf32 : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @column_major_dot_b
  tt.func public @column_major_dot_b(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i32 = arith.constant 64 : i32
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c32_i64 = arith.constant 32 : i64
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      // COM: One DPAS operand B per load instruction.
        // SMALL-BLOCK-SIZE-TRANS-B-COUNT-8:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
      // COM: Two interleaved DPAS operand B per load instruction. Need to shuffle the loaded value to decompose the VNNI format DPAS operand B.
        // LARGE-BLOCK-SIZE-TRANS-B:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_68:.*]] = llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_69:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_68]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_71:.*]] = llvm.shufflevector %[[VAL_68]], %[[VAL_68]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_103:.*]] = llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_104:.*]] = llvm.shufflevector %[[VAL_103]], %[[VAL_103]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_106:.*]] = llvm.shufflevector %[[VAL_103]], %[[VAL_103]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_138:.*]] = llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_139:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_138]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_141:.*]] = llvm.shufflevector %[[VAL_138]], %[[VAL_138]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_173:.*]] = llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_174:.*]] = llvm.shufflevector %[[VAL_173]], %[[VAL_173]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi32>
        // LARGE-BLOCK-SIZE-TRANS-B:    %[[VAL_176:.*]] = llvm.shufflevector %[[VAL_173]], %[[VAL_173]] [1, 3, 5, 7, 9, 11, 13, 15] : vector<16xi32>
      %45 = tt.load %21 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      tt.return
  }
}

// -----

// CHECK:   llvm.func spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @column_major_dot_b
  tt.func public @column_major_dot_b(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      // CHECK:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      // CHECK:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv
      // CHECK:    llvm.shufflevector {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7] : vector<8xi32>
      %45 = tt.load %21 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @non_contiguous_load_dot_layout
  // COM: Check mask is not generated when boundary_check is not set.
  // CHECK-NOT: llvm.icmp "slt"
  tt.func public @non_contiguous_load_dot_layout(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #dot_b>>
      // CHECK-COUNT-64: llvm.load {{.*}} -> i16
      %1 = tt.load %0 : !tt.ptr<tensor<64x16xf16, #dot_b>>
      %2 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #dot_a>>
      // CHECK-COUNT-64: llvm.load {{.*}} -> i16
      %3 = tt.load %2 : !tt.ptr<tensor<64x16xf16, #dot_a>>
      tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @blocked_layout
  tt.func public @blocked_layout(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #blocked>>
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %45 = tt.load %21 {triton_intel_gpu.block_io = "column_major"} : !tt.ptr<tensor<64x16xf16, #blocked>>
      tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.func spir_kernelcc @boundary_check
  tt.func public @boundary_check(%arg0: !tt.ptr<f16>, %col_stride: i64) {
      %c64_i64 = arith.constant 64 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %21 = tt.make_tensor_ptr %arg0, [%c64_i64, %c64_i64], [%c1_i64, %col_stride], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x16xf16, #blocked>>
      // CHECK-NOT: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %45 = tt.load %21 : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-16: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %46 = tt.load %21 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-16: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %47 = tt.load %21 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x16xf16, #blocked>>

      // CHECK-COUNT-32: llvm.icmp "slt"
      // CHECK-COUNT-32: llvm.load {{.*}} -> i16
      %48 = tt.load %21 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x16xf16, #blocked>>
      tt.return
  }
}
