// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 67584 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @convert_dpas(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                          %[[SCRATCH_SLM:.*]]: !llvm.ptr<3>) attributes {noinline = false, triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
  tt.func public @convert_dpas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #mma>

    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_264:.*]] = llvm.mlir.constant(264 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // COM: The following operations is generated for the conversion of DPAS layout to blocked layout.  The conversion replica size is 128*256. So there is 1 round of load/store with synchronization.
    // CHECK:           %[[threadId_64:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]]) {function_type = !llvm.func<i64 (i32)>, linkage = #llvm.linkage<external>, passthrough = ["nounwind", "willreturn", ["memory", "0"]], sym_name = "_Z12get_local_idj", visibility_ = 0 : i64} : (i32) -> i64
    // CHECK:           %[[threadId:.*]] = llvm.trunc %[[threadId_64]] : i64 to i32
    // CHECK:           %[[warpId:.*]] = llvm.udiv %[[threadId]], %[[CST_16]]  : i32
    // CHECK:           %[[laneId:.*]] = llvm.urem %[[threadId]], %[[CST_16]]  : i32
    // CHECK:           %[[WARP_ID_N:.*]] = llvm.urem %[[warpId]], %[[CST_8]]  : i32
    // CHECK:           %[[WARP_ID_M_0:.*]] = llvm.udiv %[[warpId]], %[[CST_8]]  : i32
    // CHECK:           %[[WARP_ID_M_1:.*]] = llvm.urem %[[WARP_ID_M_0]], %[[CST_4]]  : i32
    // CHECK:           %[[rowWarpId:.*]] = llvm.urem %[[WARP_ID_M_1]], %[[CST_4]]  : i32
    // CHECK:           %[[colWarpId:.*]] = llvm.urem %[[WARP_ID_N]], %[[CST_8]]  : i32
    // CHECK:           %[[rowWarpOffset:.*]] = llvm.mul %[[rowWarpId]], %[[CST_32]] : i32
    // CHECK:           %[[colWarpOffset:.*]] = llvm.mul %[[colWarpId]], %[[CST_32]] : i32
    // CHECK:           %[[LANE_ID_M:.*]] = llvm.udiv %[[laneId]], %[[CST_16]]  : i32
    // CHECK:           %[[multiDimBase_0:.*]] = llvm.add %[[LANE_ID_M]], %[[rowWarpOffset]] : i32
    // CHECK:           %[[LANE_ID_N:.*]] = llvm.urem %[[laneId]], %[[CST_16]]  : i32
    // CHECK:           %[[multiDimBase_1:.*]] = llvm.add %[[LANE_ID_N]], %[[colWarpOffset]] : i32
    // CHECK:           %[[multiDimOffset_0:.*]] = llvm.add %[[multiDimBase_0]], %[[CST_0]] : i32
    // CHECK:           %[[multiDimOffset_1:.*]] = llvm.add %[[multiDimBase_1]], %[[CST_0]] : i32

    // COM: The size 264 is calculated based on the size of the DPAS layout on dim 1 plus the padded size with 8.
    // CHECK:           %[[VAL_63:.*]] = llvm.mul %[[multiDimOffset_0]], %[[CST_264]] : i32
    // CHECK:           %[[offset:.*]] = llvm.add %[[VAL_63]], %[[multiDimOffset_1]] : i32
    // CHECK:           %[[VAL_65:.*]] = llvm.getelementptr %[[SCRATCH_SLM]]{{\[}}%[[offset]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:           %[[VAL_66:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[CST_0]] : i32] : vector<1xf16>

    // COM: Because the values per thread of DPAS layout is not contiguous. The values are stored in the SLM in a non-vectorized way.
    // COM: Total 64 stores are generated to save the tensor of the DPAS layout to the SLM. 128*256/(4*8*16) = 64
    // CHECK:           llvm.store %[[VAL_66]], %[[VAL_65]] : vector<1xf16>, !llvm.ptr<3>
    // CHECK-COUNT-63:  llvm.store {{.*}}, {{.*}} : vector<1xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {function_type = !llvm.func<void (i32)>, linkage = #llvm.linkage<external>, passthrough = ["convergent"], sym_name = "_Z7barrierj", visibility_ = 0 : i64} : (i32) -> ()

    // COM: Because the values per thread of blocked layout is contiguous. The values are loaded from the SLM in a vectorized way.
    // COM: Total 8 loads are generated to load the tensor of the blocked layout from the SLM. 128*256/(16*2*16*8) = 8
    // CHECK-COUNT-8:    {{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>

    %93 = triton_gpu.convert_layout %cst {allocation.offset = 0 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %80 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %83 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %83, %93 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}


// -----


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [2, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 67584 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @convert_dpas(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                          %[[SCRATCH_SLM:.*]]: !llvm.ptr<3>) attributes {noinline = false, triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
  tt.func public @convert_dpas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #mma>

    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[CST_264:.*]] = llvm.mlir.constant(264 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32

    // COM: The following operations is generated for the conversion of DPAS layout to blocked layout. The conversion replica size is 64*256. So there are 2 round of load/store with synchronization.
    // CHECK:           %[[threadId_64:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]]) {function_type = !llvm.func<i64 (i32)>, linkage = #llvm.linkage<external>, passthrough = ["nounwind", "willreturn", ["memory", "0"]], sym_name = "_Z12get_local_idj", visibility_ = 0 : i64} : (i32) -> i64
    // CHECK:           %[[threadId:.*]] = llvm.trunc %[[threadId_64]] : i64 to i32
    // CHECK:           %[[warpId:.*]] = llvm.udiv %[[threadId]], %[[CST_16]]  : i32
    // CHECK:           %[[laneId:.*]] = llvm.urem %[[threadId]], %[[CST_16]]  : i32
    // CHECK:           %[[WARP_ID_N:.*]] = llvm.urem %[[warpId]], %[[CST_8]]  : i32
    // CHECK:           %[[WARP_ID_M_0:.*]] = llvm.udiv %[[warpId]], %[[CST_8]]  : i32
    // CHECK:           %[[WARP_ID_M_1:.*]] = llvm.urem %[[WARP_ID_M_0]], %[[CST_4]]  : i32
    // CHECK:           %[[rowWarpId:.*]] = llvm.urem %[[WARP_ID_M_1]], %[[CST_8]]  : i32
    // CHECK:           %[[colWarpId:.*]] = llvm.urem %[[WARP_ID_N]], %[[CST_8]]  : i32
    // CHECK:           %[[rowWarpOffset:.*]] = llvm.mul %[[rowWarpId]], %[[CST_16]] : i32
    // CHECK:           %[[colWarpOffset:.*]] = llvm.mul %[[colWarpId]], %[[CST_32]] : i32
    // CHECK:           %[[LANE_ID_M:.*]] = llvm.udiv %[[laneId]], %[[CST_16]]  : i32
    // CHECK:           %[[multiDimBase_0:.*]] = llvm.add %[[LANE_ID_M]], %[[rowWarpOffset]] : i32
    // CHECK:           %[[LANE_ID_N:.*]] = llvm.urem %[[laneId]], %[[CST_16]]  : i32
    // CHECK:           %[[multiDimBase_1:.*]] = llvm.add %[[LANE_ID_N]], %[[colWarpOffset]] : i32
    // CHECK:           %[[multiDimOffset_0:.*]] = llvm.add %[[multiDimBase_0]], %[[CST_0]] : i32
    // CHECK:           %[[multiDimOffset_1:.*]] = llvm.add %[[multiDimBase_1]], %[[CST_0]] : i32

    // COM: The size 264 is calculated based on the size of the DPAS layout on dim 1 plus the padded size with 8.
    // CHECK:           %[[VAL_63:.*]] = llvm.mul %[[multiDimOffset_0]], %[[CST_264]] : i32
    // CHECK:           %[[offset:.*]] = llvm.add %[[VAL_63]], %[[multiDimOffset_1]] : i32
    // CHECK:           %[[VAL_65:.*]] = llvm.getelementptr %[[SCRATCH_SLM]]{{\[}}%[[offset]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:           %[[VAL_66:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[CST_0]] : i32] : vector<1xf16>

    // COM: Because the values per thread of DPAS layout is not contiguous. The values are stored in the SLM in a non-vectorized way.
    // COM: Total 32 stores are generated to save the tensor of the DPAS layout to the SLM. 64*256/(4*8*16) = 32
    // CHECK:           llvm.store %[[VAL_66]], %[[VAL_65]] : vector<1xf16>, !llvm.ptr<3>
    // CHECK-COUNT-31:  llvm.store {{.*}}, {{.*}} : vector<1xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {function_type = !llvm.func<void (i32)>, linkage = #llvm.linkage<external>, passthrough = ["convergent"], sym_name = "_Z7barrierj", visibility_ = 0 : i64} : (i32) -> ()

    // COM: Because the values per thread of blocked layout is contiguous. The values are loaded from the SLM in a vectorized way.
    // COM: Total 4 loads are generated to load the tensor of the blocked layout from the SLM. 128*256/(16*2*16*8) = 8
    // CHECK-COUNT-4:    {{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>

    // COM: The 2nd round of exchanging values.
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {function_type = !llvm.func<void (i32)>, linkage = #llvm.linkage<external>, passthrough = ["convergent"], sym_name = "_Z7barrierj", visibility_ = 0 : i64} : (i32) -> ()
    // CHECK-COUNT-32:  llvm.store {{.*}}, {{.*}} : vector<1xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {function_type = !llvm.func<void (i32)>, linkage = #llvm.linkage<external>, passthrough = ["convergent"], sym_name = "_Z7barrierj", visibility_ = 0 : i64} : (i32) -> ()
    // CHECK-COUNT-4:    {{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>

    %93 = triton_gpu.convert_layout %cst {allocation.offset = 0 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %80 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %83 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %83, %93 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
