// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 67584 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @convert_dpas(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<1>)
// CHECK-SAME:                                          attributes {intel_reqd_sub_group_size = 16 : i32, noinline = false, reqd_work_group_size = array<i32: 512, 1, 1>} {
  tt.func public @convert_dpas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #mma>

    // CHECK-DAG:       %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:       %[[CST_16384:.*]] = llvm.mlir.constant(16384 : i32) : i32
    // CHECK-DAG:       %[[CST_8192:.*]] = llvm.mlir.constant(8192 : i32) : i32
    // CHECK-DAG:       %[[CST_128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:       %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:       %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:       %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:       %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:       %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:       %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:       %[[SMEM:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK-DAG:       %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:       %[[CST_511:.*]] = llvm.mlir.constant(511 : i32) : i32
    // CHECK-DAG:       %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // COM: The following operations is generated for the conversion of DPAS layout to blocked layout.  The conversion replica size is 128*256. So there is 1 round of load/store with synchronization.
    // CHECK:           %[[threadId_64:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]]) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return} : (i32) -> i64
    // CHECK:           %[[threadId:.*]] = llvm.trunc %[[threadId_64]] : i64 to i32
    // CHECK:           %[[rtid:.*]] = llvm.and %[[threadId:.*]], %[[CST_511]] : i32
    // CHECK:           %[[laneId:.*]] = llvm.urem %[[rtid]], %[[CST_16]]  : i32
    // CHECK:           %[[warpId:.*]] = llvm.udiv %[[rtid]], %[[CST_16]]  : i32
    // CHECK:           %[[VAL_25:.*]] = llvm.and %[[laneId]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_26:.*]] = llvm.icmp "eq" %[[VAL_25]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_27:.*]] = llvm.select %[[VAL_26]], %[[CST_0]], %[[CST_1]] : i1, i32
    // CHECK:           %[[VAL_28:.*]] = llvm.xor %[[CST_0]], %[[VAL_27]] : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.and %[[laneId]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_30:.*]] = llvm.icmp "eq" %[[VAL_29]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_31:.*]] = llvm.select %[[VAL_30]], %[[CST_0]], %[[CST_2]] : i1, i32
    // CHECK:           %[[VAL_32:.*]] = llvm.xor %[[VAL_28]], %[[VAL_31]] : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.and %[[laneId]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_34:.*]] = llvm.icmp "eq" %[[VAL_33]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.select %[[VAL_34]], %[[CST_0]], %[[CST_4]] : i1, i32
    // CHECK:           %[[VAL_36:.*]] = llvm.xor %[[VAL_32]], %[[VAL_35]] : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.and %[[laneId]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.icmp "eq" %[[VAL_37]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.select %[[VAL_38]], %[[CST_0]], %[[CST_8]] : i1, i32
    // CHECK:           %[[VAL_40:.*]] = llvm.xor %[[VAL_36]], %[[VAL_39]] : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.and %[[warpId]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_42:.*]] = llvm.icmp "eq" %[[VAL_41]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_43:.*]] = llvm.select %[[VAL_42]], %[[CST_0]], %[[CST_32]] : i1, i32
    // CHECK:           %[[VAL_44:.*]] = llvm.xor %[[VAL_40]], %[[VAL_43]] : i32
    // CHECK:           %[[VAL_45:.*]] = llvm.and %[[warpId]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_46:.*]] = llvm.icmp "eq" %[[VAL_45]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_47:.*]] = llvm.select %[[VAL_46]], %[[CST_0]], %[[CST_64]]  : i1, i32
    // CHECK:           %[[VAL_48:.*]] = llvm.xor %[[VAL_44]], %[[VAL_47]] : i32
    // CHECK:           %[[VAL_49:.*]] = llvm.and %[[warpId]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_50:.*]] = llvm.icmp "eq" %[[VAL_49]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_51:.*]] = llvm.select %[[VAL_50]], %[[CST_0]], %[[CST_128]] : i1, i32
    // CHECK:           %[[VAL_52:.*]] = llvm.xor %[[VAL_48]], %[[VAL_51]] : i32
    // CHECK:           %[[VAL_53:.*]] = llvm.and %[[warpId]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_54:.*]] = llvm.icmp "eq" %[[VAL_53]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_55:.*]] = llvm.select %[[VAL_54]], %[[CST_0]], %[[CST_8192]] : i1, i32
    // CHECK:           %[[VAL_56:.*]] = llvm.xor %[[VAL_52]], %[[VAL_55]] : i32
    // CHECK:           %[[VAL_57:.*]] = llvm.and %[[warpId]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_58:.*]] = llvm.icmp "eq" %[[VAL_57]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_59:.*]] = llvm.select %[[VAL_58]], %[[CST_0]], %[[CST_16384]] : i1, i32
    // CHECK:           %[[VAL_60:.*]] = llvm.xor %[[VAL_56]], %[[VAL_59]] : i32
    // CHECK:           %[[VAL_61:.*]] = llvm.xor %[[VAL_60]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_62:.*]] = llvm.lshr %[[VAL_61]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_63:.*]] = llvm.shl %[[VAL_62]], %[[CST_3]] : i32
    // CHECK:           %[[offset:.*]] = llvm.add %[[VAL_63]], %[[VAL_61]] : i32
    // CHECK:           %[[VAL_65:.*]] = llvm.getelementptr inbounds %[[SMEM]]{{\[}}%[[offset]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:           %[[VAL_66:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[CST_0]] : i32] : vector<1xf16>

    // COM: Because the values per thread of DPAS layout is not contiguous. The values are stored in the SLM in a non-vectorized way.
    // COM: Total 64 stores are generated to save the tensor of the DPAS layout to the SLM. 128*256/(4*8*16) = 64
    // CHECK:           llvm.store %[[VAL_66]], %[[VAL_65]] : vector<1xf16>, !llvm.ptr<3>
    // CHECK-COUNT-63:  llvm.store {{.*}}, {{.*}} : vector<1xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {convergent, no_unwind, will_return} : (i32) -> ()

    // COM: Because the values per thread of blocked layout is contiguous. The values are loaded from the SLM in a vectorized way.
    // COM: Total 8 loads are generated to load the tensor of the blocked layout from the SLM. 128*256/(16*2*16*8) = 8
    // CHECK-COUNT-8:    {{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>

    %93 = ttg.convert_layout %cst {allocation.offset = 0 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %80 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %83 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %83, %93 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}


// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [2, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 67584 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @convert_dpas(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<1>)
// CHECK-SAME:                                          attributes {intel_reqd_sub_group_size = 16 : i32, noinline = false, reqd_work_group_size = array<i32: 512, 1, 1>} {
  tt.func public @convert_dpas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #mma>

    // CHECK-DAG:           %[[CST_3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG:           %[[CST_8192:.*]] = llvm.mlir.constant(8192 : i32) : i32
    // CHECK-DAG:           %[[CST_4096:.*]] = llvm.mlir.constant(4096 : i32) : i32
    // CHECK-DAG:           %[[CST_128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:           %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:           %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:           %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:           %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:           %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:           %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:           %[[SMEM:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK-DAG:           %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:           %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:           %[[CST_511:.*]] = llvm.mlir.constant(511 : i32) : i32

    // COM: The following operations is generated for the conversion of DPAS layout to blocked layout. The conversion replica size is 64*256. So there are 2 round of load/store with synchronization.
    // CHECK:           %[[threadId_64:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]]) {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return} : (i32) -> i64
    // CHECK:           %[[threadId:.*]] = llvm.trunc %[[threadId_64]] : i64 to i32
    // CHECK:           %[[rtid:.*]] = llvm.and %[[threadId]], %[[CST_511]] : i32
    // CHECK:           %[[laneId:.*]] = llvm.urem %[[rtid]], %[[CST_16]]  : i32
    // CHECK:           %[[warpId:.*]] = llvm.udiv %[[rtid]], %[[CST_16]]  : i32
    // CHECK:           %[[VAL_25:.*]] = llvm.and %[[laneId]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_26:.*]] = llvm.icmp "eq" %[[VAL_25]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_27:.*]] = llvm.select %[[VAL_26]], %[[CST_0]], %[[CST_1]] : i1, i32
    // CHECK:           %[[VAL_28:.*]] = llvm.xor %[[CST_0]], %[[VAL_27]] : i32
    // CHECK:           %[[VAL_29:.*]] = llvm.and %[[laneId]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_30:.*]] = llvm.icmp "eq" %[[VAL_29]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_31:.*]] = llvm.select %[[VAL_30]], %[[CST_0]], %[[CST_2]] : i1, i32
    // CHECK:           %[[VAL_32:.*]] = llvm.xor %[[VAL_28]], %[[VAL_31]] : i32
    // CHECK:           %[[VAL_33:.*]] = llvm.and %[[laneId]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_34:.*]] = llvm.icmp "eq" %[[VAL_33]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_35:.*]] = llvm.select %[[VAL_34]], %[[CST_0]], %[[CST_4]] : i1, i32
    // CHECK:           %[[VAL_36:.*]] = llvm.xor %[[VAL_32]], %[[VAL_35]] : i32
    // CHECK:           %[[VAL_37:.*]] = llvm.and %[[laneId]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_38:.*]] = llvm.icmp "eq" %[[VAL_37]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_39:.*]] = llvm.select %[[VAL_38]], %[[CST_0]], %[[CST_8]] : i1, i32
    // CHECK:           %[[VAL_40:.*]] = llvm.xor %[[VAL_36]], %[[VAL_39]] : i32
    // CHECK:           %[[VAL_41:.*]] = llvm.and %[[warpId]], %[[CST_1]] : i32
    // CHECK:           %[[VAL_42:.*]] = llvm.icmp "eq" %[[VAL_41]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_43:.*]] = llvm.select %[[VAL_42]], %[[CST_0]], %[[CST_32]] : i1, i32
    // CHECK:           %[[VAL_44:.*]] = llvm.xor %[[VAL_40]], %[[VAL_43]] : i32
    // CHECK:           %[[VAL_45:.*]] = llvm.and %[[warpId]], %[[CST_2]] : i32
    // CHECK:           %[[VAL_46:.*]] = llvm.icmp "eq" %[[VAL_45]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_47:.*]] = llvm.select %[[VAL_46]], %[[CST_0]], %[[CST_64]]  : i1, i32
    // CHECK:           %[[VAL_48:.*]] = llvm.xor %[[VAL_44]], %[[VAL_47]] : i32
    // CHECK:           %[[VAL_49:.*]] = llvm.and %[[warpId]], %[[CST_4]] : i32
    // CHECK:           %[[VAL_50:.*]] = llvm.icmp "eq" %[[VAL_49]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_51:.*]] = llvm.select %[[VAL_50]], %[[CST_0]], %[[CST_128]] : i1, i32
    // CHECK:           %[[VAL_52:.*]] = llvm.xor %[[VAL_48]], %[[VAL_51]] : i32
    // CHECK:           %[[VAL_53:.*]] = llvm.and %[[warpId]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_54:.*]] = llvm.icmp "eq" %[[VAL_53]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_55:.*]] = llvm.select %[[VAL_54]], %[[CST_0]], %[[CST_4096]] : i1, i32
    // CHECK:           %[[VAL_56:.*]] = llvm.xor %[[VAL_52]], %[[VAL_55]] : i32
    // CHECK:           %[[VAL_57:.*]] = llvm.and %[[warpId]], %[[CST_16]] : i32
    // CHECK:           %[[VAL_58:.*]] = llvm.icmp "eq" %[[VAL_57]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_59:.*]] = llvm.select %[[VAL_58]], %[[CST_0]], %[[CST_8192]] : i1, i32
    // CHECK:           %[[VAL_60:.*]] = llvm.xor %[[VAL_56]], %[[VAL_59]] : i32
    // CHECK:           %[[VAL_61:.*]] = llvm.xor %[[VAL_60]], %[[CST_0]] : i32
    // CHECK:           %[[VAL_62:.*]] = llvm.lshr %[[VAL_61]], %[[CST_8]] : i32
    // CHECK:           %[[VAL_63:.*]] = llvm.shl %[[VAL_62]], %[[CST_3]] : i32
    // CHECK:           %[[offset:.*]] = llvm.add %[[VAL_63]], %[[VAL_61]] : i32
    // CHECK:           %[[VAL_65:.*]] = llvm.getelementptr inbounds %[[SMEM]]{{\[}}%[[offset]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:           %[[VAL_66:.*]] = llvm.insertelement {{.*}}, {{.*}}{{\[}}%[[CST_0]] : i32] : vector<1xf16>

    // COM: Because the values per thread of DPAS layout is not contiguous. The values are stored in the SLM in a non-vectorized way.
    // COM: Total 32 stores are generated to save the tensor of the DPAS layout to the SLM. 64*256/(4*8*16) = 32
    // CHECK:           llvm.store %[[VAL_66]], %[[VAL_65]] : vector<1xf16>, !llvm.ptr<3>
    // CHECK-COUNT-31:  llvm.store {{.*}}, {{.*}} : vector<1xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {convergent, no_unwind, will_return} : (i32) -> ()

    // COM: Because the values per thread of blocked layout is contiguous. The values are loaded from the SLM in a vectorized way.
    // COM: Total 4 loads are generated to load the tensor of the blocked layout from the SLM. 128*256/(16*2*16*8) = 8
    // CHECK-COUNT-4:    {{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>

    // COM: The 2nd round of exchanging values.
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {convergent, no_unwind, will_return} : (i32) -> ()
    // CHECK-COUNT-32:  llvm.store {{.*}}, {{.*}} : vector<1xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj(%[[CST_1]]) {convergent, no_unwind, will_return} : (i32) -> ()
    // CHECK-COUNT-4:    {{.*}} = llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>

    %93 = ttg.convert_layout %cst {allocation.offset = 0 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %80 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %83 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %83, %93 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
