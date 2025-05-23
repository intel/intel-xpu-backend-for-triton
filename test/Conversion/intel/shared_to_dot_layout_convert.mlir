// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm -canonicalize | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @convert_dot(
  // CHECK-SAME:    %[[VAL_0:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>)
  // CHECK-SAME:    attributes {intel_reqd_sub_group_size = 16 : i32, {{.*}}} {
  tt.func @convert_dot(%A: tensor<128x64xf16, #blocked0>) {
    // CHECK-DAG:     %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    %AA = ttg.local_alloc %A : (tensor<128x64xf16, #blocked0>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>

    // CHECK:         llvm.call spir_funccc @_Z7barrierj
    // COM:   Start of ttg.local_load. Load the value from SLM to register.
    // CHECK:         %[[WORK_ITEM_ID_:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:         %[[WORK_ITEM_ID:.*]] = llvm.trunc %[[WORK_ITEM_ID_]] : i64 to i32
    // CHECK-COUNT-128:        %[[LD_RES:.*]] = llvm.load {{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xf16>
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> tensor<128x64xf16, #dot_operand_a>

    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #dot_operand_b>
    %D = tt.dot %AA_DOT, %cst1, %cst0 : tensor<128x64xf16, #dot_operand_a> * tensor<64x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>

    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [32, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @convert_dot(
  // CHECK-SAME:    %[[VAL_0:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>)
  // CHECK-SAME:    attributes {intel_reqd_sub_group_size = 16 : i32, {{.*}}} {
  tt.func @convert_dot(%A: tensor<128x64xf16, #blocked0>) {
    // CHECK-DAG:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:     %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    %AA = ttg.local_alloc %A : (tensor<128x64xf16, #blocked0>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>

    // CHECK:         llvm.call spir_funccc @_Z7barrierj
    // COM:   Start of ttg.local_load. Load the value from SLM to register.
    // CHECK:         %[[WORK_ITEM_ID_:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:         %[[WORK_ITEM_ID:.*]] = llvm.trunc %[[WORK_ITEM_ID_]] : i64 to i32
    // CHECK-COUNT-128:        %[[LD_RES:.*]] = llvm.load {{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xf16>
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> -> tensor<128x64xf16, #dot_operand_a>

    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #dot_operand_b>
    %D = tt.dot %AA_DOT, %cst1, %cst0 : tensor<128x64xf16, #dot_operand_a> * tensor<64x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>

    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#dpas, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @convert_dot(
  // CHECK-SAME:    %[[VAL_1:.*]]: !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>)
  // CHECK-SAME:    attributes {intel_reqd_sub_group_size = 16 : i32, {{.*}}} {
  tt.func @convert_dot(%B: tensor<64x256xf16, #blocked1>) {
    // CHECK-DAG:     %[[CST_128:.*]] = llvm.mlir.constant(128 : i32) : i32
    // CHECK-DAG:     %[[CST_256:.*]] = llvm.mlir.constant(256 : i32) : i32
    // CHECK-DAG:     %[[CST_32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-DAG:     %[[CST_4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-DAG:     %[[CST_2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:     %[[CST_8:.*]] = llvm.mlir.constant(8 : i32) : i32
    // CHECK-DAG:     %[[CST_16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG:     %[[CST_64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK-DAG:     %[[CST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:     %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    %BB = ttg.local_alloc %B : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #ttg.shared_memory>

    // CHECK:         llvm.call spir_funccc @_Z7barrierj
    // COM:   Start of ttg.local_load. Load the value from SLM to register.
    // CHECK:         %[[WORK_ITEM_ID_:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[CST_0]])
    // CHECK:         %[[WORK_ITEM_ID:.*]] = llvm.trunc %[[WORK_ITEM_ID_]] : i64 to i32
    // CHECK-COUNT-128:        %[[LD_RES:.*]] = llvm.load {{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xf16>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<64x256xf16, #shared, #ttg.shared_memory> -> tensor<64x256xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #dot_operand_a>

    %D = tt.dot %cst1, %BB_DOT, %cst0 : tensor<128x64xf16, #dot_operand_a> * tensor<64x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>

    tt.return
  }
}
