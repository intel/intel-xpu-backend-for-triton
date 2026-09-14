// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions -verify-each 2>&1 | FileCheck %s --enable-var-scope

// COM: Test that RemoveLayoutConversions does not propagate an mma-derived
// COM: encoding onto the operands of a store when one of them is an
// COM: arith.constant.
// COM:
// COM: A store operand that is shared with another store carries the
// COM: mma-derived candidate into that store. When the second store's value
// COM: is a constant, the candidate is pushed onto the constant as well.
// COM: resolveConflicts() prefers mma-derived encodings for non-load/store
// COM: ops, so the constant is resolved to the mma-derived encoding, and
// COM: rewriteOp() re-types the constant in place. A constant's value
// COM: attribute must keep the result type, so the rewritten op fails the
// COM: arith.constant verifier. Skip propagating onto such stores: a
// COM: constant operand cannot be re-encoded, and propagating onto the
// COM: remaining operands only would leave the store with mixed operand
// COM: encodings.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dotop1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @store_constant_operand
  tt.func public @store_constant_operand(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // COM: The zero constant must survive on its original blocked encoding; it
    // COM: cannot be re-typed to the mma-derived encoding.
    // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #[[BLOCKED:.+]]>
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %cstAcc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %acc = ttg.convert_layout %cstAcc : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cond = arith.cmpi slt, %c0, %c1 : i32
    %m = tt.splat %cond : i1 -> tensor<16x16xi1, #blocked>
    %off = arith.constant dense<0> : tensor<16x16xi32, #blocked>
    %pb = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %p0 = tt.addptr %pb, %off : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    %pa = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    %a = tt.load %pa, %m : tensor<16x16x!tt.ptr<f16>, #blocked>
    %a0 = ttg.convert_layout %a : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dotop0>
    %a1 = ttg.convert_layout %a : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dotop1>
    %d = tt.dot %a0, %a1, %acc : tensor<16x16xf16, #dotop0> * tensor<16x16xf16, #dotop1> -> tensor<16x16xf32, #dpas>
    %d2 = ttg.convert_layout %d : tensor<16x16xf32, #dpas> -> tensor<16x16xf32, #blocked>
    // COM: The first store is folded onto the dot encoding as usual.
    // CHECK: tt.store {{.+}}, {{.+}}, {{.+}} : tensor<16x16x!tt.ptr<f32>, #[[DPAS:.+]]>
    tt.store %p0, %d2, %m : tensor<16x16x!tt.ptr<f32>, #blocked>
    // COM: The store of the constant keeps a single encoding shared by all of
    // COM: its operands: the blocked zero constant is stored as the value.
    // CHECK: tt.store {{.+}}, %[[CST]], {{.+}} : tensor<16x16x!tt.ptr<f32>, #[[BLOCKED]]>
    tt.store %p0, %cst, %m : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: Same guard when the constant is the store mask rather than the value:
// COM: the mma-derived candidate would be pushed onto the constant mask,
// COM: which then fails the arith.constant verifier for the same reason.

#blocked_m = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas_m = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dotop0_m = #ttg.dot_op<{opIdx = 0, parent = #dpas_m, kWidth = 1}>
#dotop1_m = #ttg.dot_op<{opIdx = 1, parent = #dpas_m, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @store_constant_mask
  tt.func public @store_constant_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // COM: The mask constant must survive on its original blocked encoding.
    // CHECK: %[[MASK:.*]] = arith.constant dense<true> : tensor<16x16xi1, #[[BLOCKED:.+]]>
    %cstMask = arith.constant dense<true> : tensor<16x16xi1, #blocked_m>
    %cstAcc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked_m>
    %acc = ttg.convert_layout %cstAcc : tensor<16x16xf32, #blocked_m> -> tensor<16x16xf32, #dpas_m>
    %off = arith.constant dense<0> : tensor<16x16xi32, #blocked_m>
    %pb = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked_m>
    %p0 = tt.addptr %pb, %off : tensor<16x16x!tt.ptr<f32>, #blocked_m>, tensor<16x16xi32, #blocked_m>
    %pa = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked_m>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>, #blocked_m>
    %a0 = ttg.convert_layout %a : tensor<16x16xf16, #blocked_m> -> tensor<16x16xf16, #dotop0_m>
    %a1 = ttg.convert_layout %a : tensor<16x16xf16, #blocked_m> -> tensor<16x16xf16, #dotop1_m>
    %d = tt.dot %a0, %a1, %acc : tensor<16x16xf16, #dotop0_m> * tensor<16x16xf16, #dotop1_m> -> tensor<16x16xf32, #dpas_m>
    %d2 = ttg.convert_layout %d : tensor<16x16xf32, #dpas_m> -> tensor<16x16xf32, #blocked_m>
    // COM: The store of the constant mask keeps a single encoding shared by
    // COM: all operands.
    // CHECK: tt.store {{.+}}, {{.+}}, %[[MASK]] : tensor<16x16x!tt.ptr<f32>, #[[BLOCKED]]>
    tt.store %p0, %d2, %cstMask : tensor<16x16x!tt.ptr<f32>, #blocked_m>
    tt.return
  }
}
