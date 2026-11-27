// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions -verify-each 2>&1 | FileCheck %s --enable-var-scope

// COM: Test that RemoveLayoutConversions never rewrites a tt.store so that its
// COM: ptr/value/mask operands end up with different encodings.
// COM:
// COM: In LayoutPropagation, the candidate encodings of each store operand are
// COM: accumulated independently by different upstream chains (here: two
// COM: distinct dpas encodings feeding the value and the mask through
// COM: convert_layouts, plus the pointer argument anchored in a blocked
// COM: layout). resolveConflicts() picks one encoding per value independently,
// COM: so the operands can resolve to *different* layouts. rewriteStoreOp()
// COM: previously only checked that every operand changed encoding, and then
// COM: assigned each operand its own resolved encoding, producing a store that
// COM: violates the tt.store verifier ("value type matches ptr type").
// COM:
// COM: With the fix, the incompatible encodings are not propagated onto the
// COM: store operands in the first place, and the inconsistent rewrite is
// COM: rejected as a backstop. The store keeps a single encoding for all
// COM: operands and the converts feeding it are preserved.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas_a = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas_b = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttg.target = "xpu", ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @store_operand_encoding_conflict
  // COM: The pointer is anchored in the blocked layout, the value and mask in
  // COM: two different dpas layouts.
  tt.func @store_operand_encoding_conflict(
      %ptr: tensor<16x16x!tt.ptr<f32>, #blocked_a>,
      %val: tensor<16x16xf32, #dpas_b>,
      %mask: tensor<16x16xi1, #dpas_a>) {
    // COM: The value and mask reach the store through converts from two
    // COM: different dpas encodings. Forward propagation pushes each dpas
    // COM: encoding onto all three store operands; the operands then resolve
    // COM: their conflicts independently and may pick different layouts.
    // COM: The converts must survive with their original dpas sources, i.e.
    // COM: the dpas encodings must not be folded into the store operands.
    // CHECK: %[[V:.*]] = ttg.convert_layout %arg1 : tensor<16x16xf32, #[[DPAS_B:.+]]> -> tensor<16x16xf32, #[[BLOCKED:.+]]>
    // CHECK: %[[M:.*]] = ttg.convert_layout %arg2 : tensor<16x16xi1, #[[DPAS_A:.+]]> -> tensor<16x16xi1, #[[BLOCKED]]>
    %v = ttg.convert_layout %val : tensor<16x16xf32, #dpas_b> -> tensor<16x16xf32, #blocked_a>
    %m = ttg.convert_layout %mask : tensor<16x16xi1, #dpas_a> -> tensor<16x16xi1, #blocked_a>
    // COM: The store must survive with all operands on a single, consistent
    // COM: encoding.
    // CHECK: tt.store %arg0, %[[V]], %[[M]] : tensor<16x16x!tt.ptr<f32>, #[[BLOCKED]]>
    tt.store %ptr, %v, %m : tensor<16x16x!tt.ptr<f32>, #blocked_a>
    tt.return
  }
}

// -----

// COM: The propagation gate above only covers encodings that reach the store
// COM: through forward propagation onto the store operands. A store operand
// COM: can also accumulate a private candidate encoding from its own def
// COM: chain: here every operand reaches the store through a convert_layout
// COM: whose source carries a distinct dpas encoding, so the gate never
// COM: fires and resolveConflicts() may resolve the operands to *different*
// COM: encodings. rewriteStoreOp() must reject such a rewrite: without the
// COM: consistency check it would assign each operand its own encoding and
// COM: produce a store that violates the tt.store verifier.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas_p = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas_v = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas_m = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttg.target = "xpu", ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @store_operand_backstop
  tt.func @store_operand_backstop(
      %off: tensor<16x16xi32, #dpas_p>,
      %msk: tensor<16x16xi1, #dpas_m>,
      %val: tensor<16x16xf32, #dpas_v>,
      %base: tensor<16x16x!tt.ptr<f32>, #blocked>) {
    %offc = ttg.convert_layout %off : tensor<16x16xi32, #dpas_p> -> tensor<16x16xi32, #blocked>
    %p = tt.addptr %base, %offc : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    %v = ttg.convert_layout %val : tensor<16x16xf32, #dpas_v> -> tensor<16x16xf32, #blocked>
    %m = ttg.convert_layout %msk : tensor<16x16xi1, #dpas_m> -> tensor<16x16xi1, #blocked>
    // COM: The operands resolve to three different dpas encodings; the
    // COM: rewrite is rejected and every operand is converted back to the
    // COM: single blocked encoding the store keeps.
    // CHECK: tt.addptr
    // CHECK: %[[P:.*]] = ttg.convert_layout {{.*}} : tensor<16x16x!tt.ptr<f32>, #{{.+}}> -> tensor<16x16x!tt.ptr<f32>, #[[ENC:.+]]>
    // CHECK: %[[V:.*]] = ttg.convert_layout {{.*}} : tensor<16x16xf32, #{{.+}}> -> tensor<16x16xf32, #[[ENC]]>
    // CHECK: %[[M:.*]] = ttg.convert_layout {{.*}} : tensor<16x16xi1, #{{.+}}> -> tensor<16x16xi1, #[[ENC]]>
    // CHECK: tt.store %[[P]], %[[V]], %[[M]] : tensor<16x16x!tt.ptr<f32>, #[[ENC]]>
    tt.store %p, %v, %m : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
