// RUN: triton-opt %s -split-input-file --tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Test that hoistConvertDotOperand hoists a ConvertLayoutOp targeting a
// COM: DPAS DotOperandEncodingAttr next to the load, past elementwise ops.
// COM:
// COM: The elementwise op (arith.addf) has a user outside the backward slice
// COM: (the tt.store), which forces backward rematerialization to account for
// COM: duplication cost and skip the rewrite. hoistConvertDotOperand is not
// COM: subject to this cost model — it unconditionally hoists the convert next
// COM: to the load.
// COM:
// COM: Before the hoist:
// COM:   load(blocked) -> addf(blocked) --+--> convert_layout(dot_op) -> dot
// COM:                                    \--> store
// COM: After the hoist:
// COM:   load(blocked) -> convert_layout(dot_op) -> addf(dot_op) -> dot
// COM:   load(blocked) -> addf(blocked) -> store  [original addf kept for store user]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @hoist_convert_over_elementwise_dpas
  tt.func public @hoist_convert_over_elementwise_dpas(
      %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>,
      %arg2: !tt.ptr<f32>, %arg3: tensor<64x32x!tt.ptr<f16>, #blocked>,
      %arg4: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <32x256xf16>
    // COM: In the loop, the addf result feeds both the convert_layout (for dot)
    // COM: and a store. The store use prevents backward rematerialization from
    // COM: firing (duplication cost). hoistConvertDotOperand should still hoist
    // COM: the convert next to the load.
    //
    // CHECK: scf.for
    // CHECK:   %[[A:.*]] = tt.descriptor_load {{.*}} -> tensor<64x32xf16, #blocked>
    // CHECK:   %[[A_DOT:.*]] = ttg.convert_layout %[[A]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK:   %[[A2_DOT:.*]] = arith.addf %[[A_DOT]], %[[A_DOT]] : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK:   %[[A2_BLOCKED:.*]] = arith.addf %[[A]], %[[A]] : tensor<64x32xf16, #blocked>
    // CHECK:   tt.dot %[[A2_DOT]], {{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x256xf32, #mma>
    // CHECK:   tt.store %arg3, %[[A2_BLOCKED]] : tensor<64x32x!tt.ptr<f16>, #blocked>
    // CHECK:   scf.yield
    %result = scf.for %iv = %c0_i32 to %arg4 step %c32_i32 iter_args(%acc = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %desc_a[%c0_i32, %iv] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
      %b = tt.descriptor_load %desc_b[%iv, %c0_i32] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #blocked1>
      %a2 = arith.addf %a, %a : tensor<64x32xf16, #blocked>
      %a_dot = ttg.convert_layout %a2 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %b_dot = ttg.convert_layout %b : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      // COM: This store keeps the addf result alive outside the backward slice,
      // COM: preventing backward rematerialization from duplicating it.
      tt.store %arg3, %a2 : tensor<64x32x!tt.ptr<f16>, #blocked>
      scf.yield %d : tensor<64x256xf32, #dpas>
    }
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <64x256xf32>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %result : !tt.tensordesc<64x256xf32>, tensor<64x256xf32, #dpas>
    tt.return
  }
}
