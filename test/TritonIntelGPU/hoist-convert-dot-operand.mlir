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

// -----

// COM: Negative case: an fp8 -> fp16 upcast sits between the load and the
// COM: convert_layout. The convert target is #dot_op<dpas, kWidth=2> whose
// COM: kWidth is parameterized for the fp16 dot operand. The hoist must NOT
// COM: cross the upcast — doing so would place a convert_layout on 8-bit
// COM: data with a layout sized for 16-bit, collapsing to a sub-group
// COM: transpose-through-SLM and disabling 2D block I/O on the load.
// COM: Issue #6737.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_hoist_across_fp8_upcast
  tt.func public @no_hoist_across_fp8_upcast(
      %arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f16>,
      %arg2: !tt.ptr<f32>, %arg4: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f8E4M3FN>, <64x32xf8E4M3FN>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <32x256xf16>
    // CHECK: scf.for
    // COM: The f8 descriptor_load must remain paired with fp_to_fp (not with a
    // COM: convert_layout). The convert_layout must stay on the fp16 side.
    // CHECK:   %[[A_F8:.*]] = tt.descriptor_load {{.*}} -> tensor<64x32xf8E4M3FN, #blocked>
    // CHECK-NOT: ttg.convert_layout %[[A_F8]]
    // CHECK:   %[[A_F16:.*]] = tt.fp_to_fp %[[A_F8]] : tensor<64x32xf8E4M3FN, #blocked> -> tensor<64x32xf16, #blocked>
    // CHECK:   ttg.convert_layout %[[A_F16]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK:   tt.dot
    // CHECK:   scf.yield
    %result = scf.for %iv = %c0_i32 to %arg4 step %c32_i32 iter_args(%acc = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %a_f8 = tt.descriptor_load %desc_a[%c0_i32, %iv] : !tt.tensordesc<64x32xf8E4M3FN> -> tensor<64x32xf8E4M3FN, #blocked>
      %a_f16 = tt.fp_to_fp %a_f8 : tensor<64x32xf8E4M3FN, #blocked> -> tensor<64x32xf16, #blocked>
      %b = tt.descriptor_load %desc_b[%iv, %c0_i32] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #blocked1>
      %a_dot = ttg.convert_layout %a_f16 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %b_dot = ttg.convert_layout %b : tensor<32x256xf16, #blocked1> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %d : tensor<64x256xf32, #dpas>
    }
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <64x256xf32>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %result : !tt.tensordesc<64x256xf32>, tensor<64x256xf32, #dpas>
    tt.return
  }
}

// -----

// COM: Negative case: an arith.extf f16 -> f32 sits between the load and the
// COM: convert_layout. Same reasoning as the fp8 upcast case.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @no_hoist_across_extf
  tt.func public @no_hoist_across_extf(
      %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f32>,
      %arg2: !tt.ptr<f32>, %arg4: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <32x256xf32>
    // CHECK: scf.for
    // CHECK:   %[[A_F16:.*]] = tt.descriptor_load {{.*}} -> tensor<64x32xf16, #blocked>
    // CHECK-NOT: ttg.convert_layout %[[A_F16]]
    // CHECK:   %[[A_F32:.*]] = arith.extf %[[A_F16]] : tensor<64x32xf16, #blocked> to tensor<64x32xf32, #blocked>
    // CHECK:   ttg.convert_layout %[[A_F32]]
    // CHECK:   tt.dot
    // CHECK:   scf.yield
    %result = scf.for %iv = %c0_i32 to %arg4 step %c32_i32 iter_args(%acc = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %a_f16 = tt.descriptor_load %desc_a[%c0_i32, %iv] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
      %a_f32 = arith.extf %a_f16 : tensor<64x32xf16, #blocked> to tensor<64x32xf32, #blocked>
      %b = tt.descriptor_load %desc_b[%iv, %c0_i32] : !tt.tensordesc<32x256xf32> -> tensor<32x256xf32, #blocked1>
      %a_dot = ttg.convert_layout %a_f32 : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dot0>
      %b_dot = ttg.convert_layout %b : tensor<32x256xf32, #blocked1> -> tensor<32x256xf32, #dot1>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf32, #dot0> * tensor<32x256xf32, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %d : tensor<64x256xf32, #dpas>
    }
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <64x256xf32>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %result : !tt.tensordesc<64x256xf32>, tensor<64x256xf32, #dpas>
    tt.return
  }
}

// -----

// COM: Positive case: a W-chain where an intermediate convert_layout
// COM: (blocked -> blocked2) sits between the load and the dot-operand
// COM: convert. Both the intermediate convert and the DPAS consumer are inside
// COM: the loop. hoistConvertDotOperand should propagate through the absorbed
// COM: convert and place the dot-operand convert next to the load.
// COM:
// COM: Before:
// COM:   load(blocked) -> convert_layout(blocked2) -> addf(blocked2)
// COM:     -> convert_layout(dot_op) -> dot
// COM: After:
// COM:   load(blocked) -> convert_layout(dot_op) -> addf(dot_op) -> dot

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @hoist_through_convert_in_loop
  tt.func public @hoist_through_convert_in_loop(
      %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>,
      %arg2: !tt.ptr<f32>, %arg4: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <32x256xf16>
    // CHECK: scf.for
    // CHECK:   %[[A:.*]] = tt.descriptor_load {{.*}} -> tensor<64x32xf16, #blocked>
    // CHECK:   %[[A_DOT:.*]] = ttg.convert_layout %[[A]] : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK:   arith.addf %[[A_DOT]], %[[A_DOT]] : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK:   tt.dot
    // CHECK:   scf.yield
    %result = scf.for %iv = %c0_i32 to %arg4 step %c32_i32 iter_args(%acc = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %desc_a[%c0_i32, %iv] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
      %a2 = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked2>
      %a3 = arith.addf %a2, %a2 : tensor<64x32xf16, #blocked2>
      %a_dot = ttg.convert_layout %a3 : tensor<64x32xf16, #blocked2> -> tensor<64x32xf16, #dot0>
      %b = tt.descriptor_load %desc_b[%iv, %c0_i32] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #blocked3>
      %b_dot = ttg.convert_layout %b : tensor<32x256xf16, #blocked3> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %d : tensor<64x256xf32, #dpas>
    }
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <64x256xf32>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %result : !tt.tensordesc<64x256xf32>, tensor<64x256xf32, #dpas>
    tt.return
  }
}

// -----

// COM: Negative case (G1): a slice value produced by the absorbed
// COM: convert_layout has two users — one inside the slice (addf feeding the
// COM: dot) and one outside (a store). The single-use guard must prevent the
// COM: hoist so that the store's input is not broken.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @hoist_bail_on_multi_use
  tt.func public @hoist_bail_on_multi_use(
      %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>,
      %arg2: !tt.ptr<f32>, %arg3: tensor<64x32x!tt.ptr<f16>, #blocked2>,
      %arg4: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <32x256xf16>
    // COM: The load result feeds both the addf (via the intermediate convert,
    // COM: inside the slice) and a store (directly, outside the slice). G1
    // COM: detects the out-of-slice user and prevents propagating through the
    // COM: intermediate convert. The intermediate convert is preserved for the
    // COM: store; the dot gets its own direct convert from the load.
    // CHECK: scf.for
    // CHECK-DAG:   ttg.convert_layout {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK-DAG:   ttg.convert_layout {{.*}} -> tensor<64x32xf16, #blocked>
    // CHECK:   tt.dot
    // CHECK:   scf.yield
    %result = scf.for %iv = %c0_i32 to %arg4 step %c32_i32 iter_args(%acc = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %desc_a[%c0_i32, %iv] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
      %a2 = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked2>
      %a3 = arith.addf %a2, %a2 : tensor<64x32xf16, #blocked2>
      %a_dot = ttg.convert_layout %a3 : tensor<64x32xf16, #blocked2> -> tensor<64x32xf16, #dot0>
      %b = tt.descriptor_load %desc_b[%iv, %c0_i32] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #blocked3>
      %b_dot = ttg.convert_layout %b : tensor<32x256xf16, #blocked3> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      // COM: This store uses %a2 directly, putting a user of the absorbed
      // COM: convert's result outside the slice.
      tt.store %arg3, %a2 : tensor<64x32x!tt.ptr<f16>, #blocked2>
      scf.yield %d : tensor<64x256xf32, #dpas>
    }
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <64x256xf32>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %result : !tt.tensordesc<64x256xf32>, tensor<64x256xf32, #dpas>
    tt.return
  }
}

// -----

// COM: Preheader load case: the load and intermediate convert are in the
// COM: function entry block (preheader). LayoutPropagation runs before
// COM: hoistConvertDotOperand and assigns #dot0 to the preheader convert's
// COM: result, so by the time hoisting runs the #dot0 convert is already in
// COM: the preheader. hoistConvertDotOperand then hoists normally (no
// COM: propagation through convert needed).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @hoist_bail_on_out_of_loop_root
  tt.func public @hoist_bail_on_out_of_loop_root(
      %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>,
      %arg2: !tt.ptr<f32>, %arg4: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas>
    %desc_a = tt.make_tensor_descriptor %arg0, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <64x32xf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f16>, <32x256xf16>
    // COM: LayoutPropagation converts the preheader chain to #dot0 before
    // COM: hoistConvertDotOperand runs, so the #dot0 convert lands in the
    // COM: preheader and the loop body only sees addf + dot.
    %a = tt.descriptor_load %desc_a[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    // CHECK: ttg.convert_layout {{.*}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %a2 = ttg.convert_layout %a : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked2>
    // CHECK: scf.for
    // CHECK:   arith.addf {{.*}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK:   tt.dot
    // CHECK:   scf.yield
    %result = scf.for %iv = %c0_i32 to %arg4 step %c32_i32 iter_args(%acc = %cst) -> (tensor<64x256xf32, #dpas>) : i32 {
      %a3 = arith.addf %a2, %a2 : tensor<64x32xf16, #blocked2>
      %a_dot = ttg.convert_layout %a3 : tensor<64x32xf16, #blocked2> -> tensor<64x32xf16, #dot0>
      %b = tt.descriptor_load %desc_b[%iv, %c0_i32] : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #blocked3>
      %b_dot = ttg.convert_layout %b : tensor<32x256xf16, #blocked3> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<64x256xf32, #dpas>
      scf.yield %d : tensor<64x256xf32, #dpas>
    }
    %desc_c = tt.make_tensor_descriptor %arg2, [%arg4, %arg4], [%c0_i64, %c1_i64] : <f32>, <64x256xf32>
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %result : !tt.tensordesc<64x256xf32>, tensor<64x256xf32, #dpas>
    tt.return
  }
}
