// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness | FileCheck %s

// COM: A dot operand that is *already loaded inside* the loop body, but consumed by
// COM: a dot much later in that same body, keeps its whole per-lane footprint live
// COM: for nearly the entire iteration. With "ttg.threads-per-warp" = 16 and the
// COM: default GRF mode the per-lane budget is 16384/16 = 1024 B/lane (see
// COM: reduce-variable-liveness-grf-mode.mlir for why "default" maps to the largest
// COM: GRF size). A 64x64xbf16 DPAS B operand under warpsPerCTA = [8, 1] is
// COM: replicated in every warp and so costs 64*64*2/16 = 512 B/lane on its own.
// COM:
// COM: Every peak quoted below is what
// COM: `triton-opt <this file> -split-input-file --test-register-pressure` reports
// COM: for the `scf.for` body block, not a hand-derived figure.

// COM: Peak 1472 B/lane, above the 1024 B/lane budget, and the candidate's own live
// COM: range covers that peak: the load of the second dot's B operand sinks from the
// COM: top of the body down to just before that dot. The A operand's load stays put:
// COM: its own range peaks at 836 B/lane, within budget.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @sink_late_used_operand(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @sink_late_used_operand
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    // CHECK: %[[DESC_A:.*]] = tt.make_tensor_descriptor %arg0
    %desc_a = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_c = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x64xf32>
    // CHECK: scf.for
    // COM: The candidate must no longer be the first load in the body...
    // CHECK-NOT: tt.descriptor_load %[[DESC_A]]
    // CHECK:     tt.dot
    // COM: ...it must sit between the two dots.
    // CHECK:     tt.descriptor_load %[[DESC_A]]
    // CHECK:     tt.dot
    %r:2 = scf.for %iv = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%acc0 = %cst, %acc1 = %cst) -> (tensor<64x64xf32, #dpas>, tensor<64x64xf32, #dpas>) : i32 {
      %late = tt.descriptor_load %desc_a[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1>
      %a = tt.descriptor_load %desc_b[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot0>
      %b = tt.descriptor_load %desc_b[%c0_i32, %iv] {ttig.block_io = "column_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1>
      %d0 = tt.dot %a, %b, %acc0, inputPrecision = tf32 : tensor<64x64xbf16, #dot0> * tensor<64x64xbf16, #dot1> -> tensor<64x64xf32, #dpas>
      %d1 = tt.dot %a, %late, %acc1, inputPrecision = tf32 : tensor<64x64xbf16, #dot0> * tensor<64x64xbf16, #dot1> -> tensor<64x64xf32, #dpas>
      scf.yield %d0, %d1 : tensor<64x64xf32, #dpas>, tensor<64x64xf32, #dpas>
    }
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %r#0 : !tt.tensordesc<64x64xf32>, tensor<64x64xf32, #dpas>
    tt.descriptor_store %desc_c[%c64_i32, %c0_i32], %r#1 : !tt.tensordesc<64x64xf32>, tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----

// COM: Same loop, but the candidate is loaded *after* the heavy part of the body, so
// COM: the interval the sink would shorten peaks at 836 B/lane -- below the budget --
// COM: even though the loop's peak (1348 B/lane) is above it.
// COM: Shortening a live range can only lower the pressure at points inside the range
// COM: it removes, so this sink could not relieve any spilling and would only expose
// COM: the load's latency: the load stays put. This is the case that distinguishes
// COM: the per-load check from the loop-wide gate; without it the load would move.
#dpas1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot0_1 = #ttg.dot_op<{opIdx = 0, parent = #dpas1, kWidth = 1}>
#dot1_1 = #ttg.dot_op<{opIdx = 1, parent = #dpas1, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @keep_operand_outside_pressure_peak(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @keep_operand_outside_pressure_peak
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas1>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    // CHECK: %[[DESC_A:.*]] = tt.make_tensor_descriptor %arg0
    %desc_a = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_c = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x64xf32>
    // CHECK:     scf.for
    // CHECK:     tt.dot
    // COM: The candidate stays where it was: above the prefetch, not below it. (The
    // COM: prefetch is not what holds it back -- an L2Cache write is deliberately not
    // COM: a barrier here, see @sink_past_shared_memory_write -- it is only a marker
    // COM: that survives dead-code elimination and so pins the load's position.)
    // CHECK:     tt.descriptor_load %[[DESC_A]]
    // CHECK:     ttig.descriptor_prefetch
    // CHECK:     tt.dot
    %r:2 = scf.for %iv = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%acc0 = %cst, %acc1 = %cst) -> (tensor<64x64xf32, #dpas1>, tensor<64x64xf32, #dpas1>) : i32 {
      %a = tt.descriptor_load %desc_b[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot0_1>
      %b = tt.descriptor_load %desc_b[%c0_i32, %iv] {ttig.block_io = "column_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_1>
      %b2 = tt.descriptor_load %desc_b[%iv, %iv] {ttig.block_io = "column_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_1>
      %d0 = tt.dot %a, %b, %acc0, inputPrecision = tf32 : tensor<64x64xbf16, #dot0_1> * tensor<64x64xbf16, #dot1_1> -> tensor<64x64xf32, #dpas1>
      %dead = tt.dot %a, %b2, %acc0, inputPrecision = tf32 : tensor<64x64xbf16, #dot0_1> * tensor<64x64xbf16, #dot1_1> -> tensor<64x64xf32, #dpas1>
      %late = tt.descriptor_load %desc_a[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_1>
      ttig.descriptor_prefetch %desc_b[%iv, %c0_i32] : !tt.tensordesc<64x64xbf16>
      %d1 = tt.dot %a, %late, %dead, inputPrecision = tf32 : tensor<64x64xbf16, #dot0_1> * tensor<64x64xbf16, #dot1_1> -> tensor<64x64xf32, #dpas1>
      scf.yield %d0, %d1 : tensor<64x64xf32, #dpas1>, tensor<64x64xf32, #dpas1>
    }
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %r#0 : !tt.tensordesc<64x64xf32>, tensor<64x64xf32, #dpas1>
    tt.descriptor_store %desc_c[%c64_i32, %c0_i32], %r#1 : !tt.tensordesc<64x64xf32>, tensor<64x64xf32, #dpas1>
    tt.return
  }
}

// -----

// COM: Peak 1156 B/lane. A shared-memory write between the load and its use does not
// COM: block the sink: it
// COM: cannot change what a global load reads. This is the shape the flash-attention
// COM: backward kernel has, where the SLM round trip of a `tt.trans` sits between the
// COM: operand load and the dot that consumes it.
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot0_2 = #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth = 1}>
#dot1_2 = #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth = 2}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @sink_past_shared_memory_write(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @sink_past_shared_memory_write
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas2>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    // CHECK: %[[DESC_A:.*]] = tt.make_tensor_descriptor %arg0
    %desc_a = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_c = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x64xf32>
    // CHECK:     scf.for
    // CHECK-NOT: tt.descriptor_load %[[DESC_A]]
    // CHECK:     ttg.local_alloc
    // CHECK:     ttg.local_load
    // CHECK:     tt.descriptor_load %[[DESC_A]]
    // CHECK:     tt.dot
    %r = scf.for %iv = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x64xf32, #dpas2>) : i32 {
      %late = tt.descriptor_load %desc_a[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_2>
      %b = tt.descriptor_load %desc_b[%c0_i32, %iv] {ttig.block_io = "column_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_2>
      %alloc = ttg.local_alloc %b : (tensor<64x64xbf16, #dot1_2>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
      %a = ttg.local_load %alloc : !ttg.memdesc<64x64xbf16, #shared, #smem> -> tensor<64x64xbf16, #dot0_2>
      %d = tt.dot %a, %late, %acc, inputPrecision = tf32 : tensor<64x64xbf16, #dot0_2> * tensor<64x64xbf16, #dot1_2> -> tensor<64x64xf32, #dpas2>
      scf.yield %d : tensor<64x64xf32, #dpas2>
    }
    tt.descriptor_store %desc_c[%c0_i32, %c0_i32], %r : !tt.tensordesc<64x64xf32>, tensor<64x64xf32, #dpas2>
    tt.return
  }
}

// -----

// COM: Peak 1348 B/lane. A global-memory write between the load and its use does block
// COM: the sink: the
// COM: store may alias the loaded tile, so moving the load past it would change what
// COM: the load reads.
#dpas3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dot0_3 = #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>
#dot1_3 = #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @keep_operand_across_global_write(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @keep_operand_across_global_write
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas3>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    // CHECK: %[[DESC_A:.*]] = tt.make_tensor_descriptor %arg0
    %desc_a = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_b = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <bf16>, <64x64xbf16>
    %desc_c = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x64xf32>
    // CHECK:     scf.for
    // COM: The candidate stays above the store.
    // CHECK:     tt.descriptor_load %[[DESC_A]]
    // CHECK:     tt.descriptor_store
    // CHECK:     tt.dot
    %r = scf.for %iv = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<64x64xf32, #dpas3>) : i32 {
      %late = tt.descriptor_load %desc_a[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_3>
      %a = tt.descriptor_load %desc_b[%iv, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot0_3>
      %b = tt.descriptor_load %desc_b[%c0_i32, %iv] {ttig.block_io = "column_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #dot1_3>
      %d0 = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x64xbf16, #dot0_3> * tensor<64x64xbf16, #dot1_3> -> tensor<64x64xf32, #dpas3>
      tt.descriptor_store %desc_c[%iv, %c0_i32], %d0 : !tt.tensordesc<64x64xf32>, tensor<64x64xf32, #dpas3>
      %d1 = tt.dot %a, %late, %d0, inputPrecision = tf32 : tensor<64x64xbf16, #dot0_3> * tensor<64x64xbf16, #dot1_3> -> tensor<64x64xf32, #dpas3>
      scf.yield %d1 : tensor<64x64xf32, #dpas3>
    }
    tt.return
  }
}
