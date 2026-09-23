// RUN: triton-opt %s -split-input-file -test-register-pressure | FileCheck %s
// RUN: triton-opt %s -split-input-file -test-register-pressure=per-op | FileCheck %s --check-prefix=PEROP

// The test pass prints the bare function name, then the analysis report, then
// the IR. Anchor CHECK-LABEL on the pass's bare-name line so the report checks
// are contiguous.
//
// The first report line is the peak over *every* block, which a per-kernel
// allocation must cover; the per-block lines follow.
//
// A value nothing reads contributes nothing to any of these figures: it never
// needs to occupy a register. Below, the outer tt.func block's own top-level ops
// hold nothing for the scf.for's result (tt.return does not take it), but the
// block's reported peak still comes from the nested loop body: a block's peak
// covers regions nested inside it too, not just its own top-level ops.
// CHECK-LABEL: loop_with_dpas_accumulator
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 256 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 256 bytes, live-in = 0 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 256 bytes, live-in = 0 bytes
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @loop_with_dpas_accumulator(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    // COM: A loop holding a DPAS accumulator. At the tt.dot in the loop body the
    // COM: live values (per-thread bytes) are: operand A tensor<8x64xf16,#dot0>
    // COM: = 32 elems * 2B = 64; operand B tensor<64x16xf16,#dot1> = 64 * 2B =
    // COM: 128; accumulator iter_arg tensor<8x16xf32,#dpas> = 8 * 4B = 32; dot
    // COM: result (same type) = 32. Peak of the loop body block = 256 bytes.
    // COM: The outer func block's own top-level ops only hold the scf.for result,
    // COM: which tt.return does not take, so it costs 0 bytes there (nothing
    // COM: consumes it) -- but a block's peak covers nested regions too, so the
    // COM: func block still reports the loop body's 256 bytes.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst) -> (tensor<8x16xf32, #dpas>) : i32 {
      %3 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0>
      %4 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1>
      %5 = tt.dot %3, %4, %arg3, inputPrecision = tf32 : tensor<8x64xf16, #dot0> * tensor<64x16xf16, #dot1> -> tensor<8x16xf32, #dpas>
      scf.yield %5 : tensor<8x16xf32, #dpas>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: block_with_excluded_ops
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 0 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 0 bytes, live-in = 0 bytes
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @block_with_excluded_ops(%arg0: !tt.ptr<f32>) {
    // COM: This test checks that rematerializable values (constants, splat, make_range)
    // COM: are excluded from register pressure.
    // COM: Every value here is rematerializable and excluded: the constant
    // COM: tensor (arith.constant), the make_range, and the splat (whose source
    // COM: %arg0 is a block argument — but the splat result itself is excluded
    // COM: because it is a splat of a value with no live cost here). The i32
    // COM: constant is also a constant. Peak = 0 bytes.
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x256xf32, #blocked>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: scalar_only_block
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 16 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 16 bytes, live-in = 0 bytes

// COM: The same function under `per-op` mode, which is the only way a lit file
// COM: can reach `pressureBefore`: it takes an operation, and the per-block
// COM: report cannot name one. All values here are i32 = 4 bytes, so the figures
// COM: are checkable by hand -- and they have to be checked, because the
// COM: whole-function peak gate in HoistLayoutConversions prices an insertion
// COM: point with this primitive and would silently mis-price it if it regressed
// COM: to `pressureAt`.
// COM: Each figure is the pressure across the program point the named operation
// COM: follows, so the sequence also pins where values die.
// PEROP-LABEL: scalar_only_block
// COM: Above the first addi only {%arg0, %arg1} exist: 8 bytes. `pressureAt`
// COM: reports 12 here, because it charges %0 at the operation defining it --
// COM: which is exactly why an insertion point needs `pressureBefore`.
// PEROP-NEXT: Before arith.addi: 8 bytes
// COM: {%arg0, %arg1, %0} all cross into the muli: 12 bytes.
// PEROP-NEXT: Before arith.muli: 12 bytes
// COM: %arg1 and %0 have their last use at the muli, so neither survives it:
// COM: only {%arg0, %1} = 8 bytes do. Dropping values that die at the preceding
// COM: operation is what makes this exact rather than merely conservative.
// PEROP-NEXT: Before arith.addi: 8 bytes
// PEROP-NEXT: Before tt.return: 4 bytes
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scalar_only_block(%arg0: i32, %arg1: i32) -> i32 {
    // COM: Scalars only (i32 = 4 bytes each). currentlyLiveValues takes an
    // COM: expansive view (a value defined-by or consumed-by the op is live), so
    // COM: peak is set at %1 = muli where {%arg0, %arg1, %0, %1} are all live =
    // COM: 4 * 4B = 16 bytes.
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg1 : i32
    %2 = arith.addi %1, %arg0 : i32
    tt.return %2 : i32
  }
}

// -----

// CHECK-LABEL: loop_with_live_in_operand
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 256 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 256 bytes, live-in = 64 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 256 bytes, live-in = 0 bytes
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_2 = #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth=1}>
#dot1_2 = #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @loop_with_live_in_operand(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    // COM: This test exercises non-zero live-in and rematerializable exclusion.
    // COM: Operand A (%3) is loaded OUTSIDE the loop and consumed INSIDE, making
    // COM: it live-in to the loop body: tensor<8x64xf16, #dot0_2> = 32 elems per
    // COM: thread * 2B = 64 bytes per thread. This should appear in live-in.
    // COM: The constant tensor %cst_add is defined outside and used inside (added
    // COM: to the dot result). If counted, it would add 32 bytes (8 elems * 4B).
    // COM: Because constants are rematerializable, %cst_add is EXCLUDED from
    // COM: live-in, so live-in = 64 bytes (only operand A).
    // COM: Loop body peak = operand A (64) + operand B (128) + accumulator (32)
    // COM: + dot result (32) = 256 bytes (peak occurs at the tt.dot operation).
    // COM: The outer func block's own top-level ops hold operand A (64 bytes); the
    // COM: loop result is unread so it adds nothing there -- but the block's peak
    // COM: covers the nested loop body too, so it reports 256 bytes.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas2>
    %cst_add = arith.constant dense<1.000000e+00> : tensor<8x16xf32, #dpas2>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %3 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0_2>
    %2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst) -> (tensor<8x16xf32, #dpas2>) : i32 {
      %4 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1_2>
      %5 = tt.dot %3, %4, %arg3, inputPrecision = tf32 : tensor<8x64xf16, #dot0_2> * tensor<64x16xf16, #dot1_2> -> tensor<8x16xf32, #dpas2>
      %6 = arith.addf %5, %cst_add : tensor<8x16xf32, #dpas2>
      scf.yield %6 : tensor<8x16xf32, #dpas2>
    }
    tt.return
  }
}

// -----


// COM: Gap 1: the peak of a block can sit inside a nested region. The loop body's
// COM: own operations are cheap, but the scf.if it holds is not, so a block-local
// COM: scan of the loop body would report the loop as low pressure.
// CHECK-LABEL: loop_with_nested_if
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 1569 bytes
// CHECK-NEXT: Block {{.*}} in scf.if: peak = 1569 bytes, live-in = 0 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 1569 bytes, live-in = 1 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 1569 bytes, live-in = 0 bytes
#dpas3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_3 = #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth=1}>
#dot1_3 = #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth=2}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @loop_with_nested_if(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i1) {
    // COM: tensor<64x128xf32, #blocked3> is 128 elems/lane * 4B = 512 bytes/lane.
    // COM: The peak is at the arith.addf inside the scf.if, where the two loaded
    // COM: tensors and the sum are all live: 3 * 512 = 1536 bytes. On top of that
    // COM: the analysis adds the values live *through* the enclosing scf.if: the
    // COM: dot result %6, yielded after the scf.if (32), and the condition %arg3
    // COM: (1 byte, i1). 1536 + 32 + 1 = 1569 bytes. %arg3 is live at both nesting
    // COM: levels and is counted once, not twice. The scf.for's own result %3 is
    // COM: not counted while its body executes: that register is the one holding
    // COM: the yielded %6, already counted above.
    // COM: All three blocks report 1569: a block's peak covers nested regions.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas3>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %2 = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x128xf32>
    %3 = scf.for %arg4 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg5 = %cst) -> (tensor<8x16xf32, #dpas3>) : i32 {
      %4 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0_3>
      %5 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1_3>
      %6 = tt.dot %4, %5, %arg5, inputPrecision = tf32 : tensor<8x64xf16, #dot0_3> * tensor<64x16xf16, #dot1_3> -> tensor<8x16xf32, #dpas3>
      scf.if %arg3 {
        %7 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked3>
        %8 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked3>
        %9 = arith.addf %7, %8 : tensor<64x128xf32, #blocked3>
        tt.descriptor_store %2[%c0_i32, %c0_i32], %9 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked3>
      }
      scf.yield %6 : tensor<8x16xf32, #dpas3>
    }
    tt.return
  }
}

// -----

// COM: Gap 2, on top of Gap 1: a value defined before a region-holding op and
// COM: used after it holds a register for the whole duration of that op, yet is
// COM: neither defined nor used inside the nested block, so the nested block's
// COM: own liveness info does not know about it.
// CHECK-LABEL: loop_with_nested_if_and_live_through
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 2081 bytes
// CHECK-NEXT: Block {{.*}} in scf.if: peak = 2081 bytes, live-in = 0 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 2081 bytes, live-in = 1 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 2081 bytes, live-in = 0 bytes
#dpas4 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_4 = #ttg.dot_op<{opIdx = 0, parent = #dpas4, kWidth=1}>
#dot1_4 = #ttg.dot_op<{opIdx = 1, parent = #dpas4, kWidth=2}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @loop_with_nested_if_and_live_through(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i1) {
    // COM: Identical to @loop_with_nested_if except for %7, a 512 byte/lane value
    // COM: defined immediately before the scf.if and consumed immediately after
    // COM: it, and never referenced inside it. The peak is at the same arith.addf
    // COM: inside the scf.if and grows by exactly 512 bytes, from 1569 to 2081:
    // COM: the live-through value is added once, not once per nesting level.
    // COM: %7 is invisible to the scf.if body block's own liveness info (neither
    // COM: defined nor used there), so without the enclosing-scope walk the body
    // COM: would still report 1569 despite %7 occupying a register throughout.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas4>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %2 = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x128xf32>
    %3 = scf.for %arg4 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg5 = %cst) -> (tensor<8x16xf32, #dpas4>) : i32 {
      %4 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0_4>
      %5 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1_4>
      %6 = tt.dot %4, %5, %arg5, inputPrecision = tf32 : tensor<8x64xf16, #dot0_4> * tensor<64x16xf16, #dot1_4> -> tensor<8x16xf32, #dpas4>
      %7 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked4>
      scf.if %arg3 {
        %10 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked4>
        %11 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked4>
        %12 = arith.addf %10, %11 : tensor<64x128xf32, #blocked4>
        tt.descriptor_store %2[%c0_i32, %c0_i32], %12 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked4>
      }
      %8 = arith.addf %7, %7 : tensor<64x128xf32, #blocked4>
      tt.descriptor_store %2[%c0_i32, %c0_i32], %8 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked4>
      scf.yield %6 : tensor<8x16xf32, #dpas4>
    }
    tt.return
  }
}

// -----

// COM: Three nesting levels, with a live-through value contributed by each of
// COM: them: the peak inside the innermost block must gather live-through values
// COM: from every enclosing scope, each counted exactly once.
// CHECK-LABEL: three_level_nesting_live_through
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 2594 bytes
// CHECK-NEXT: Block {{.*}} in scf.if: peak = 2594 bytes, live-in = 0 bytes
// CHECK-NEXT: Block {{.*}} in scf.if: peak = 2594 bytes, live-in = 1 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 2594 bytes, live-in = 2 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 2594 bytes, live-in = 0 bytes

// COM: A handful of the boundary crossings above under `per-op` mode, chosen to
// COM: catch exactly the kind of drift BLOCKING-1 introduced: `pressureBefore`
// COM: at a nested-region boundary must track what the block-peak figures above
// COM: already assert, not recompute liveness some other, potentially
// COM: inconsistent way. Not adjacent output lines, so plain `PEROP:` (not
// COM: `-NEXT`) is used for all but the label.
// PEROP-LABEL: three_level_nesting_live_through
// COM: Entering the second nesting level: the outer scf.if's own live-through
// COM: contribution (the loop body's %7, 512 bytes) adds to what already lived
// COM: through the loop (%arg3/%arg4 at 1 byte each, %6 at 32).
// PEROP: Before scf.if: 546 bytes
// COM: Entering the third nesting level: the inner scf.if's own live-through
// COM: contribution (the outer if body's %10, 512 bytes) adds on top.
// PEROP: Before scf.if: 1058 bytes
// COM: Back at the outer scf.if's own nesting level, after the inner scf.if has
// COM: closed: the inner if's live-through contribution is gone, but the outer
// COM: if's own (%7, 512 bytes) is not -- exactly the region-boundary
// COM: bookkeeping BLOCKING-1 got wrong.
// PEROP: Before arith.addf: 1058 bytes
// COM: Back in the loop body, after the outer scf.if has closed entirely: only
// COM: the loop's own live-through value (%6 at 32) and the two conditions
// COM: remain, none of either scf.if's own contributions.
// PEROP: Before scf.yield: 34 bytes
#dpas5 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_5 = #ttg.dot_op<{opIdx = 0, parent = #dpas5, kWidth=1}>
#dot1_5 = #ttg.dot_op<{opIdx = 1, parent = #dpas5, kWidth=2}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @three_level_nesting_live_through(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i1, %arg4: i1) {
    // COM: The peak is at the arith.addf in the innermost scf.if body, three
    // COM: levels down: 3 x 512 for its own values, plus one live-through value
    // COM: from each enclosing scope (%10 at 512 from the outer scf.if body, %7 at
    // COM: 512 from the loop body, %6 at 32 carried out of the loop) plus both i1
    // COM: conditions at 1 byte each, for 2594 bytes. Each is counted once even
    // COM: though the conditions and %6 are live at several nesting levels.
    // COM: %arg3/%arg4 ARE operands of the scf.if they each condition, and are
    // COM: each dropped by #8053's operand-supersession fix at that scf.if's own
    // COM: level (see getLiveThroughAncestorSet), since neither is touched inside
    // COM: its scf.if's body and neither is live immediately after it. But each is
    // COM: separately, correctly recovered one level up: its only real use is
    // COM: nested inside the scf.for's region too, so it is also raw-live at the
    // COM: scf.for's own point, where it is NOT an operand (the scf.for's operands
    // COM: are only its bounds and the accumulator init) and so is kept there
    // COM: unconditionally. The recursive ancestor union re-unions that scf.for-
    // COM: level contribution back in, so the byte the fix removes at the scf.if
    // COM: level is restored at the scf.for level and the total does not move.
    // COM: By the time %15 executes both conditions have already branched, so
    // COM: charging them here at all is a ~2-byte overcount -- but it is *not*
    // COM: pre-existing or unrelated to the fix above: it is the very same
    // COM: back-edge-unconditional charge at the scf.for level that recovers
    // COM: %arg3/%arg4's contribution in the first place (a value merely
    // COM: *referenced* inside a loop is charged for the loop's whole duration,
    // COM: since MLIR's own per-block liveness cannot see the back edge). Here
    // COM: that same rule overcounts by ~2 bytes; at the scale of a value that
    // COM: is only read once early in a much larger loop body, the identical
    // COM: mechanism is what costs 2048 B/lane in the cross-loop-forwarding
    // COM: case elsewhere in this file. Both are the same accepted, deliberate
    // COM: trade-off (soundness over precision), not two different defects.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas5>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %2 = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x128xf32>
    %3 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg6 = %cst) -> (tensor<8x16xf32, #dpas5>) : i32 {
      %4 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0_5>
      %5 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1_5>
      %6 = tt.dot %4, %5, %arg6, inputPrecision = tf32 : tensor<8x64xf16, #dot0_5> * tensor<64x16xf16, #dot1_5> -> tensor<8x16xf32, #dpas5>
      %7 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked5>
      scf.if %arg3 {
        %10 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked5>
        scf.if %arg4 {
          %13 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked5>
          %14 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked5>
          %15 = arith.addf %13, %14 : tensor<64x128xf32, #blocked5>
          tt.descriptor_store %2[%c0_i32, %c0_i32], %15 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked5>
        }
        %11 = arith.addf %10, %10 : tensor<64x128xf32, #blocked5>
        tt.descriptor_store %2[%c0_i32, %c0_i32], %11 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked5>
      }
      %8 = arith.addf %7, %7 : tensor<64x128xf32, #blocked5>
      tt.descriptor_store %2[%c0_i32, %c0_i32], %8 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked5>
      scf.yield %6 : tensor<8x16xf32, #dpas5>
    }
    tt.return
  }
}

// -----

// COM: A loop nested in a loop, the scenario issue #8053 calls out for real
// consumers: both loop bodies carry an iter_arg, and a value live-in to the
// outer loop is consumed inside the inner loop body.
// CHECK-LABEL: nested_loops_live_through
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 2080 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 2080 bytes, live-in = 512 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 2080 bytes, live-in = 512 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 2080 bytes, live-in = 0 bytes
#dpas6 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_6 = #ttg.dot_op<{opIdx = 0, parent = #dpas6, kWidth=1}>
#dot1_6 = #ttg.dot_op<{opIdx = 1, parent = #dpas6, kWidth=2}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @nested_loops_live_through(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>) {
    // COM: tensor<64x128xf32, #blocked6> is 512 bytes/lane, tensor<8x16xf32,
    // COM: #dpas6> is 32. The peak is at %14, the second arith.addf in the inner
    // COM: loop body, and is 2080 bytes:
    // COM:   inner block, at %14: %12, %13 and %14 itself, 3 x 512 = 1536. The
    // COM:     inner iter_arg %arg8 is NOT live here: it is a block argument, so
    // COM:     its range runs from the block front to its last use at %12.
    // COM:   enclosing outer loop body, at the inner scf.for %9: the dot result
    // COM:     %7 (32), which is live across the inner loop because scf.yield uses
    // COM:     it afterwards, plus %3 (512): %3's only use is inside the inner
    // COM:     region, which liveness attributes to the scf.for op itself. That is
    // COM:     the term the enclosing-scope walk exists for -- inside the inner
    // COM:     block %3 looks dead from %13 onward, yet it must stay in a register
    // COM:     for every inner iteration. The loop-carried init %8 is NOT counted
    // COM:     here: %8's only use anywhere is as %9's own init operand, so it dies
    // COM:     the instant %9 starts executing -- from then on the register it named
    // COM:     is either %arg8 (iteration 0, already counted by the inner block's
    // COM:     own live set) or the yielded %14 (later iterations, likewise already
    // COM:     counted). Charging %8 as well would count that one physical register
    // COM:     twice (see getLiveThroughAncestorSet's operand-supersession check).
    // COM:   enclosing tt.func block, at the outer scf.for %4: %3 again (consumed
    // COM:     by %4), already in the set, so it adds nothing. The tensor
    // COM:     descriptors contribute 0 bytes and %cst is rematerializable.
    // COM: 1536 + 32 + 512 = 2080. All three blocks report it.
    // COM: Live-in is 512 for both loop bodies (%3, the only non-rematerializable
    // COM: non-descriptor value defined outside and used inside); the iter_args do
    // COM: not appear because LivenessBlockInfo::in() excludes block arguments.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas6>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %2 = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x128xf32>
    %3 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked6>
    %4 = scf.for %arg5 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg6 = %cst) -> (tensor<8x16xf32, #dpas6>) : i32 {
      %5 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0_6>
      %6 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1_6>
      %7 = tt.dot %5, %6, %arg6, inputPrecision = tf32 : tensor<8x64xf16, #dot0_6> * tensor<64x16xf16, #dot1_6> -> tensor<8x16xf32, #dpas6>
      %8 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked6>
      %9 = scf.for %arg7 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg8 = %8) -> (tensor<64x128xf32, #blocked6>) : i32 {
        %12 = arith.addf %arg8, %3 : tensor<64x128xf32, #blocked6>
        %13 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked6>
        %14 = arith.addf %12, %13 : tensor<64x128xf32, #blocked6>
        scf.yield %14 : tensor<64x128xf32, #blocked6>
      }
      %10 = arith.addf %9, %9 : tensor<64x128xf32, #blocked6>
      tt.descriptor_store %2[%c0_i32, %c0_i32], %10 : !tt.tensordesc<64x128xf32>, tensor<64x128xf32, #blocked6>
      scf.yield %7 : tensor<8x16xf32, #dpas6>
    }
    tt.return
  }
}

// -----


// COM: Originally meant as the hazard the whole-function peak line exists to
// COM: expose, measured on the same function before and after hoisting the
// COM: conversion by hand. With the region-aware register pressure fix, it no
// COM: longer demonstrates that hazard: both figures below actually *fall*
// COM: (loop body 8320 -> 6272, whole-function 8320 -> 7296), so there is no
// COM: rise for the whole-function line to catch that the loop-level figure
// COM: alone would miss -- hoisting is a strict win here by either metric now.
// COM: This is a factual correction of a stale claim (a prior version of this
// COM: comment said the whole-function figure "rises (6272 -> 7296)", citing
// COM: the wrong pre-hoist number -- 6272 is `no_spanning_in_loop`'s loop-body
// COM: figure below, not this case's whole-function one, which the CHECK lines
// COM: two cases down have always correctly asserted as 8320). The underlying
// COM: design question -- does this repo still have a case demonstrating the
// COM: whole-function-only hazard #8068's gate exists for -- is answered by
// COM: `gate_needed_in_loop`/`gate_needed_hoisted` at the end of this file: a
// COM: straight-line chain of temporaries between the conversion's source and
// COM: the loop, rather than a single value read only after it, is what
// COM: creates the asymmetry this pair no longer can.
// COM:
// COM: %span is computed above the loop and read below it, so it is live across
// COM: the whole straight-line region -- and the hoisted conversion's result now
// COM: has to be live there too, alongside it. The loop-level figure alone still
// COM: cannot see this: hoisting *lowers* the loop body peak (8320 -> 6272),
// COM: because the conversion no longer executes there -- though not down to a
// COM: convert-free loop's 4224, since #8053's enclosing-scope accounting now
// COM: also charges the loop body for %span (and, pre-hoist, the unconverted
// COM: %arg0 too): each is read only once, right next to the loop, but is
// COM: charged as occupying a register for the loop's *entire* duration, since
// COM: the loop body's own liveness info cannot see that the real last use sits
// COM: just outside it.
#blockedspan = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasspan = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_aspan = #ttg.dot_op<{opIdx = 0, parent = #dpasspan, kWidth = 1}>
#dot_bspan = #ttg.dot_op<{opIdx = 1, parent = #dpasspan, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: Loop body peak (8320) = the dot's own local pressure (converted A +
  // COM: B + accumulator-in + dot result = 4224, same as a convert-free loop)
  // COM: plus two enclosing-scope live-through values the loop body's own
  // COM: liveness cannot see: %arg0 (2048, still read once inside the loop by
  // COM: the convert, but charged for the loop's whole duration) and %span
  // COM: (2048, read only after the loop, at tt.return). 4224 + 2048 + 2048 =
  // COM: 8320.
  // CHECK-LABEL: spanning_in_loop
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 8320 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 8320 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 8320 bytes, live-in = 0 bytes
  tt.func @spanning_in_loop(%arg0: tensor<256x64xf16, #blockedspan>, %arg1: tensor<64x16xf16, #dot_bspan>, %arg2: tensor<256x16xf32, #dpasspan>) -> (tensor<256x16xf32, #dpasspan>, tensor<256x64xf16, #blockedspan>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %span = arith.addf %arg0, %arg0 : tensor<256x64xf16, #blockedspan>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpasspan>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blockedspan> -> tensor<256x64xf16, #dot_aspan>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_aspan> * tensor<64x16xf16, #dot_bspan> -> tensor<256x16xf32, #dpasspan>
      scf.yield %dot : tensor<256x16xf32, #dpasspan>
    }
    tt.return %result, %span : tensor<256x16xf32, #dpasspan>, tensor<256x64xf16, #blockedspan>
  }

  // COM: Loop body peak (6272) = the dot's own local pressure (4224, the
  // COM: converted A is now a live-in rather than freshly produced, but the
  // COM: total is the same) plus %span (2048), the one remaining live-through
  // COM: value -- the converted A itself is no longer an extra live-through
  // COM: charge, since it is already live-in to the loop body directly.
  // COM: 4224 + 2048 = 6272. The whole-function peak (7296) sits elsewhere
  // COM: entirely, at the hoisted convert itself, where %arg0 (2048, still
  // COM: live for its own use right here), %arg1 (128) and the accumulator
  // COM: init %arg2 (1024) are all live-through the loop alongside %span
  // COM: (2048) and the convert's own result (2048): 2048+128+1024+2048+2048
  // COM: = 7296, unaffected by #8053 since it is already a single block-local
  // COM: point.
  // CHECK-LABEL: spanning_hoisted
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 7296 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 6272 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 7296 bytes, live-in = 0 bytes
  tt.func @spanning_hoisted(%arg0: tensor<256x64xf16, #blockedspan>, %arg1: tensor<64x16xf16, #dot_bspan>, %arg2: tensor<256x16xf32, #dpasspan>) -> (tensor<256x16xf32, #dpasspan>, tensor<256x64xf16, #blockedspan>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %span = arith.addf %arg0, %arg0 : tensor<256x64xf16, #blockedspan>
    %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blockedspan> -> tensor<256x64xf16, #dot_aspan>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpasspan>) : i32 {
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_aspan> * tensor<64x16xf16, #dot_bspan> -> tensor<256x16xf32, #dpasspan>
      scf.yield %dot : tensor<256x16xf32, #dpasspan>
    }
    tt.return %result, %span : tensor<256x16xf32, #dpasspan>, tensor<256x64xf16, #blockedspan>
  }
}

// -----

// COM: The same A/B on the shape that has no spanning value, which is the shape
// COM: issue #7993 claimed raises the whole-function peak on its own. It does
// COM: not: with nothing spanning the hoist site, hoisting cannot raise the
// COM: figure the way it does for the spanning pair above. But #8053's
// COM: enclosing-scope accounting adds a different, narrower effect here:
// COM: keeping the convert *inside* the loop conservatively charges the loop
// COM: for %arg0's whole duration (it is read once, by the convert, but the
// COM: loop body's own liveness cannot see that the real last use sits inside
// COM: it rather than after it), so the in-loop peak (6272) is actually
// COM: *higher* than the hoisted one (5248) -- hoisting *lowers* the peak here,
// COM: the opposite direction from the spanning pair, because there is no
// COM: spanning value to raise it and one fewer enclosing-scope charge once the
// COM: convert moves outside the loop.
#blockednospan = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasnospan = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_anospan = #ttg.dot_op<{opIdx = 0, parent = #dpasnospan, kWidth = 1}>
#dot_bnospan = #ttg.dot_op<{opIdx = 1, parent = #dpasnospan, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: Loop body peak (6272) = the dot's own local pressure (4224) plus
  // COM: %arg0 (2048), charged as live through the whole loop even though its
  // COM: one real use (the convert) is the loop's first op. 4224 + 2048 = 6272.
  // CHECK-LABEL: no_spanning_in_loop
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 6272 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 6272 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 6272 bytes, live-in = 0 bytes
  tt.func @no_spanning_in_loop(%arg0: tensor<256x64xf16, #blockednospan>, %arg1: tensor<64x16xf16, #dot_bnospan>, %arg2: tensor<256x16xf32, #dpasnospan>) -> tensor<256x16xf32, #dpasnospan> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpasnospan>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blockednospan> -> tensor<256x64xf16, #dot_anospan>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_anospan> * tensor<64x16xf16, #dot_bnospan> -> tensor<256x16xf32, #dpasnospan>
      scf.yield %dot : tensor<256x16xf32, #dpasnospan>
    }
    tt.return %result : tensor<256x16xf32, #dpasnospan>
  }

  // CHECK-LABEL: no_spanning_hoisted
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 5248 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 4224 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 5248 bytes, live-in = 0 bytes
  tt.func @no_spanning_hoisted(%arg0: tensor<256x64xf16, #blockednospan>, %arg1: tensor<64x16xf16, #dot_bnospan>, %arg2: tensor<256x16xf32, #dpasnospan>) -> tensor<256x16xf32, #dpasnospan> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blockednospan> -> tensor<256x64xf16, #dot_anospan>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpasnospan>) : i32 {
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_anospan> * tensor<64x16xf16, #dot_bnospan> -> tensor<256x16xf32, #dpasnospan>
      scf.yield %dot : tensor<256x16xf32, #dpasnospan>
    }
    tt.return %result : tensor<256x16xf32, #dpasnospan>
  }
}

// -----

// COM: The hazard `spanning_in_loop`/`spanning_hoisted` above no longer demonstrates
// COM: under the corrected analysis (see the note there): a hoist whose loop-level
// COM: peak does not rise, but whose *whole-function* peak does, so a gate that only
// COM: consulted the loop-level figure would take a hoist the whole-function one
// COM: correctly rejects. This is what #8068's whole-function peak gate exists for.
// COM:
// COM: The shape: %span feeds the conversion inside the loop and is also read again
// COM: after it (tt.return), same as `spanning_in_loop` above, so its own live-through
// COM: contribution is identical whether or not the conversion is hoisted -- it cannot
// COM: create an asymmetry on its own (this is exactly why the case above stopped
// COM: working: %span there contributes the same either way too). The asymmetry
// COM: instead comes from a straight-line chain of temporaries (%big1..%big3) sitting
// COM: *between* the conversion's source and the loop, built so %big3 is also read
// COM: after the loop (giving the chain a real peak of its own, at %big3's own point,
// COM: rather than being dead code once computed).
// COM:
// COM: Pre-hoist, the conversion exists only inside the loop and never touches that
// COM: chain, so the chain's own peak is unaffected by it: 2048 (%span) + 128 (%arg1)
// COM: + 1024 (%arg2) + 1024 (%big1) + 1024 (%big2) + 1024 (%big3) = 6272, below the
// COM: loop's own peak (7296), so the loop dominates and the whole-function figure
// COM: (7296) equals it exactly.
// COM:
// COM: Post-hoist, the conversion's result is defined *before* the chain and used
// COM: only inside the loop (after it), so it is live-through the entire chain too --
// COM: a charge the pre-hoist version never had at all: 6272 + 2048 (the conversion's
// COM: own result) = 8320, now *above* the loop's own peak. The loop's own peak is
// COM: unchanged at 7296 either way (the conversion's own contribution there is the
// COM: same 2048 bytes whether it is locally defined or live-through), so a gate that
// COM: only looked at the loop would see no reason to reject this hoist -- only the
// COM: whole-function figure (7296 -> 8320) shows the real rise. Note the *visible*
// COM: rise (1024) is smaller than the conversion's own size (2048): part of the new
// COM: charge is masked by the loop's pre-existing, unrelated 7296 ceiling, and only
// COM: the portion that pushes past it is observable in the whole-function peak --
// COM: an honest artifact of "peak", not a miscount.
#blockedgate = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasgate = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_agate = #ttg.dot_op<{opIdx = 0, parent = #dpasgate, kWidth = 1}>
#dot_bgate = #ttg.dot_op<{opIdx = 1, parent = #dpasgate, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: gate_needed_in_loop
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 7296 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 7296 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 7296 bytes, live-in = 0 bytes
  tt.func @gate_needed_in_loop(%arg0: tensor<256x64xf16, #blockedgate>, %arg1: tensor<64x16xf16, #dot_bgate>, %arg2: tensor<256x16xf32, #dpasgate>, %big0: tensor<256x16xf32, #dpasgate>) -> (tensor<256x16xf32, #dpasgate>, tensor<256x64xf16, #blockedgate>, tensor<256x16xf32, #dpasgate>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %span = arith.addf %arg0, %arg0 : tensor<256x64xf16, #blockedgate>
    %big1 = arith.addf %big0, %big0 : tensor<256x16xf32, #dpasgate>
    %big2 = arith.addf %big1, %big0 : tensor<256x16xf32, #dpasgate>
    %big3 = arith.addf %big2, %big1 : tensor<256x16xf32, #dpasgate>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpasgate>) : i32 {
      %cvt = ttg.convert_layout %span : tensor<256x64xf16, #blockedgate> -> tensor<256x64xf16, #dot_agate>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_agate> * tensor<64x16xf16, #dot_bgate> -> tensor<256x16xf32, #dpasgate>
      scf.yield %dot : tensor<256x16xf32, #dpasgate>
    }
    tt.return %result, %span, %big3 : tensor<256x16xf32, #dpasgate>, tensor<256x64xf16, #blockedgate>, tensor<256x16xf32, #dpasgate>
  }

  // COM: Same shape, conversion hoisted above the chain by hand. Loop-level peak is
  // COM: unchanged (7296); whole-function peak rises (7296 -> 8320) because the
  // COM: hoisted result now spans the chain too. See the comment above this pair.
  // CHECK-LABEL: gate_needed_hoisted
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 8320 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 7296 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 8320 bytes, live-in = 0 bytes
  tt.func @gate_needed_hoisted(%arg0: tensor<256x64xf16, #blockedgate>, %arg1: tensor<64x16xf16, #dot_bgate>, %arg2: tensor<256x16xf32, #dpasgate>, %big0: tensor<256x16xf32, #dpasgate>) -> (tensor<256x16xf32, #dpasgate>, tensor<256x64xf16, #blockedgate>, tensor<256x16xf32, #dpasgate>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %span = arith.addf %arg0, %arg0 : tensor<256x64xf16, #blockedgate>
    %cvt = ttg.convert_layout %span : tensor<256x64xf16, #blockedgate> -> tensor<256x64xf16, #dot_agate>
    %big1 = arith.addf %big0, %big0 : tensor<256x16xf32, #dpasgate>
    %big2 = arith.addf %big1, %big0 : tensor<256x16xf32, #dpasgate>
    %big3 = arith.addf %big2, %big1 : tensor<256x16xf32, #dpasgate>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpasgate>) : i32 {
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_agate> * tensor<64x16xf16, #dot_bgate> -> tensor<256x16xf32, #dpasgate>
      scf.yield %dot : tensor<256x16xf32, #dpasgate>
    }
    tt.return %result, %span, %big3 : tensor<256x16xf32, #dpasgate>, tensor<256x64xf16, #blockedgate>, tensor<256x16xf32, #dpasgate>
  }
}
