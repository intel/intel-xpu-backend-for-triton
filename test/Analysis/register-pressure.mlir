// RUN: triton-opt %s -split-input-file -test-register-pressure | FileCheck %s
// RUN: triton-opt %s -split-input-file -test-register-pressure=per-op | FileCheck %s --check-prefix=PEROP

// The test pass prints the bare function name, then the analysis report, then
// the IR. Anchor CHECK-LABEL on the pass's bare-name line so the report checks
// are contiguous.
//
// The first report line is the peak over *every* block, which a per-kernel
// allocation must cover; the per-block lines follow.
//
// A value nothing reads contributes nothing to any of these figures. A block's
// peak includes the regions nested in it, so the outer tt.func block below
// reports its loop body's peak.
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
    // COM: The scf.for result is unread, so the func block's own ops hold nothing,
    // COM: but its peak still includes the loop body's 256 bytes.
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
    // COM: The func block's own ops hold operand A (64 bytes), but its peak
    // COM: includes the loop body's 256 bytes.
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


// COM: The peak of a block can sit inside a nested region: the loop body's own
// COM: ops are cheap, the scf.if it holds is not.
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
    // COM: Peak at the arith.addf in the scf.if: 3 * 512 = 1536, plus the values
    // COM: live through the if: the dot result %6 (32, yielded after the if) and
    // COM: the condition %arg3 (1) = 1569. A value live at several nesting levels
    // COM: is counted once. The scf.for's own result is not counted while its body
    // COM: runs. All three blocks report 1569.
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

// COM: A value defined before a region-holding op and used after it holds a
// COM: register for the whole op, yet the nested block's own liveness does not
// COM: know it.
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
    // COM: As @loop_with_nested_if plus %7 (512), defined before the scf.if and
    // COM: read after it, never inside it. The body's own liveness cannot see %7,
    // COM: yet the peak must include it, once: 1569 + 512 = 2081.
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

// COM: Per-op figures at the nested-region boundaries must agree with the block
// COM: peaks above. The output lines are not adjacent, so they are not chained
// COM: with -NEXT.
// PEROP-LABEL: three_level_nesting_live_through
// COM: Entering the second level: the outer scf.if's live-through value (the loop
// COM: body's %7, 512) adds to what lived through the loop (%arg3/%arg4 at 1 byte
// COM: each, %6 at 32).
// PEROP: Before scf.if: 546 bytes
// COM: Entering the third level: the inner scf.if's live-through value (the outer
// COM: if body's %10, 512) adds on top.
// PEROP: Before scf.if: 1058 bytes
// COM: After the inner scf.if closes, its live-through value is gone but the
// COM: outer if's (%7, 512) is not.
// PEROP: Before arith.addf: 1058 bytes
// COM: After the outer scf.if closes, only the loop's %6 (32) and the two
// COM: conditions remain.
// PEROP: Before scf.yield: 34 bytes
#dpas5 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_5 = #ttg.dot_op<{opIdx = 0, parent = #dpas5, kWidth=1}>
#dot1_5 = #ttg.dot_op<{opIdx = 1, parent = #dpas5, kWidth=2}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @three_level_nesting_live_through(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>, %arg3: i1, %arg4: i1) {
    // COM: Peak at the arith.addf in the innermost scf.if body, three levels down:
    // COM: 3 x 512, plus one live-through value per enclosing scope (%10 512, %7
    // COM: 512, %6 32) and both i1 conditions = 2594. The conditions are still
    // COM: charged after they branch: a value read anywhere in a loop is charged
    // COM: for the whole loop, since per-block liveness cannot see the back edge.
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

// COM: A loop nested in a loop: a value live-in to the outer loop is read only
// COM: inside the inner loop body.
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
    // COM: #dpas6> is 32. Peak at %14 = 2080: %12, %13, %14 (3 x 512 = 1536), plus
    // COM: %7 (32, read after the inner loop), plus %3 (512, read only inside the
    // COM: inner loop, so live for its whole duration). The inner init %8 dies when
    // COM: the inner loop starts and is not charged again.
    // COM: Live-in is 512 for both bodies (%3); block arguments are not in
    // COM: LivenessBlockInfo::in().
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

// COM: %v is used only as the inner loop's init. It is superseded for the inner
// COM: loop, but not for the outer one: the outer back edge re-evaluates that init
// COM: each iteration, so the outer body is charged for %v. Peak 1056 = %arg8 512
// COM: + %arg6 32 + %v 512; dropping %v at the outer level gives 544. Live-in
// COM: does not depend on this rule, so the peak is the field that pins it.
#blocked8 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas8 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: outer_loop_value_forwarded_only_to_inner_init
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 1056 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 1056 bytes, live-in = 0 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 1056 bytes, live-in = 512 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 1056 bytes, live-in = 0 bytes
  tt.func @outer_loop_value_forwarded_only_to_inner_init(%arg0: !tt.ptr<f32>) {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas8>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f32>, <64x128xf32>
    %v = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32, #blocked8>
    %outer = scf.for %arg5 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg6 = %cst) -> (tensor<8x16xf32, #dpas8>) : i32 {
      %inner = scf.for %arg7 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg8 = %v) -> (tensor<64x128xf32, #blocked8>) : i32 {
        scf.yield %arg8 : tensor<64x128xf32, #blocked8>
      }
      %11 = arith.addf %arg6, %arg6 : tensor<8x16xf32, #dpas8>
      scf.yield %11 : tensor<8x16xf32, #dpas8>
    }
    tt.return
  }
}

// -----


// COM: %span is computed above the loop and read below it. Hoisting the
// COM: conversion lowers both the body peak (8320 to 6272, the conversion leaves
// COM: the body) and the function peak (8320 to 7296). The body is still charged
// COM: for %span, and before the hoist for %arg0, for the whole loop.
#blockedspan = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasspan = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_aspan = #ttg.dot_op<{opIdx = 0, parent = #dpasspan, kWidth = 1}>
#dot_bspan = #ttg.dot_op<{opIdx = 1, parent = #dpasspan, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: Body peak 8320 = the dot's own 4224 (A 2048 + B 128 + accumulator 1024 +
  // COM: result 1024) + %arg0 2048 (read by the convert, charged for the whole
  // COM: loop) + %span 2048 (read after the loop).
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

  // COM: Body peak 6272 = 4224 + %span 2048. The function peak 7296 is at the
  // COM: hoisted convert: %arg0 2048, %arg1 128, %arg2 1024, %span 2048, result
  // COM: 2048.
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

// COM: The same A/B with nothing spanning the hoist site. Keeping the convert in
// COM: the loop charges the body for %arg0 for the whole loop, so hoisting lowers
// COM: the peak (6272 to 5248).
#blockednospan = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasnospan = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_anospan = #ttg.dot_op<{opIdx = 0, parent = #dpasnospan, kWidth = 1}>
#dot_bnospan = #ttg.dot_op<{opIdx = 1, parent = #dpasnospan, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: Body peak 6272 = 4224 + %arg0 2048, charged for the whole loop although
  // COM: its one use (the convert) is the loop's first op.
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

// COM: A hoist that leaves the loop peak unchanged (7296) but raises the function
// COM: peak (7296 to 8320): the hoisted result spans the %big1..%big3 chain
// COM: between the source and the loop. A gate that read only the loop figure
// COM: would accept it. The chain peaks at 6272 before the hoist (%span 2048 +
// COM: %arg1 128 + %arg2 1024 + three 1024 temporaries) and at 8320 with the
// COM: result live through it; the visible rise (1024) is smaller than the
// COM: result (2048) because the loop's own 7296 masks part of it.
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
