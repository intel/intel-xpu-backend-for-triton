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
// needs to occupy a register. That is why the outer block below reports 0 rather
// than the 32 bytes of the scf.for result -- nothing consumes it.
// CHECK-LABEL: loop_with_dpas_accumulator
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Peak over all blocks in tt.func: 256 bytes
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 256 bytes, live-in = 0 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 0 bytes, live-in = 0 bytes
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
    // COM: The outer func block holds only the scf.for result, which tt.return
    // COM: does not take, so nothing reads it and it costs 0 bytes.
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
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 64 bytes, live-in = 0 bytes
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
    // COM: Outer func block peak = operand A (64); the loop result is unread, so
    // COM: it adds nothing.
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

// COM: The hazard the whole-function peak line exists to expose, measured on the
// COM: same function before and after hoisting the conversion by hand.
// COM:
// COM: %span is computed above the loop and read below it, so it is live across
// COM: the whole straight-line region -- and the hoisted conversion's result now
// COM: has to be live there too, alongside it. The loop-level figure cannot see
// COM: this: hoisting *lowers* the loop body peak (5248 -> 4224), because the
// COM: conversion no longer executes there, while the whole-function peak
// COM: *rises* (6272 -> 7296). A gate that consults only the loop body would
// COM: read the fall and take the hoist.
#blockedspan = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasspan = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_aspan = #ttg.dot_op<{opIdx = 0, parent = #dpasspan, kWidth = 1}>
#dot_bspan = #ttg.dot_op<{opIdx = 1, parent = #dpasspan, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: spanning_in_loop
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 6272 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 5248 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 6272 bytes, live-in = 0 bytes
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

  // CHECK-LABEL: spanning_hoisted
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 7296 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 4224 bytes, live-in = 2176 bytes
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
// COM: issue #7993 claimed raises the whole-function peak. It does not: the peak
// COM: is 5248 bytes either way. Hoisting moves *where* the peak is -- out of the
// COM: loop body and into the straight-line region above it -- but the loop's own
// COM: live-ins are the only thing the conversion's result has to coexist with,
// COM: and it displaces the source it converts. Only a value that spans the hoist
// COM: site, as in the pair above, actually pushes the figure up.
#blockednospan = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpasnospan = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_anospan = #ttg.dot_op<{opIdx = 0, parent = #dpasnospan, kWidth = 1}>
#dot_bnospan = #ttg.dot_op<{opIdx = 1, parent = #dpasnospan, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: no_spanning_in_loop
  // CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
  // CHECK-NEXT: Peak over all blocks in tt.func: 5248 bytes
  // CHECK-NEXT: Block {{.*}} in scf.for: peak = 5248 bytes, live-in = 2176 bytes
  // CHECK-NEXT: Block {{.*}} in tt.func: peak = 4224 bytes, live-in = 0 bytes
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
