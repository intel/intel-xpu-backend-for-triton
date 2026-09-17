// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s --check-prefixes=CHECK,KEEP2,KEEP3,KEEP4
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=128 -cse | FileCheck %s --check-prefixes=CHECK,SINK2,SINK3,SINK4
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=256 -cse | FileCheck %s --check-prefixes=CHECK,SINK2,KEEP3,KEEP4
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=auto -cse | FileCheck %s --check-prefixes=CHECK,KEEP2,KEEP3,KEEP4
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=512 -cse | FileCheck %s --check-prefixes=CHECK,KEEP2,KEEP3,KEEP4

// COM: A loop-invariant 2D load is sunk into the loop when the loop body's *peak*
// COM: register pressure is at or above the per-lane GRF budget of the selected GRF
// COM: mode. The peaks quoted below are not hand-derived: they are what
// COM: `triton-opt <this file> -split-input-file --test-register-pressure` reports
// COM: for the `scf.for` body block.
// COM:
// COM: With "ttg.threads-per-warp" = 16 the per-lane budgets are:
// COM:   128           -> 4096 B/thread / 16 = 256 B/lane
// COM:   256           -> 8192 B/thread / 16 = 512 B/lane
// COM:   default / auto / 512 -> 16384 B/thread / 16 = 1024 B/lane
// COM:
// COM: "default"/"auto" match "512", not "128": the true GRF size isn't known at
// COM: this point in the pipeline, and this gate treats the budget as a threshold
// COM: to sink rather than a ceiling on what may be added, so the safe assumption
// COM: under uncertainty is the *largest* GRF size the device supports (assuming
// COM: the smallest would make the pass sink more than the real hardware, once
// COM: known, would ever have required -- see RegisterPressureAnalysis's
// COM: UnknownGRFSizeAssumption).
// COM:
// COM: The three modules have peaks of 2564, 644 and 388 B/lane, one per bucket, so
// COM: together they pin every budget boundary. The peaks deliberately sit inside
// COM: their bucket rather than exactly on a boundary: a loop body always keeps a
// COM: few bytes of scalar values live too (the induction variable and friends), so
// COM: an exactly-power-of-two peak is not reachable, and a calibration that only
// COM: just clears a boundary breaks on any unrelated change to the analysis. The
// COM: fourth module below is the exception, landing exactly on the 256 B/lane
// COM: boundary on purpose -- see its own comment.

// COM: Peak 2564 B/lane -- above every budget, so the A operand's load sinks in all
// COM: five GRF modes.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @high_pressure_all_modes(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @high_pressure_all_modes
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <256x128xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x256xf16>
    // CHECK:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16>
    // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<256x128xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #dot0>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<128x256xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32, #dpas>, i32)  : i32 {
      // CHECK:      scf.for
      // CHECK:      tt.descriptor_load {{.*}} : !tt.tensordesc<256x128xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<128x256xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<128x256xf16> -> tensor<128x256xf16, #dot1>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<256x128xf16, #dot0> * tensor<128x256xf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %9, %7 : tensor<256x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

// COM: Peak 644 B/lane -- at or above the 256 B/lane budget only, so the A load
// COM: sinks in the 128 and 256 modes, and stays put in default, auto and 512
// COM: (whose budgets are 1024 B/lane).
#dpas1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_1 = #ttg.dot_op<{opIdx = 0, parent = #dpas1, kWidth=1}>
#dot1_1 = #ttg.dot_op<{opIdx = 1, parent = #dpas1, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @mid_pressure_gate(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @mid_pressure_gate
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #dpas1>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x128xf16>
    // COM: In the KEEP case no prefetch of the A descriptor may be emitted at all;
    // COM: this directive is the first KEEP2 one in the block, so its search range
    // COM: starts at the CHECK-LABEL rather than part-way down the function.
    // KEEP2-NOT:  ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
    // SINK2:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<128x64xf16>
    // SINK2-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
    // KEEP2:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<128x64xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0_1>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<128x128xf32, #dpas1>, i32)  : i32 {
      // CHECK:      scf.for
      // SINK2:      tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
      // KEEP2-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<64x128xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #dot1_1>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<128x64xf16, #dot0_1> * tensor<64x128xf16, #dot1_1> -> tensor<128x128xf32, #dpas1>
      scf.yield %9, %7 : tensor<128x128xf32, #dpas1>, i32
    }
    tt.return
  }
}

// -----

// COM: Peak 388 B/lane -- at or above the 256 B/lane budget only, so the A load
// COM: sinks in the 128 mode only, and stays put in default, auto, 256 and 512
// COM: (whose budgets are all 512 B/lane or higher).
// COM: The A tile is 64x64, far below the 128x128-element floor the pass used to
// COM: require before a load could be sunk, so this module also pins the move from
// COM: a fixed tensor-size gate to a pressure-based one.
#dpas2 = #ttig.dpas<{repeatCount = 4, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [4, 16], B = [16, 16], C = [4, 16]}>
#dot0_2 = #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth=1}>
#dot1_2 = #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @small_tile_high_pressure(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @small_tile_high_pressure
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #dpas2>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x128xf16>
    // KEEP3-NOT:  ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<64x64xf16>
    // SINK3:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<64x64xf16>
    // SINK3-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<64x64xf16>
    // KEEP3:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<64x64xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #dot0_2>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<64x128xf32, #dpas2>, i32)  : i32 {
      // CHECK:      scf.for
      // SINK3:      tt.descriptor_load {{.*}} : !tt.tensordesc<64x64xf16>
      // KEEP3-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<64x64xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<64x128xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #dot1_2>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<64x64xf16, #dot0_2> * tensor<64x128xf16, #dot1_2> -> tensor<64x128xf32, #dpas2>
      scf.yield %9, %7 : tensor<64x128xf32, #dpas2>, i32
    }
    tt.return
  }
}

// -----

// COM: Peak 256 B/lane -- exactly at the 128 mode's budget boundary, so the A
// COM: load sinks there (the gate is `>=`) and stays put in default, auto, 256
// COM: and 512, whose budgets are all strictly above it (default/auto now
// COM: share 512's 1024 B/lane budget, not 128's -- see the file comment).
// COM: Landing exactly on a boundary (unlike the three loops above, which
// COM: deliberately sit inside their bucket) pins the `>=` vs `>` choice
// COM: itself: flipping the comparison to `>` would flip this case's outcome
// COM: while leaving every other loop in this file unchanged. The shape is
// COM: `@loop_with_dpas_accumulator` from test/Analysis/register-pressure.mlir
// COM: (peak = 256 bytes there, confirmed by --test-register-pressure), with
// COM: the A descriptor's load hoisted above the loop so it is live-in and
// COM: thus a sink candidate; the set of values live at the tt.dot is
// COM: unchanged, so the peak is still exactly 256.
#dpas3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_3 = #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth=1}>
#dot1_3 = #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @boundary_pressure_gate(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @boundary_pressure_gate
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas3>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    // KEEP4-NOT:  ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<8x64xf16>
    // SINK4:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<8x64xf16>
    // SINK4-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<8x64xf16>
    // KEEP4:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<8x64xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0_3>
    %3 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst) -> (tensor<8x16xf32, #dpas3>) : i32 {
      // CHECK:      scf.for
      ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16>
      %4 = tt.descriptor_load %1[%c0_i32, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1_3>
      // SINK4:      tt.descriptor_load {{.*}} : !tt.tensordesc<8x64xf16>
      // KEEP4-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<8x64xf16>
      %5 = tt.dot %2, %4, %arg3, inputPrecision = tf32 : tensor<8x64xf16, #dot0_3> * tensor<64x16xf16, #dot1_3> -> tensor<8x16xf32, #dpas3>
      scf.yield %5 : tensor<8x16xf32, #dpas3>
    }
    tt.return
  }
}
