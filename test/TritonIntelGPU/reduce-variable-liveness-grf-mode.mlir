// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s --check-prefixes=CHECK,SINK2,SINK3,SINK4,SINK5,SINK6
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=128 -cse | FileCheck %s --check-prefixes=CHECK,SINK2,SINK3,SINK4,SINK5,SINK6
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=256 -cse | FileCheck %s --check-prefixes=CHECK,SINK2,KEEP3,KEEP4,SINK5,KEEP6
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=auto -cse | FileCheck %s --check-prefixes=CHECK,SINK2,SINK3,SINK4,SINK5,SINK6
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=512 -cse | FileCheck %s --check-prefixes=CHECK,KEEP2,KEEP3,KEEP4,KEEP5,KEEP6
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=160 -cse | FileCheck %s --check-prefixes=CHECK,SINK2,SINK3,KEEP4,SINK5,SINK6
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=192 -cse | FileCheck %s --check-prefixes=CHECK,SINK2,SINK3,KEEP4,SINK5,KEEP6

// COM: A loop-invariant 2D load is sunk into the loop when the loop body's *peak*
// COM: register pressure is at or above the per-lane GRF budget of the selected GRF
// COM: mode. The peaks quoted below are not hand-derived: they are what
// COM: `triton-opt <this file> -split-input-file --test-register-pressure` reports
// COM: for the `scf.for` body block.
// COM:
// COM: With "ttg.threads-per-warp" = 16 and 32-byte registers the per-lane budgets are:
// COM:   128, default and auto -> 4096 B/thread / 16 = 256 B/lane
// COM:   160    -> 5120 B/thread / 16 = 320 B/lane
// COM:   192    -> 6144 B/thread / 16 = 384 B/lane
// COM:   256    -> 8192 B/thread / 16 = 512 B/lane
// COM:   512    -> 16384 B/thread / 16 = 1024 B/lane
// COM:
// COM: "default" and "auto" get the 128 budget: on BMG, PVC and DG2 such a
// COM: compile starts at 128 registers and only moves to a larger mode if that
// COM: build spills, so sinking against a larger budget would leave the pressure
// COM: that pushes it there.
// COM:
// COM: The first four modules have peaks of 2564, 644, 388 and 256 B/lane. 388 and
// COM: 256 share a bucket at 128, default, auto, 256 and 512, which is why SINK3/SINK4
// COM: and KEEP3/KEEP4 are paired on those RUN lines; 160 and 192 split them (388
// COM: sinks, 256 stays). A loop body always keeps a few bytes of scalar values live
// COM: too (the induction variable and friends), so an exactly-power-of-two peak is
// COM: hard to hit by accident. Three peaks sit on a boundary on purpose: the fourth
// COM: module lands exactly on 256 B/lane (see its own comment), 388 is 4 B above
// COM: 192's 384 B/lane, and the sixth module's 644 is 4 B above 160's 640 B/lane at
// COM: 64-byte registers.
// COM:
// COM: The fifth module below sets `ttig.max_grf_mode = "512"` (stamped by
// COM: `TritonAnnotateModule` from compiler.py's `get_max_grf_mode()`) and pins
// COM: that it does not raise the "default"/"auto" budget.
// COM:
// COM: The sixth and seventh modules set `ttig.grf_register_size = 64` as well
// COM: (the 64-byte registers of PVC/BMG/CRI, see compiler.py's
// COM: `get_grf_register_size()`), which doubles every budget above. Modules
// COM: 1-5 have no such attribute and
// COM: get the 32-byte fallback the budgets above are computed with.

// COM: Peak 2564 B/lane, above every budget, so the A operand's load sinks in all
// COM: seven GRF modes.
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

// COM: Peak 644 B/lane, at or above the 256 and 512 B/lane budgets, so the A
// COM: load sinks in the 128, default, auto and 256 modes, and stays put in 512
// COM: (whose budget is 1024 B/lane).
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

// COM: Peak 388 B/lane, at or above the 256, 320 and 384 B/lane budgets, so the A
// COM: load sinks in the 128, default, auto, 160 and 192 modes, and stays put in 256
// COM: and 512 (whose budgets are 512 B/lane or higher).
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

// COM: Peak 256 B/lane, exactly at the 128, default and auto budget boundary,
// COM: so the A load sinks there (the gate is `>=`) and stays put in 160, 192,
// COM: 256 and 512, whose budgets are strictly above it.
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

// -----

// COM: Peak 644 B/lane, the same shape as @mid_pressure_gate above, but this module
// COM: also sets `ttig.max_grf_mode = "512"`. The attribute does not change any
// COM: budget, so the outcome matches @mid_pressure_gate: the A load sinks in
// COM: "128", "default", "auto" and "256", and stays put in "512". "512" because
// COM: it is the value that would move the outcome if "default"/"auto" read it
// COM: (1024 B/lane > 644); "256" (512 B/lane) would not.
#dpas4 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_4 = #ttg.dot_op<{opIdx = 0, parent = #dpas4, kWidth=1}>
#dot1_4 = #ttg.dot_op<{opIdx = 1, parent = #dpas4, kWidth=2}>
module attributes {ttig.support_2d_block_io, ttig.max_grf_mode = "512", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @max_grf_mode_attr_gate(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @max_grf_mode_attr_gate
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #dpas4>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x128xf16>
    // COM: In the KEEP case no prefetch of the A descriptor may be emitted at all;
    // COM: this directive is the first KEEP5 one in the block, so its search range
    // COM: starts at the CHECK-LABEL rather than part-way down the function.
    // KEEP5-NOT:  ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
    // SINK5:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<128x64xf16>
    // SINK5-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
    // KEEP5:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<128x64xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0_4>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<128x128xf32, #dpas4>, i32)  : i32 {
      // CHECK:      scf.for
      // SINK5:      tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
      // KEEP5-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<64x128xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #dot1_4>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<128x64xf16, #dot0_4> * tensor<64x128xf16, #dot1_4> -> tensor<128x128xf32, #dpas4>
      scf.yield %9, %7 : tensor<128x128xf32, #dpas4>, i32
    }
    tt.return
  }
}

// -----

// COM: Peak 644 B/lane, the shape of @max_grf_mode_attr_gate above, with
// COM: `ttig.max_grf_mode = "256"` and `ttig.grf_register_size = 64`, so
// COM: every budget doubles: 512 B/lane for "128", "default" and "auto", 640 for
// COM: "160", 768 for "192", 1024 for "256", 2048 for "512". The A load now stays
// COM: put in "192" and "256", where with 32-byte registers it sank.
#dpas5 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0_5 = #ttg.dot_op<{opIdx = 0, parent = #dpas5, kWidth=1}>
#dot1_5 = #ttg.dot_op<{opIdx = 1, parent = #dpas5, kWidth=2}>
module attributes {ttig.support_2d_block_io, ttig.max_grf_mode = "256", ttig.grf_register_size = 64 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @grf_register_size_attr_gate(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @grf_register_size_attr_gate
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #dpas5>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x128xf16>
    // KEEP6-NOT:  ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
    // SINK6:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<128x64xf16>
    // SINK6-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
    // KEEP6:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<128x64xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0_5>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<128x128xf32, #dpas5>, i32)  : i32 {
      // CHECK:      scf.for
      // SINK6:      tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
      // KEEP6-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<128x64xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<64x128xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #dot1_5>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<128x64xf16, #dot0_5> * tensor<64x128xf16, #dot1_5> -> tensor<128x128xf32, #dpas5>
      scf.yield %9, %7 : tensor<128x128xf32, #dpas5>, i32
    }
    tt.return
  }
}

// -----

// COM: Peak 388 B/lane, the shape of @small_tile_high_pressure, with
// COM: `ttig.grf_register_size = 64`: "128", "default" and "auto" get 512 B/lane,
// COM: so the A load stays put in every mode. With the 32-byte fallback it sinks
// COM: in "128", "default", "auto", "160" and "192": this pins that the
// COM: "default"/"auto" budget, not only the explicit ones, reads the attribute.
#dpas6 = #ttig.dpas<{repeatCount = 4, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [4, 16], B = [16, 16], C = [4, 16]}>
#dot0_6 = #ttg.dot_op<{opIdx = 0, parent = #dpas6, kWidth=1}>
#dot1_6 = #ttg.dot_op<{opIdx = 1, parent = #dpas6, kWidth=2}>
module attributes {ttig.support_2d_block_io, ttig.grf_register_size = 64 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @grf_register_size_default_budget(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @grf_register_size_default_budget
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #dpas6>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x128xf16>
    // CHECK-NOT:  ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<64x64xf16>
    // CHECK:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<64x64xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #dot0_6>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<64x128xf32, #dpas6>, i32)  : i32 {
      // CHECK:      scf.for
      // CHECK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<64x64xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<64x128xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #dot1_6>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<64x64xf16, #dot0_6> * tensor<64x128xf16, #dot1_6> -> tensor<64x128xf32, #dpas6>
      scf.yield %9, %7 : tensor<64x128xf32, #dpas6>, i32
    }
    tt.return
  }
}
