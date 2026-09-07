// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=128 -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=256 -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=auto -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-variable-liveness=grf-mode=512 -cse | FileCheck %s --check-prefixes=CHECK,KEEP

// COM: This module's `scf.for` body measures exactly 1024 B/lane live-in
// COM: pressure (the 256x128 f16 A operand = 256×128×2 bytes / 32 warps / 16 threads
// COM: = 1024 B/lane). This value sits exactly on the 256-GRF-mode per-lane budget's
// COM: 200% threshold (512 B/thread * 2 / 16 threads = 1024 B/lane), testing the
// COM: >= comparison (pressure at threshold sinks), and exactly between that floor
// COM: and the 512-GRF-mode 200% threshold (1024 B/thread * 2 / 16 threads = 2048 B/lane).
// COM: So default/128/256 modes sink the A operand's load into the loop (pressure
// COM: >= threshold) while 512-mode keeps it outside (pressure < threshold).
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @grf_mode_gate(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @grf_mode_gate
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <256x128xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x256xf16>
    // SINK:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16>
    // SINK-NOT:  tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16>
    // KEEP:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // KEEP-NOT:  ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #dot0>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<128x256xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32, #dpas>, i32)  : i32 {
      // CHECK:      scf.for
      // SINK:       tt.descriptor_load {{.*}} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
      // KEEP-NOT:   tt.descriptor_load {{.*}} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
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

// COM: Test small tile (sub-128 dimension) can now sink on high pressure alone.
// COM: This tests the semantic change: old logic had hard 128x128x2 size floor.
// COM: 64x128 f16 A operand = 64×128×2 bytes / 32 warps / 16 threads = 512 B/lane.
// COM: Pressure exactly at default-mode 200% threshold (512 bytes).
#dpas2 = #ttig.dpas<{repeatCount = 4, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [4, 16], B = [16, 16], C = [4, 16]}>
#dot0_2 = #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth=1}>
#dot1_2 = #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @small_tile_high_pressure(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @small_tile_high_pressure
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #dpas2>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x128xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x256xf16>
    // SINK:      ttig.descriptor_prefetch
    // SINK-NOT:  tt.descriptor_load {{.*}} : !tt.tensordesc<64x128xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16, #dot0_2>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<128x256xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<64x256xf32, #dpas2>, i32)  : i32 {
      // SINK:  tt.descriptor_load {{.*}} : !tt.tensordesc<64x128xf16>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<128x256xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<128x256xf16> -> tensor<128x256xf16, #dot1_2>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<64x128xf16, #dot0_2> * tensor<128x256xf16, #dot1_2> -> tensor<64x256xf32, #dpas2>
      scf.yield %9, %7 : tensor<64x256xf32, #dpas2>, i32
    }
    tt.return
  }
}
