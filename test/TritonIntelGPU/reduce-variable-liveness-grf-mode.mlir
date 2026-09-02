// RUN: triton-opt %s -tritonintelgpu-reduce-variable-liveness -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -tritonintelgpu-reduce-variable-liveness=grf-mode=128 -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -tritonintelgpu-reduce-variable-liveness=grf-mode=256 -cse | FileCheck %s --check-prefixes=CHECK,SINK
// RUN: triton-opt %s -tritonintelgpu-reduce-variable-liveness=grf-mode=512 -cse | FileCheck %s --check-prefixes=CHECK,KEEP
// RUN: triton-opt %s -tritonintelgpu-reduce-variable-liveness=grf-mode=bogus -cse | FileCheck %s --check-prefixes=CHECK,SINK

// COM: This module's `scf.for` body measures ~1536 B/lane live-in pressure at
// COM: pass-run time (the 256x128 A operand contributes 1024 B/lane, a second
// COM: live-in 128x128 tensor contributes another 512 B/lane -- confirmed via
// COM: `-test-register-pressure`; that second tensor is dead code and gets
// COM: cleaned up by the `-cse` in the RUN lines above, after the pass has
// COM: already made its sink/keep decision). That value sits strictly between
// COM: the 256-GRF-mode per-lane budget's 200% floor (1024 B) and the
// COM: 512-GRF-mode one (2048 B), so default/128/256 sink the A operand's
// COM: load into the loop while 512 keeps it outside. A bogus grf-mode falls
// COM: back to the 'default' budget, so it also sinks.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @grf_mode_gate(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func @grf_mode_gate
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <256x128xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x256xf16>
    %5 = tt.make_tensor_descriptor %arg2, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <128x128xf16>
    // SINK:      ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16>
    // KEEP:      tt.descriptor_load %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
    // KEEP-NOT:  ttig.descriptor_prefetch %{{.*}}[%c0_i32, %c0_i32] {{.*}} : !tt.tensordesc<256x128xf16>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #dot0>
    %bias = tt.descriptor_load %5[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<128x128xf16> -> tensor<128x128xf16, #dot0>
    ttig.descriptor_prefetch %1[%c0_i32, %c0_i32] : !tt.tensordesc<128x256xf16>
    %4:2 = scf.for %arg3 = %c0_i32 to %c128_i32 step %c128_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<256x256xf32, #dpas>, i32)  : i32 {
      // CHECK:      scf.for
      // SINK:       tt.descriptor_load {{.*}} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
      // KEEP-NOT:   tt.descriptor_load {{.*}} : !tt.tensordesc<256x128xf16> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>
      %7 = arith.addi %arg5, %c128_i32 : i32
      ttig.descriptor_prefetch %1[%7, %c0_i32] : !tt.tensordesc<128x256xf16>
      %8 = tt.descriptor_load %1[%arg5, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<128x256xf16> -> tensor<128x256xf16, #dot1>
      %biasuse = arith.addf %bias, %bias : tensor<128x128xf16, #dot0>
      %9 = tt.dot %2, %8, %arg4, inputPrecision = tf32 : tensor<256x128xf16, #dot0> * tensor<128x256xf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %9, %7 : tensor<256x256xf32, #dpas>, i32
    }
    tt.return
  }
}
