// RUN: triton-opt %s -split-input-file --tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: scf.for with pass-through yield (descriptor unchanged across iterations).
// COM: findAllMakeTensorDescOps should resolve to the unique MakeTensorDescOp.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_passthrough_yield
  tt.func @for_passthrough_yield(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
      scf.yield %iter_desc : !tt.tensordesc<64x32xf16, #dot_a>
    }
    tt.return
  }
}

// -----

// COM: scf.for where the yield provides a different MakeTensorDescOp with compatible
// COM: alignment properties. All candidates satisfy constraints, so block_io is set.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_multiple_compatible_descs
  tt.func @for_multiple_compatible_descs(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %desc2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc1) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
      scf.yield %desc2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    tt.return
  }
}

// -----

// COM: scf.for where the yield provides a MakeTensorDescOp with incompatible
// COM: pitch (not divisible by 128/elementWidth). block_io should NOT be set.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @for_incompatible_pitch
  tt.func @for_incompatible_pitch(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch_good: i64 {tt.divisibility = 16 : i32}, %pitch_bad: i64 {tt.divisibility = 3 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %desc1 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch_good, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %desc2 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch_bad, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #dot_a>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter_desc = %desc1) -> (!tt.tensordesc<64x32xf16, #dot_a>) : i32 {
      // CHECK: tt.descriptor_load
      // CHECK-NOT: ttig.block_io
      %ld = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #dot_a> -> tensor<64x32xf16, #dot_a>
      scf.yield %desc2 : !tt.tensordesc<64x32xf16, #dot_a>
    }
    tt.return
  }
}
