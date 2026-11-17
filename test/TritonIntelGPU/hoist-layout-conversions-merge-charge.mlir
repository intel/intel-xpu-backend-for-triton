// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128 corridor-op-cap=5" | FileCheck %s

// COM: A twin that merges into a hoisted sibling charges its loop for the merge,
// COM: whichever is decided first. Own file: corridor-op-cap=5 would rerun every
// COM: case of hoist-layout-conversions.mlir under that cap.

// COM: %t is refused after %h, the hoist it merges into, so its loop is charged
// COM: at the refusal (+48) and the later group {%y} stays stamped. %t's own
// COM: fate is not checked.

#blk = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dpasr = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#dot_ar = #ttg.dot_op<{opIdx = 0, parent = #dpasr, kWidth = 1}>
#dot_br = #ttg.dot_op<{opIdx = 1, parent = #dpasr, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @late_refusal_merge_now_charged
  tt.func @late_refusal_merge_now_charged(%arg0: tensor<32x16xf16, #blk>, %arg1: tensor<16x16xf16, #blk>, %arg2: tensor<16x16xf16, #blk>,
      %arg3: tensor<128x32xf16, #dot_a>, %arg4: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %zA32 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #dot_a>
    %zA16 = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_a>
    %zAr = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_ar>
    %zCr = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpasr>
    // CHECK: %[[S:.*]] = arith.addf %arg0, %arg0
    // CHECK: ttg.convert_layout %[[S]] : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %s = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blk>
    // CHECK: %[[Q:.*]] = arith.addf %arg1, %arg1
    // CHECK: ttg.convert_layout %[[Q]] : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %q = arith.addf %arg1, %arg1 : tensor<16x16xf16, #blk>
    // CHECK: scf.for
    %r2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %arg4) -> (tensor<128x16xf32, #dpas>) : i32 {
      %t = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %arg3, %t, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      %fx = arith.addf %arg2, %arg2 : tensor<16x16xf16, #blk>
      %x1 = arith.addf %d, %d : tensor<128x16xf32, #dpas>
      %x2 = arith.addf %x1, %d : tensor<128x16xf32, #dpas>
      %x3 = arith.addf %x2, %d : tensor<128x16xf32, #dpas>
      %x4 = arith.addf %x3, %d : tensor<128x16xf32, #dpas>
      // CHECK: ttg.convert_layout %[[Q]] {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
      %y = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_b>
      %e = tt.dot %zA16, %y, %x4, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %e : tensor<128x16xf32, #dpas>
    }
    %r1 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %r2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %h = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %zA32, %h, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d : tensor<128x16xf32, #dpas>
    }
    %r3 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zCr) -> (tensor<128x16xf32, #dpasr>) : i32 {
      %z = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_br>
      %d = tt.dot %zAr, %z, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_ar> * tensor<16x16xf16, #dot_br> -> tensor<128x16xf32, #dpasr>
      scf.yield %d : tensor<128x16xf32, #dpasr>
    }
    %p1 = arith.addf %r1, %r1 : tensor<128x16xf32, #dpas>
    %p2 = arith.addf %p1, %r1 : tensor<128x16xf32, #dpas>
    %p3 = arith.addf %p2, %p1 : tensor<128x16xf32, #dpas>
    %p4 = arith.addf %p3, %p2 : tensor<128x16xf32, #dpas>
    %p5 = arith.addf %p4, %p3 : tensor<128x16xf32, #dpas>
    %p6 = arith.addf %p5, %p4 : tensor<128x16xf32, #dpas>
    %p7 = arith.addf %p6, %p1 : tensor<128x16xf32, #dpas>
    %p8 = arith.addf %p7, %p2 : tensor<128x16xf32, #dpas>
    %p9 = arith.addf %p8, %p3 : tensor<128x16xf32, #dpas>
    %p10 = arith.addf %p9, %p4 : tensor<128x16xf32, #dpas>
    %p11 = arith.addf %p10, %p5 : tensor<128x16xf32, #dpas>
    %p12 = arith.addf %p11, %p6 : tensor<128x16xf32, #dpas>
    %p13 = arith.addf %p12, %p7 : tensor<128x16xf32, #dpas>
    tt.return %p13, %r3, %p8 : tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>
  }
}

// -----

// COM: %h hoists uncredited; %t still merges into it, so loop 2 is charged
// COM: (+48) and the later group {%y} stays stamped. %t's own fate is not
// COM: checked.

#blk = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dpasr = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 1], A = [16, 16], B = [16, 16], C = [16, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#dot_ar = #ttg.dot_op<{opIdx = 0, parent = #dpasr, kWidth = 1}>
#dot_br = #ttg.dot_op<{opIdx = 1, parent = #dpasr, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @uncharged_merge_now_charged
  tt.func @uncharged_merge_now_charged(%arg0: tensor<32x16xf16, #blk>, %arg1: tensor<16x16xf16, #blk>, %arg2: tensor<16x16xf16, #blk>,
      %arg3: tensor<128x32xf16, #dot_a>, %arg4: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %zA32 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #dot_a>
    %zA16 = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_a>
    %zAr = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #dot_ar>
    %zCr = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #dpasr>
    // CHECK: %[[S:.*]] = arith.addf %arg0, %arg0
    // CHECK: ttg.convert_layout %[[S]] : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %s = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blk>
    // CHECK: scf.for
    %r1 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %arg4) -> (tensor<128x16xf32, #dpas>) : i32 {
      %h = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %zA32, %h, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d : tensor<128x16xf32, #dpas>
    }
    // CHECK: %[[Q:.*]] = arith.addf %arg1, %arg1
    // CHECK: ttg.convert_layout %[[Q]] : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    %q = arith.addf %arg1, %arg1 : tensor<16x16xf16, #blk>
    // CHECK: scf.for
    %r2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %r1) -> (tensor<128x16xf32, #dpas>) : i32 {
      %t = ttg.convert_layout %s : tensor<32x16xf16, #blk> -> tensor<32x16xf16, #dot_b>
      %d = tt.dot %arg3, %t, %acc, inputPrecision = tf32 : tensor<128x32xf16, #dot_a> * tensor<32x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      %fx = arith.addf %arg2, %arg2 : tensor<16x16xf16, #blk>
      %x1 = arith.addf %d, %d : tensor<128x16xf32, #dpas>
      %x2 = arith.addf %x1, %d : tensor<128x16xf32, #dpas>
      %x3 = arith.addf %x2, %d : tensor<128x16xf32, #dpas>
      %x4 = arith.addf %x3, %d : tensor<128x16xf32, #dpas>
      // CHECK: ttg.convert_layout %[[Q]] {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
      %y = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_b>
      %e = tt.dot %zA16, %y, %x4, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %e : tensor<128x16xf32, #dpas>
    }
    %r3 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zCr) -> (tensor<128x16xf32, #dpasr>) : i32 {
      %z = ttg.convert_layout %q : tensor<16x16xf16, #blk> -> tensor<16x16xf16, #dot_br>
      %d = tt.dot %zAr, %z, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_ar> * tensor<16x16xf16, #dot_br> -> tensor<128x16xf32, #dpasr>
      scf.yield %d : tensor<128x16xf32, #dpasr>
    }
    %p1 = arith.addf %r2, %r2 : tensor<128x16xf32, #dpas>
    %p2 = arith.addf %p1, %r2 : tensor<128x16xf32, #dpas>
    %p3 = arith.addf %p2, %p1 : tensor<128x16xf32, #dpas>
    %p4 = arith.addf %p3, %p2 : tensor<128x16xf32, #dpas>
    %p5 = arith.addf %p4, %p3 : tensor<128x16xf32, #dpas>
    %p6 = arith.addf %p5, %p4 : tensor<128x16xf32, #dpas>
    %p7 = arith.addf %p6, %p1 : tensor<128x16xf32, #dpas>
    %p8 = arith.addf %p7, %p2 : tensor<128x16xf32, #dpas>
    %p9 = arith.addf %p8, %p3 : tensor<128x16xf32, #dpas>
    %p10 = arith.addf %p9, %p4 : tensor<128x16xf32, #dpas>
    %p11 = arith.addf %p10, %p5 : tensor<128x16xf32, #dpas>
    %p12 = arith.addf %p11, %p6 : tensor<128x16xf32, #dpas>
    %p13 = arith.addf %p12, %p7 : tensor<128x16xf32, #dpas>
    tt.return %p13, %r3, %p8 : tensor<128x16xf32, #dpas>, tensor<128x16xf32, #dpasr>, tensor<128x16xf32, #dpas>
  }
}
