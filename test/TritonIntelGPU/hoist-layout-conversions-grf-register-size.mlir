// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" | FileCheck %s --check-prefixes=CHECK,SMALL
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=auto" | FileCheck %s --check-prefixes=CHECK,SMALL
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128" | FileCheck %s --check-prefixes=CHECK,SMALL
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=256" | FileCheck %s --check-prefixes=CHECK,LARGE

// COM: The budget scales with ttig.grf_register_size. Each module is case 13 of
// COM: hoist-layout-conversions.mlir (projection 288 + 64 = 352 B/lane, no
// COM: credit) with a different register size. With 32 B registers the
// COM: threshold is 204 at 128 GRF ("default" and "auto" included), so the hoist
// COM: is refused, and 409 at 256 GRF, where it fits. With 64 B registers it is
// COM: 409 at 128 GRF and 819 at 256 GRF, so it hoists in every mode.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.grf_register_size = 64 : i32} {
  // CHECK-LABEL: tt.func @grf_register_size_64
  tt.func @grf_register_size_64(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<16x16xf16, #dot_b>, %arg2: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    // CHECK: tt.dot
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %dot : tensor<128x16xf32, #dpas>
    }
    %post = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked>
    tt.return %result, %post : tensor<128x16xf32, #dpas>, tensor<128x16xf16, #blocked>
  }
}

// -----

// COM: 32 B registers, stated explicitly: the same budget as no attribute.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.grf_register_size = 32 : i32} {
  // CHECK-LABEL: tt.func @grf_register_size_32
  tt.func @grf_register_size_32(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<16x16xf16, #dot_b>, %arg2: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // LARGE: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // LARGE-NEXT: scf.for
    // SMALL: scf.for
    // SMALL-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %dot : tensor<128x16xf32, #dpas>
    }
    %post = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked>
    tt.return %result, %post : tensor<128x16xf32, #dpas>, tensor<128x16xf16, #blocked>
  }
}

// -----

// COM: A width other than 32 or 64 is ignored and the 32 B default applies.
// COM: Taken at face value, 128 B would give a 819 B/lane threshold at 128 GRF
// COM: and the hoist would fit.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.grf_register_size = 128 : i32} {
  // CHECK-LABEL: tt.func @grf_register_size_unknown
  tt.func @grf_register_size_unknown(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<16x16xf16, #dot_b>, %arg2: tensor<128x16xf32, #dpas>) -> (tensor<128x16xf32, #dpas>, tensor<128x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // LARGE: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // LARGE-NEXT: scf.for
    // SMALL: scf.for
    // SMALL-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %dot : tensor<128x16xf32, #dpas>
    }
    %post = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked>
    tt.return %result, %post : tensor<128x16xf32, #dpas>, tensor<128x16xf16, #blocked>
  }
}
