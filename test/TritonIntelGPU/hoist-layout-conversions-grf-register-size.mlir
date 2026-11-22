// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" | FileCheck %s --check-prefixes=CHECK,SMALL
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=auto" | FileCheck %s --check-prefixes=CHECK,SMALL
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=128" | FileCheck %s --check-prefixes=CHECK,SMALL
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=256" | FileCheck %s --check-prefixes=CHECK,LARGE

// COM: The budget scales with ttig.grf_register_size. Each module is case 25
// COM: (@ceiling_is_budget_reject) of hoist-layout-conversions.mlir with a
// COM: different register size: the hoist raises the function peak from
// COM: prePeak=241 to corridor=257 B/lane, so only the budget can admit it. With
// COM: 32 B registers the budget is 256 at 128 GRF ("default" and "auto"
// COM: included), so the hoist is refused, and 512 at 256 GRF, where it fits.
// COM: With 64 B registers it is 512 at 128 GRF and 1024 at 256 GRF, so it
// COM: hoists in every mode.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.grf_register_size = 64 : i32} {
  // CHECK-LABEL: tt.func @grf_register_size_64
  tt.func @grf_register_size_64(%arg0: tensor<32x16xf16, #blocked>, %arg1: tensor<32x16xf16, #blocked>, %argB: tensor<16x16xf16, #dot_b>, %acc0: tensor<32x16xf32, #dpas>, %cond: i1, %e1: tensor<32x16xf16, #blocked>, %e2: tensor<32x16xf16, #blocked>, %e3: tensor<32x16xf16, #blocked>, %e4: tensor<32x16xf16, #blocked>, %e5: tensor<32x16xf16, #blocked>, %e6: tensor<32x16xf16, #blocked>, %e7: tensor<32x16xf16, #blocked>, %e8: tensor<32x16xf16, #blocked>) -> (tensor<32x16xf32, #dpas>, tensor<32x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked>
    // CHECK: arith.addf
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.if
    %hot = scf.if %cond -> (tensor<32x16xf16, #blocked>) {
      %u1 = arith.addf %src, %e1 : tensor<32x16xf16, #blocked>
      %u2 = arith.addf %u1, %e2 : tensor<32x16xf16, #blocked>
      %u3 = arith.addf %u2, %e3 : tensor<32x16xf16, #blocked>
      %u4 = arith.addf %u3, %e4 : tensor<32x16xf16, #blocked>
      %u5 = arith.addf %u4, %e5 : tensor<32x16xf16, #blocked>
      %u6 = arith.addf %u5, %e6 : tensor<32x16xf16, #blocked>
      %u7 = arith.addf %u6, %e7 : tensor<32x16xf16, #blocked>
      %u8 = arith.addf %u7, %e8 : tensor<32x16xf16, #blocked>
      scf.yield %u8 : tensor<32x16xf16, #blocked>
    } else {
      scf.yield %arg1 : tensor<32x16xf16, #blocked>
    }
    // CHECK: scf.for
    // CHECK-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<32x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked> -> tensor<32x16xf16, #dot_a>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<32x16xf32, #dpas>
      scf.yield %d : tensor<32x16xf32, #dpas>
    }
    tt.return %r, %hot : tensor<32x16xf32, #dpas>, tensor<32x16xf16, #blocked>
  }
}

// -----

// COM: 32 B registers, stated explicitly: the same budget as no attribute.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.grf_register_size = 32 : i32} {
  // CHECK-LABEL: tt.func @grf_register_size_32
  tt.func @grf_register_size_32(%arg0: tensor<32x16xf16, #blocked>, %arg1: tensor<32x16xf16, #blocked>, %argB: tensor<16x16xf16, #dot_b>, %acc0: tensor<32x16xf32, #dpas>, %cond: i1, %e1: tensor<32x16xf16, #blocked>, %e2: tensor<32x16xf16, #blocked>, %e3: tensor<32x16xf16, #blocked>, %e4: tensor<32x16xf16, #blocked>, %e5: tensor<32x16xf16, #blocked>, %e6: tensor<32x16xf16, #blocked>, %e7: tensor<32x16xf16, #blocked>, %e8: tensor<32x16xf16, #blocked>) -> (tensor<32x16xf32, #dpas>, tensor<32x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked>
    // LARGE: arith.addf
    // LARGE-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // LARGE-NEXT: scf.if
    %hot = scf.if %cond -> (tensor<32x16xf16, #blocked>) {
      %u1 = arith.addf %src, %e1 : tensor<32x16xf16, #blocked>
      %u2 = arith.addf %u1, %e2 : tensor<32x16xf16, #blocked>
      %u3 = arith.addf %u2, %e3 : tensor<32x16xf16, #blocked>
      %u4 = arith.addf %u3, %e4 : tensor<32x16xf16, #blocked>
      %u5 = arith.addf %u4, %e5 : tensor<32x16xf16, #blocked>
      %u6 = arith.addf %u5, %e6 : tensor<32x16xf16, #blocked>
      %u7 = arith.addf %u6, %e7 : tensor<32x16xf16, #blocked>
      %u8 = arith.addf %u7, %e8 : tensor<32x16xf16, #blocked>
      scf.yield %u8 : tensor<32x16xf16, #blocked>
    } else {
      scf.yield %arg1 : tensor<32x16xf16, #blocked>
    }
    // CHECK: scf.for
    // SMALL-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // LARGE-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<32x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked> -> tensor<32x16xf16, #dot_a>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<32x16xf32, #dpas>
      scf.yield %d : tensor<32x16xf32, #dpas>
    }
    tt.return %r, %hot : tensor<32x16xf32, #dpas>, tensor<32x16xf16, #blocked>
  }
}

// -----

// COM: A width other than 32 or 64 is ignored and the 32 B default applies.
// COM: Taken at face value, 128 B would give a 1024 B/lane budget at 128 GRF
// COM: and the hoist would fit.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.grf_register_size = 128 : i32} {
  // CHECK-LABEL: tt.func @grf_register_size_unknown
  tt.func @grf_register_size_unknown(%arg0: tensor<32x16xf16, #blocked>, %arg1: tensor<32x16xf16, #blocked>, %argB: tensor<16x16xf16, #dot_b>, %acc0: tensor<32x16xf32, #dpas>, %cond: i1, %e1: tensor<32x16xf16, #blocked>, %e2: tensor<32x16xf16, #blocked>, %e3: tensor<32x16xf16, #blocked>, %e4: tensor<32x16xf16, #blocked>, %e5: tensor<32x16xf16, #blocked>, %e6: tensor<32x16xf16, #blocked>, %e7: tensor<32x16xf16, #blocked>, %e8: tensor<32x16xf16, #blocked>) -> (tensor<32x16xf32, #dpas>, tensor<32x16xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %src = arith.addf %arg0, %arg0 : tensor<32x16xf16, #blocked>
    // LARGE: arith.addf
    // LARGE-NEXT: ttg.convert_layout %{{.*}} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // LARGE-NEXT: scf.if
    %hot = scf.if %cond -> (tensor<32x16xf16, #blocked>) {
      %u1 = arith.addf %src, %e1 : tensor<32x16xf16, #blocked>
      %u2 = arith.addf %u1, %e2 : tensor<32x16xf16, #blocked>
      %u3 = arith.addf %u2, %e3 : tensor<32x16xf16, #blocked>
      %u4 = arith.addf %u3, %e4 : tensor<32x16xf16, #blocked>
      %u5 = arith.addf %u4, %e5 : tensor<32x16xf16, #blocked>
      %u6 = arith.addf %u5, %e6 : tensor<32x16xf16, #blocked>
      %u7 = arith.addf %u6, %e7 : tensor<32x16xf16, #blocked>
      %u8 = arith.addf %u7, %e8 : tensor<32x16xf16, #blocked>
      scf.yield %u8 : tensor<32x16xf16, #blocked>
    } else {
      scf.yield %arg1 : tensor<32x16xf16, #blocked>
    }
    // CHECK: scf.for
    // SMALL-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<32x16xf16, #{{.*}}> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // LARGE-NOT: ttg.convert_layout
    %r = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<32x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %src : tensor<32x16xf16, #blocked> -> tensor<32x16xf16, #dot_a>
      %d = tt.dot %cvt, %argB, %a, inputPrecision = tf32 : tensor<32x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<32x16xf32, #dpas>
      scf.yield %d : tensor<32x16xf32, #dpas>
    }
    tt.return %r, %hot : tensor<32x16xf32, #dpas>, tensor<32x16xf16, #blocked>
  }
}
