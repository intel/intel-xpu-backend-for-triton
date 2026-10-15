// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=1" | FileCheck %s --check-prefixes=CHECK,CAP1
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=3" | FileCheck %s --check-prefixes=CHECK,CAP3
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=1" 2>&1 | FileCheck %s --check-prefix=STATS1
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="corridor-op-cap=3" 2>&1 | FileCheck %s --check-prefix=STATS3

// COM: cap=1 charges both corridors conservatively and refuses both; cap=3
// COM: prices the second loop's corridor and accepts it.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @corridor_cap_two_free_hoists
  tt.func @corridor_cap_two_free_hoists(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<128x16xf16, #blocked>, %argB: tensor<16x16xf16, #dot_b>, %acc0: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %s1 = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked>
    %s2 = arith.addf %arg1, %arg1 : tensor<128x16xf16, #blocked>

    // COM: cap=1: neither conversion moves; both are stamped tt.no_licm.
    // CAP1: arith.addf
    // CAP1-NEXT: arith.addf
    // CAP1-NEXT: scf.for
    // CAP1-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP1: scf.for
    // CAP1-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>

    // COM: cap=3: only the second loop's conversion moves up.
    // CAP3: arith.addf
    // CAP3-NEXT: arith.addf
    // CAP3-NEXT: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP3-NEXT: scf.for
    // CAP3-NEXT: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CAP3: scf.for
    // CAP3-NOT: ttg.convert_layout
    %r1 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %acc0) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt1 = ttg.convert_layout %s1 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %d1 = tt.dot %cvt1, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d1 : tensor<128x16xf32, #dpas>
    }
    %r2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%a = %r1) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt2 = ttg.convert_layout %s2 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %d2 = tt.dot %cvt2, %argB, %a, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %d2 : tensor<128x16xf32, #dpas>
    }
    tt.return %r2 : tensor<128x16xf32, #dpas>
  }
}

// COM: An unwalked corridor counts as a fallback rejection, not an exact one.
// STATS1: [HoistLayoutConversions] considered=2 hoisted=0 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=2 skipped_other=0
// STATS3: [HoistLayoutConversions] considered=2 hoisted=1 rejected_pressure=0 rejected_function_peak_exact=0 rejected_function_peak_fallback=1 skipped_other=0
