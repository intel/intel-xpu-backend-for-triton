// RUN: triton-opt %s -split-input-file -tritonintelgpu-reorder-for-liveness | FileCheck %s

// COM: Five independent chains, emitted interleaved so that all five are live at
// COM: once. Each value is 32 f64 per lane, so the block is over budget and the
// COM: pass issues one chain at a time, holding three values instead of five.

#blocked = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @interleaved_chains(%arg0: tensor<1024xf64, #blocked>, %arg1: tensor<1024xf64, #blocked>, %arg2: tensor<1024xf64, #blocked>, %arg3: tensor<1024xf64, #blocked>, %arg4: tensor<1024xf64, #blocked>) -> tensor<1024xf64, #blocked> {
    %a0 = arith.mulf %arg0, %arg0 : tensor<1024xf64, #blocked>
    %b0 = arith.mulf %arg1, %arg1 : tensor<1024xf64, #blocked>
    %c0 = arith.mulf %arg2, %arg2 : tensor<1024xf64, #blocked>
    %d0 = arith.mulf %arg3, %arg3 : tensor<1024xf64, #blocked>
    %e0 = arith.mulf %arg4, %arg4 : tensor<1024xf64, #blocked>
    %a1 = arith.mulf %a0, %a0 : tensor<1024xf64, #blocked>
    %b1 = arith.mulf %b0, %b0 : tensor<1024xf64, #blocked>
    %c1 = arith.mulf %c0, %c0 : tensor<1024xf64, #blocked>
    %d1 = arith.mulf %d0, %d0 : tensor<1024xf64, #blocked>
    %e1 = arith.mulf %e0, %e0 : tensor<1024xf64, #blocked>
    %a2 = arith.mulf %a1, %a1 : tensor<1024xf64, #blocked>
    %b2 = arith.mulf %b1, %b1 : tensor<1024xf64, #blocked>
    %c2 = arith.mulf %c1, %c1 : tensor<1024xf64, #blocked>
    %d2 = arith.mulf %d1, %d1 : tensor<1024xf64, #blocked>
    %e2 = arith.mulf %e1, %e1 : tensor<1024xf64, #blocked>
    %a3 = arith.mulf %a2, %a2 : tensor<1024xf64, #blocked>
    %b3 = arith.mulf %b2, %b2 : tensor<1024xf64, #blocked>
    %c3 = arith.mulf %c2, %c2 : tensor<1024xf64, #blocked>
    %d3 = arith.mulf %d2, %d2 : tensor<1024xf64, #blocked>
    %e3 = arith.mulf %e2, %e2 : tensor<1024xf64, #blocked>
    %a4 = arith.mulf %a3, %a3 : tensor<1024xf64, #blocked>
    %b4 = arith.mulf %b3, %b3 : tensor<1024xf64, #blocked>
    %c4 = arith.mulf %c3, %c3 : tensor<1024xf64, #blocked>
    %d4 = arith.mulf %d3, %d3 : tensor<1024xf64, #blocked>
    %e4 = arith.mulf %e3, %e3 : tensor<1024xf64, #blocked>
    %a5 = arith.mulf %a4, %a4 : tensor<1024xf64, #blocked>
    %b5 = arith.mulf %b4, %b4 : tensor<1024xf64, #blocked>
    %c5 = arith.mulf %c4, %c4 : tensor<1024xf64, #blocked>
    %d5 = arith.mulf %d4, %d4 : tensor<1024xf64, #blocked>
    %e5 = arith.mulf %e4, %e4 : tensor<1024xf64, #blocked>
    %t0 = arith.addf %a5, %b5 : tensor<1024xf64, #blocked>
    %t1 = arith.addf %t0, %c5 : tensor<1024xf64, #blocked>
    %t2 = arith.addf %t1, %d5 : tensor<1024xf64, #blocked>
    %t3 = arith.addf %t2, %e5 : tensor<1024xf64, #blocked>
    tt.return %t3 : tensor<1024xf64, #blocked>
  }
}

// CHECK-LABEL: tt.func @interleaved_chains
// CHECK:       %[[A0:.*]] = arith.mulf %arg0, %arg0
// CHECK-NEXT:  %[[A1:.*]] = arith.mulf %[[A0]], %[[A0]]
// CHECK-NEXT:  %[[A2:.*]] = arith.mulf %[[A1]], %[[A1]]
// CHECK-NEXT:  %[[A3:.*]] = arith.mulf %[[A2]], %[[A2]]
// CHECK-NEXT:  %[[A4:.*]] = arith.mulf %[[A3]], %[[A3]]
// CHECK-NEXT:  %[[A5:.*]] = arith.mulf %[[A4]], %[[A4]]
// CHECK-NEXT:  %[[B0:.*]] = arith.mulf %arg1, %arg1
// CHECK-NEXT:  %[[B1:.*]] = arith.mulf %[[B0]], %[[B0]]
// CHECK-NEXT:  %[[B2:.*]] = arith.mulf %[[B1]], %[[B1]]
// CHECK-NEXT:  %[[B3:.*]] = arith.mulf %[[B2]], %[[B2]]
// CHECK-NEXT:  %[[B4:.*]] = arith.mulf %[[B3]], %[[B3]]
// CHECK-NEXT:  %[[B5:.*]] = arith.mulf %[[B4]], %[[B4]]
// CHECK-NEXT:  %[[T0:.*]] = arith.addf %[[A5]], %[[B5]]
// CHECK-NEXT:  %[[C0:.*]] = arith.mulf %arg2, %arg2
// CHECK-NEXT:  %[[C1:.*]] = arith.mulf %[[C0]], %[[C0]]
// CHECK-NEXT:  %[[C2:.*]] = arith.mulf %[[C1]], %[[C1]]
// CHECK-NEXT:  %[[C3:.*]] = arith.mulf %[[C2]], %[[C2]]
// CHECK-NEXT:  %[[C4:.*]] = arith.mulf %[[C3]], %[[C3]]
// CHECK-NEXT:  %[[C5:.*]] = arith.mulf %[[C4]], %[[C4]]
// CHECK-NEXT:  %[[T1:.*]] = arith.addf %[[T0]], %[[C5]]
// CHECK-NEXT:  %[[D0:.*]] = arith.mulf %arg3, %arg3
// CHECK-NEXT:  %[[D1:.*]] = arith.mulf %[[D0]], %[[D0]]
// CHECK-NEXT:  %[[D2:.*]] = arith.mulf %[[D1]], %[[D1]]
// CHECK-NEXT:  %[[D3:.*]] = arith.mulf %[[D2]], %[[D2]]
// CHECK-NEXT:  %[[D4:.*]] = arith.mulf %[[D3]], %[[D3]]
// CHECK-NEXT:  %[[D5:.*]] = arith.mulf %[[D4]], %[[D4]]
// CHECK-NEXT:  %[[T2:.*]] = arith.addf %[[T1]], %[[D5]]
// CHECK-NEXT:  %[[E0:.*]] = arith.mulf %arg4, %arg4
// CHECK-NEXT:  %[[E1:.*]] = arith.mulf %[[E0]], %[[E0]]
// CHECK-NEXT:  %[[E2:.*]] = arith.mulf %[[E1]], %[[E1]]
// CHECK-NEXT:  %[[E3:.*]] = arith.mulf %[[E2]], %[[E2]]
// CHECK-NEXT:  %[[E4:.*]] = arith.mulf %[[E3]], %[[E3]]
// CHECK-NEXT:  %[[E5:.*]] = arith.mulf %[[E4]], %[[E4]]
// CHECK-NEXT:  %[[T3:.*]] = arith.addf %[[T2]], %[[E5]]
// CHECK-NEXT:  tt.return %[[T3]]

// -----

// COM: Eight chains fed by three loads. The loads and the store keep their
// COM: relative order and stay ahead of the arithmetic, so no load is sunk
// COM: towards its uses, while the chains themselves are de-interleaved.

#blocked = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @loads_keep_their_order(%arg0: !tt.ptr<f64>, %arg1: !tt.ptr<f64>, %arg2: !tt.ptr<f64>, %out: !tt.ptr<f64>) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %sp0 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %ap0 = tt.addptr %sp0, %range : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    %ld0 = tt.load %ap0 : tensor<1024x!tt.ptr<f64>, #blocked>
    %sp1 = tt.splat %arg1 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %ap1 = tt.addptr %sp1, %range : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    %ld1 = tt.load %ap1 : tensor<1024x!tt.ptr<f64>, #blocked>
    %sp2 = tt.splat %arg2 : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %ap2 = tt.addptr %sp2, %range : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    %ld2 = tt.load %ap2 : tensor<1024x!tt.ptr<f64>, #blocked>
    %a0 = arith.mulf %ld0, %ld0 : tensor<1024xf64, #blocked>
    %b0 = arith.mulf %ld1, %ld1 : tensor<1024xf64, #blocked>
    %c0 = arith.mulf %ld2, %ld2 : tensor<1024xf64, #blocked>
    %d0 = arith.mulf %ld0, %ld0 : tensor<1024xf64, #blocked>
    %e0 = arith.mulf %ld1, %ld1 : tensor<1024xf64, #blocked>
    %f0 = arith.mulf %ld2, %ld2 : tensor<1024xf64, #blocked>
    %g0 = arith.mulf %ld0, %ld0 : tensor<1024xf64, #blocked>
    %h0 = arith.mulf %ld1, %ld1 : tensor<1024xf64, #blocked>
    %a1 = arith.mulf %a0, %a0 : tensor<1024xf64, #blocked>
    %b1 = arith.mulf %b0, %b0 : tensor<1024xf64, #blocked>
    %c1 = arith.mulf %c0, %c0 : tensor<1024xf64, #blocked>
    %d1 = arith.mulf %d0, %d0 : tensor<1024xf64, #blocked>
    %e1 = arith.mulf %e0, %e0 : tensor<1024xf64, #blocked>
    %f1 = arith.mulf %f0, %f0 : tensor<1024xf64, #blocked>
    %g1 = arith.mulf %g0, %g0 : tensor<1024xf64, #blocked>
    %h1 = arith.mulf %h0, %h0 : tensor<1024xf64, #blocked>
    %a2 = arith.mulf %a1, %a1 : tensor<1024xf64, #blocked>
    %b2 = arith.mulf %b1, %b1 : tensor<1024xf64, #blocked>
    %c2 = arith.mulf %c1, %c1 : tensor<1024xf64, #blocked>
    %d2 = arith.mulf %d1, %d1 : tensor<1024xf64, #blocked>
    %e2 = arith.mulf %e1, %e1 : tensor<1024xf64, #blocked>
    %f2 = arith.mulf %f1, %f1 : tensor<1024xf64, #blocked>
    %g2 = arith.mulf %g1, %g1 : tensor<1024xf64, #blocked>
    %h2 = arith.mulf %h1, %h1 : tensor<1024xf64, #blocked>
    %a3 = arith.mulf %a2, %a2 : tensor<1024xf64, #blocked>
    %b3 = arith.mulf %b2, %b2 : tensor<1024xf64, #blocked>
    %c3 = arith.mulf %c2, %c2 : tensor<1024xf64, #blocked>
    %d3 = arith.mulf %d2, %d2 : tensor<1024xf64, #blocked>
    %e3 = arith.mulf %e2, %e2 : tensor<1024xf64, #blocked>
    %f3 = arith.mulf %f2, %f2 : tensor<1024xf64, #blocked>
    %g3 = arith.mulf %g2, %g2 : tensor<1024xf64, #blocked>
    %h3 = arith.mulf %h2, %h2 : tensor<1024xf64, #blocked>
    %t0 = arith.addf %a3, %b3 : tensor<1024xf64, #blocked>
    %t1 = arith.addf %t0, %c3 : tensor<1024xf64, #blocked>
    %t2 = arith.addf %t1, %d3 : tensor<1024xf64, #blocked>
    %t3 = arith.addf %t2, %e3 : tensor<1024xf64, #blocked>
    %t4 = arith.addf %t3, %f3 : tensor<1024xf64, #blocked>
    %t5 = arith.addf %t4, %g3 : tensor<1024xf64, #blocked>
    %t6 = arith.addf %t5, %h3 : tensor<1024xf64, #blocked>
    %spo = tt.splat %out : !tt.ptr<f64> -> tensor<1024x!tt.ptr<f64>, #blocked>
    %apo = tt.addptr %spo, %range : tensor<1024x!tt.ptr<f64>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %apo, %t6 : tensor<1024x!tt.ptr<f64>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: tt.func @loads_keep_their_order
// CHECK:       %[[AP0:.*]] = tt.addptr
// CHECK-NEXT:  %[[L0:.*]] = tt.load %[[AP0]]
// CHECK:       %[[AP1:.*]] = tt.addptr
// CHECK-NEXT:  %[[L1:.*]] = tt.load %[[AP1]]
// CHECK:       %[[AP2:.*]] = tt.addptr
// CHECK-NEXT:  %[[L2:.*]] = tt.load %[[AP2]]
// CHECK-NEXT:  %[[A0:.*]] = arith.mulf %[[L0]], %[[L0]]
// CHECK-NEXT:  %[[A1:.*]] = arith.mulf %[[A0]], %[[A0]]
// CHECK-NEXT:  %[[A2:.*]] = arith.mulf %[[A1]], %[[A1]]
// CHECK-NEXT:  %[[A3:.*]] = arith.mulf %[[A2]], %[[A2]]
// CHECK-NEXT:  %[[B0:.*]] = arith.mulf %[[L1]], %[[L1]]
// CHECK-NEXT:  %[[B1:.*]] = arith.mulf %[[B0]], %[[B0]]
// CHECK-NEXT:  %[[B2:.*]] = arith.mulf %[[B1]], %[[B1]]
// CHECK-NEXT:  %[[B3:.*]] = arith.mulf %[[B2]], %[[B2]]
// CHECK-NEXT:  %[[T0:.*]] = arith.addf %[[A3]], %[[B3]]
// CHECK:       tt.store

// -----

// COM: A block holding a dot is never reordered, so matmul and attention
// COM: schedules are left alone. The arithmetic below is interleaved and over
// COM: budget, so the dot is the only thing keeping it in place.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dota = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dotb = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @block_with_dot(%a: tensor<8x16xf16, #dota>, %b: tensor<16x16xf16, #dotb>, %arg0: tensor<8x16xf64, #dpas>, %arg1: tensor<8x16xf64, #dpas>, %arg2: tensor<8x16xf64, #dpas>, %arg3: tensor<8x16xf64, #dpas>, %arg4: tensor<8x16xf64, #dpas>) -> (tensor<8x16xf32, #dpas>, tensor<8x16xf64, #dpas>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas>
    %acc = tt.dot %a, %b, %zero : tensor<8x16xf16, #dota> * tensor<16x16xf16, #dotb> -> tensor<8x16xf32, #dpas>
    %a0 = arith.mulf %arg0, %arg0 : tensor<8x16xf64, #dpas>
    %b0 = arith.mulf %arg1, %arg1 : tensor<8x16xf64, #dpas>
    %c0 = arith.mulf %arg2, %arg2 : tensor<8x16xf64, #dpas>
    %d0 = arith.mulf %arg3, %arg3 : tensor<8x16xf64, #dpas>
    %e0 = arith.mulf %arg4, %arg4 : tensor<8x16xf64, #dpas>
    %a1 = arith.mulf %a0, %a0 : tensor<8x16xf64, #dpas>
    %b1 = arith.mulf %b0, %b0 : tensor<8x16xf64, #dpas>
    %c1 = arith.mulf %c0, %c0 : tensor<8x16xf64, #dpas>
    %d1 = arith.mulf %d0, %d0 : tensor<8x16xf64, #dpas>
    %e1 = arith.mulf %e0, %e0 : tensor<8x16xf64, #dpas>
    %a2 = arith.mulf %a1, %a1 : tensor<8x16xf64, #dpas>
    %b2 = arith.mulf %b1, %b1 : tensor<8x16xf64, #dpas>
    %c2 = arith.mulf %c1, %c1 : tensor<8x16xf64, #dpas>
    %d2 = arith.mulf %d1, %d1 : tensor<8x16xf64, #dpas>
    %e2 = arith.mulf %e1, %e1 : tensor<8x16xf64, #dpas>
    %a3 = arith.mulf %a2, %a2 : tensor<8x16xf64, #dpas>
    %b3 = arith.mulf %b2, %b2 : tensor<8x16xf64, #dpas>
    %c3 = arith.mulf %c2, %c2 : tensor<8x16xf64, #dpas>
    %d3 = arith.mulf %d2, %d2 : tensor<8x16xf64, #dpas>
    %e3 = arith.mulf %e2, %e2 : tensor<8x16xf64, #dpas>
    %a4 = arith.mulf %a3, %a3 : tensor<8x16xf64, #dpas>
    %b4 = arith.mulf %b3, %b3 : tensor<8x16xf64, #dpas>
    %c4 = arith.mulf %c3, %c3 : tensor<8x16xf64, #dpas>
    %d4 = arith.mulf %d3, %d3 : tensor<8x16xf64, #dpas>
    %e4 = arith.mulf %e3, %e3 : tensor<8x16xf64, #dpas>
    %a5 = arith.mulf %a4, %a4 : tensor<8x16xf64, #dpas>
    %b5 = arith.mulf %b4, %b4 : tensor<8x16xf64, #dpas>
    %c5 = arith.mulf %c4, %c4 : tensor<8x16xf64, #dpas>
    %d5 = arith.mulf %d4, %d4 : tensor<8x16xf64, #dpas>
    %e5 = arith.mulf %e4, %e4 : tensor<8x16xf64, #dpas>
    %t0 = arith.addf %a5, %b5 : tensor<8x16xf64, #dpas>
    %t1 = arith.addf %t0, %c5 : tensor<8x16xf64, #dpas>
    %t2 = arith.addf %t1, %d5 : tensor<8x16xf64, #dpas>
    %t3 = arith.addf %t2, %e5 : tensor<8x16xf64, #dpas>
    tt.return %acc, %t3 : tensor<8x16xf32, #dpas>, tensor<8x16xf64, #dpas>
  }
}

// CHECK-LABEL: tt.func @block_with_dot
// CHECK:       tt.dot
// CHECK-NEXT:  %[[A0:.*]] = arith.mulf %arg2, %arg2
// CHECK-NEXT:  %[[B0:.*]] = arith.mulf %arg3, %arg3
// CHECK-NEXT:  %[[C0:.*]] = arith.mulf %arg4, %arg4
// CHECK-NEXT:  %[[D0:.*]] = arith.mulf %arg5, %arg5
// CHECK-NEXT:  %[[E0:.*]] = arith.mulf %arg6, %arg6
// CHECK-NEXT:  arith.mulf %[[A0]], %[[A0]]

// -----

// COM: The same interleaved shape and op count, but each value is a single f32
// COM: per lane, so the footprint is within budget and the order is left alone.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @within_budget(%arg0: tensor<32xf32, #blocked>, %arg1: tensor<32xf32, #blocked>, %arg2: tensor<32xf32, #blocked>, %arg3: tensor<32xf32, #blocked>, %arg4: tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    %a0 = arith.mulf %arg0, %arg0 : tensor<32xf32, #blocked>
    %b0 = arith.mulf %arg1, %arg1 : tensor<32xf32, #blocked>
    %c0 = arith.mulf %arg2, %arg2 : tensor<32xf32, #blocked>
    %d0 = arith.mulf %arg3, %arg3 : tensor<32xf32, #blocked>
    %e0 = arith.mulf %arg4, %arg4 : tensor<32xf32, #blocked>
    %a1 = arith.mulf %a0, %a0 : tensor<32xf32, #blocked>
    %b1 = arith.mulf %b0, %b0 : tensor<32xf32, #blocked>
    %c1 = arith.mulf %c0, %c0 : tensor<32xf32, #blocked>
    %d1 = arith.mulf %d0, %d0 : tensor<32xf32, #blocked>
    %e1 = arith.mulf %e0, %e0 : tensor<32xf32, #blocked>
    %a2 = arith.mulf %a1, %a1 : tensor<32xf32, #blocked>
    %b2 = arith.mulf %b1, %b1 : tensor<32xf32, #blocked>
    %c2 = arith.mulf %c1, %c1 : tensor<32xf32, #blocked>
    %d2 = arith.mulf %d1, %d1 : tensor<32xf32, #blocked>
    %e2 = arith.mulf %e1, %e1 : tensor<32xf32, #blocked>
    %a3 = arith.mulf %a2, %a2 : tensor<32xf32, #blocked>
    %b3 = arith.mulf %b2, %b2 : tensor<32xf32, #blocked>
    %c3 = arith.mulf %c2, %c2 : tensor<32xf32, #blocked>
    %d3 = arith.mulf %d2, %d2 : tensor<32xf32, #blocked>
    %e3 = arith.mulf %e2, %e2 : tensor<32xf32, #blocked>
    %a4 = arith.mulf %a3, %a3 : tensor<32xf32, #blocked>
    %b4 = arith.mulf %b3, %b3 : tensor<32xf32, #blocked>
    %c4 = arith.mulf %c3, %c3 : tensor<32xf32, #blocked>
    %d4 = arith.mulf %d3, %d3 : tensor<32xf32, #blocked>
    %e4 = arith.mulf %e3, %e3 : tensor<32xf32, #blocked>
    %a5 = arith.mulf %a4, %a4 : tensor<32xf32, #blocked>
    %b5 = arith.mulf %b4, %b4 : tensor<32xf32, #blocked>
    %c5 = arith.mulf %c4, %c4 : tensor<32xf32, #blocked>
    %d5 = arith.mulf %d4, %d4 : tensor<32xf32, #blocked>
    %e5 = arith.mulf %e4, %e4 : tensor<32xf32, #blocked>
    %t0 = arith.addf %a5, %b5 : tensor<32xf32, #blocked>
    %t1 = arith.addf %t0, %c5 : tensor<32xf32, #blocked>
    %t2 = arith.addf %t1, %d5 : tensor<32xf32, #blocked>
    %t3 = arith.addf %t2, %e5 : tensor<32xf32, #blocked>
    tt.return %t3 : tensor<32xf32, #blocked>
  }
}

// CHECK-LABEL: tt.func @within_budget
// CHECK:       %[[A0:.*]] = arith.mulf %arg0, %arg0
// CHECK-NEXT:  %[[B0:.*]] = arith.mulf %arg1, %arg1
// CHECK-NEXT:  %[[C0:.*]] = arith.mulf %arg2, %arg2
// CHECK-NEXT:  %[[D0:.*]] = arith.mulf %arg3, %arg3
// CHECK-NEXT:  %[[E0:.*]] = arith.mulf %arg4, %arg4
// CHECK-NEXT:  arith.mulf %[[A0]], %[[A0]]
