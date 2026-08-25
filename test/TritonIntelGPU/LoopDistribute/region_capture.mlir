// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute | FileCheck %s

// Test: %wg is computed by an scf.if whose then-branch captures a pure chain
// (arith.sitofp/arith.truncf/tt.splat) defined earlier in the loop body via a
// plain SSA reference into the nested region, not as an scf.if operand. Since
// the scf.if is pure (no side effects) and everything it captures is pure,
// it must be cloned -- together with the captured chain -- into the loop that
// needs %wg (the loop computing dot0). No crash, correct distribution.
// CHECK-LABEL: @region_capture_pure
// CHECK: scf.for
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[KF:.*]] = arith.sitofp
// CHECK:   %[[KB:.*]] = arith.truncf %[[KF]]
// CHECK:   %[[SPLAT:.*]] = tt.splat %[[KB]]
// CHECK:   %[[WG:.*]] = scf.if
// CHECK:     arith.mulf %[[SPLAT]]
// CHECK:   tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: scf.for
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK-NOT: scf.if
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg1
// CHECK:   tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @region_capture_pure(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %cond: i1) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cstb = arith.constant dense<1.000000e+00> : tensor<64x128xbf16>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      // Pure chain, only used from *inside* the scf.if region below (a capture,
      // not an operand of scf.if).
      %kf = arith.sitofp %k : i32 to f32
      %kb = arith.truncf %kf : f32 to bf16
      %splat = tt.splat %kb : bf16 -> tensor<64x128xbf16>
      %wg = scf.if %cond -> tensor<64x128xbf16> {
        %t = arith.mulf %splat, %cstb : tensor<64x128xbf16>
        scf.yield %t : tensor<64x128xbf16>
      } else {
        scf.yield %cstb : tensor<64x128xbf16>
      }
      %wfc = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: the scf.if computing %wg is NOT pure -- its then-branch performs a
// (non-volatile) tt.load -- but all of its memory effects, computed
// recursively, are reads, so it is replicable: repeating a read is
// observationally equivalent. The pointer tensor it loads from is itself a
// loop-body value captured into the region (a pure tt.addptr, not an operand of
// the scf.if), so the capture must be cloned along with it. Both feed dot0
// only, so they land in the first distributed loop and nowhere else. Compare
// region_capture_volatile_no_distribute below, where the same shape with a
// volatile load is rejected.
// CHECK-LABEL: @region_capture_read_only_load
// CHECK: scf.for
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[PTRS:.*]] = tt.addptr
// CHECK:   %[[WG:.*]] = scf.if
// CHECK:     tt.load %[[PTRS]]
// CHECK:   tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
// CHECK: scf.for
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK-NOT: scf.if
// CHECK-NOT: tt.load
// CHECK-NOT: tt.addptr
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   scf.yield
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @region_capture_read_only_load(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.tensordesc<64x128xbf16>, %cond: i1) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cstb = arith.constant dense<1.000000e+00> : tensor<64x128xbf16>
    %woff = arith.constant dense<64> : tensor<64x128xi32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %wbase = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wraw_ptrs = tt.addptr %wbase, %woff : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
      // %wraw_ptrs is captured by the scf.if region, it is NOT an operand of
      // scf.if. The load inside the region gives the scf.if a Read effect.
      %wg = scf.if %cond -> tensor<64x128xbf16> {
        %t = tt.load %wraw_ptrs : tensor<64x128x!tt.ptr<bf16>>
        scf.yield %t : tensor<64x128xbf16>
      } else {
        scf.yield %cstb : tensor<64x128xbf16>
      }
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: negative control for region_capture_read_only_load. Here the value
// captured by the scf.if (%wraw) is a *volatile* load. Note that %wg feeds
// dot0 only, so the chain would be cloned into the first distributed loop
// alone -- duplication is not what blocks this. A volatile tt.load declares a
// Write effect (see LoadOp::getEffects) exactly so that it is neither
// duplicated nor reordered, and distribution reorders it with respect to every
// memory operation of the second loop, so isReplicable rejects it and the loop
// must stay fused.
// CHECK-LABEL: @region_capture_volatile_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK-NOT: scf.for
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @region_capture_volatile_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.tensordesc<64x128xbf16>, %cond: i1) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cstb = arith.constant dense<1.000000e+00> : tensor<64x128xbf16>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %wraw_ptrs = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wraw = tt.load %wraw_ptrs {isVolatile = true} : tensor<64x128x!tt.ptr<bf16>>
      // %wraw is captured by the scf.if region, it is NOT an operand of scf.if.
      %wg = scf.if %cond -> tensor<64x128xbf16> {
        %t = arith.mulf %wraw, %cstb : tensor<64x128xbf16>
        scf.yield %t : tensor<64x128xbf16>
      } else {
        scf.yield %cstb : tensor<64x128xbf16>
      }
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: a carried, non-accumulator chain (%carry, index 2) reads dot0's
// accumulator (%acc_g) only as a REGION CAPTURE inside an scf.if's body --
// %acc_g is referenced directly from the then-branch, it is NOT passed as a
// formal top-level operand of the scf.if (whose only top-level operand is
// %cond). This is the same hazard as invalid.mlir's
// carried_depends_on_acc_no_distribute, but reached via nested-region capture
// rather than a direct top-level operand -- the safety check must walk into
// nested regions, not just each slice member's top-level operands, to catch it.
//
// Why it is rejected rather than placed in the loop that owns %acc_g: carried
// chains are replicated into BOTH distributed loops and their result is taken
// from the first one, so this chain would read the frozen pass-through %acc_g
// slot in the second loop. Keeping it in the first loop only would need
// per-iter_arg ownership (clone into the owning loop, freeze the slot in the
// other, wire the result from the owner) plus a check that the other loop never
// reads that iter_arg; see the follow-up issue.
// CHECK-LABEL: @carried_captures_accumulator_no_distribute
// CHECK: scf.for
// CHECK:   tt.dot
// CHECK-NOT: scf.for
// CHECK:   tt.dot
// CHECK:   scf.yield
// CHECK-NOT: scf.for
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_captures_accumulator_no_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>, %cond: i1) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %carry = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // %acc_g is captured directly inside the scf.if's region body -- it is
      // NOT a formal top-level operand of the scf.if (only %cond is).
      %new_carry = scf.if %cond -> tensor<128x128xf32> {
        %t = arith.addf %acc_g, %acc_g : tensor<128x128xf32>
        scf.yield %t : tensor<128x128xf32>
      } else {
        scf.yield %carry : tensor<128x128xf32>
      }
      scf.yield %d0, %d1, %new_carry : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}
