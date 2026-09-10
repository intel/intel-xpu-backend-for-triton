// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" | FileCheck %s --check-prefixes=CHECK,GRF128
// RUN: triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=256" | FileCheck %s --check-prefixes=CHECK,GRF256
// RUN: env TRITON_INTEL_HLC_STATS=1 triton-opt %s -split-input-file -tritonintelgpu-hoist-layout-conversions="grf-mode=default" 2>&1 | FileCheck %s --check-prefix=STATS

// STATS: [HoistLayoutConversions] considered={{[0-9]+}} hoisted={{[0-9]+}} rejected_pressure={{[0-9]+}} skipped_other={{[0-9]+}}

// COM: Case 1: Hoist ConvertLayoutOp with DotOperandEncoding out of scf.for loop.
// COM: The source of the convert_layout is defined outside the loop, so the pass
// COM: should move the conversion before the loop.
// COM: This hoist *lowers* pressure and so is taken in every GRF mode: the
// COM: #blocked source is 4x fatter per lane than the #dot_op result (256 vs 64
// COM: bytes/lane), and the hoist retires the source from the loop. Projected
// COM: live-in is 288 + 64 - 256 = 96 bytes/lane, matching the measured
// COM: post-hoist figure exactly (loop peak also falls, 484 -> 356).
// COM: This is the substitution accounting from
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7993; an
// COM: additive-only estimate would have projected 352 and rejected at 128-GRF.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_dot_op_cvt
  tt.func @hoist_dot_op_cvt(%arg0: tensor<128x16xf16, #blocked>, %arg1: tensor<16x16xf16, #dot_b>, %arg2: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dot_a>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a> * tensor<16x16xf16, #dot_b> -> tensor<128x16xf32, #dpas>
      scf.yield %dot : tensor<128x16xf32, #dpas>
    }
    tt.return %result : tensor<128x16xf32, #dpas>
  }
}

// -----

// COM: Case 2: Do NOT hoist ConvertLayoutOp whose destination is NOT DotOperandEncoding.
// COM: The convert_layout goes from #blocked to #dpas (not #dot_op), so it must stay
// COM: inside the loop.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_non_dot_op_cvt
  tt.func @no_hoist_non_dot_op_cvt(%arg0: tensor<128x16xf32, #blocked>, %arg1: tensor<128x16xf32, #dpas>) -> tensor<128x16xf32, #dpas> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf32, #{{.*}}> -> tensor<128x16xf32, #mma>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<128x16xf32, #dpas>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #dpas>
      %add = arith.addf %cvt, %acc : tensor<128x16xf32, #dpas>
      scf.yield %add : tensor<128x16xf32, #dpas>
    }
    tt.return %result : tensor<128x16xf32, #dpas>
  }
}

// -----

// COM: Case 3: Do NOT hoist ConvertLayoutOp when source is defined inside the loop.

#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a2 = #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth = 1}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_src_inside_loop
  tt.func @no_hoist_src_inside_loop(%arg0: !tt.ptr<f16>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth = 2}>>, %arg2: tensor<128x16xf32, #dpas2>) -> tensor<128x16xf32, #dpas2> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: tt.splat
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas2>) : i32 {
      %splat = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked2>
      %cvt = ttg.convert_layout %splat : tensor<128x16x!tt.ptr<f16>, #blocked2> -> tensor<128x16x!tt.ptr<f16>, #dot_a2>
      scf.yield %acc : tensor<128x16xf32, #dpas2>
    }
    tt.return %result : tensor<128x16xf32, #dpas2>
  }
}

// -----

// COM: Case 4: Do NOT hoist ConvertLayoutOp nested inside scf.if within scf.for.
// COM: The scf.if condition is loop-variant (%iv), so hoisting would make a
// COM: conditional conversion unconditional.

#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a3 = #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>
#dot_b3 = #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_inside_if_variant_cond
  tt.func @no_hoist_inside_if_variant_cond(%arg0: tensor<128x16xf16, #blocked3>, %arg1: tensor<16x16xf16, #dot_b3>, %arg2: tensor<128x16xf32, #dpas3>) -> tensor<128x16xf32, #dpas3> {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: arith.cmpi
    // CHECK: scf.if
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas3>) : i32 {
      %cond = arith.cmpi slt, %iv, %c4_i32 : i32
      %res = scf.if %cond -> (tensor<128x16xf32, #dpas3>) {
        %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked3> -> tensor<128x16xf16, #dot_a3>
        %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a3> * tensor<16x16xf16, #dot_b3> -> tensor<128x16xf32, #dpas3>
        scf.yield %dot : tensor<128x16xf32, #dpas3>
      } else {
        scf.yield %acc : tensor<128x16xf32, #dpas3>
      }
      scf.yield %res : tensor<128x16xf32, #dpas3>
    }
    tt.return %result : tensor<128x16xf32, #dpas3>
  }
}

// -----

// COM: Case 5: Do NOT hoist ConvertLayoutOp nested inside scf.if even with
// COM: loop-invariant condition. The pass conservatively skips any cvt not
// COM: directly in the ForOp's body region.

#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas4 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a4 = #ttg.dot_op<{opIdx = 0, parent = #dpas4, kWidth = 1}>
#dot_b4 = #ttg.dot_op<{opIdx = 1, parent = #dpas4, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_inside_if_invariant_cond
  tt.func @no_hoist_inside_if_invariant_cond(%arg0: tensor<128x16xf16, #blocked4>, %arg1: tensor<16x16xf16, #dot_b4>, %arg2: tensor<128x16xf32, #dpas4>, %cond: i1) -> tensor<128x16xf32, #dpas4> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: scf.if
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas4>) : i32 {
      %res = scf.if %cond -> (tensor<128x16xf32, #dpas4>) {
        %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked4> -> tensor<128x16xf16, #dot_a4>
        %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a4> * tensor<16x16xf16, #dot_b4> -> tensor<128x16xf32, #dpas4>
        scf.yield %dot : tensor<128x16xf32, #dpas4>
      } else {
        scf.yield %acc : tensor<128x16xf32, #dpas4>
      }
      scf.yield %res : tensor<128x16xf32, #dpas4>
    }
    tt.return %result : tensor<128x16xf32, #dpas4>
  }
}

// -----

// COM: Case 6: Do NOT hoist at either GRF mode when the loop body's live-in
// COM: register usage is already over the per-lane GRF budget.
// COM: With warpsPerCTA=[1,1] and threadsPerWarp=16, the live-in values to the
// COM: loop body block are (arg2 is NOT live-in — it becomes a block argument
// COM: via iter_args):
// COM:   - arg0 (256x64xf16, blocked):   ~2048 bytes/lane
// COM:   - arg1 (64x16xf16, dot_b):      ~128 bytes/lane
// COM: Live-in total: ~2176 bytes/lane. The hoist would retire arg0 and admit an
// COM: equally wide ~2048-byte #dot_a value, so the projection is 2176 + 2048 -
// COM: 2048 = 2176: exactly break-even, and still over budget on its own.
// COM: 128 GRF budget = 4096/16 * 0.80 = 204 bytes -> 2176 exceeds, do NOT hoist.
// COM: 256 GRF budget = 8192/16 * 0.80 = 409 bytes -> 2176 exceeds, do NOT hoist.
// COM:
// COM: This is also the case that pins the *non-goal* recorded in the pass
// COM: description: hoisting here would lower loop peak pressure (measured 5252
// COM: -> 4228) but *raise* whole-function peak (4224 -> 5248), because the
// COM: moment where source and result are both live moves out of the loop and
// COM: into the straight-line code before it. The gate does not measure that
// COM: program point, so the rejection above rests solely on the live-in figure.
// COM: Tracked as gap (b) of
// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.

#blocked6 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas6 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a6 = #ttg.dot_op<{opIdx = 0, parent = #dpas6, kWidth = 1}>
#dot_b6 = #ttg.dot_op<{opIdx = 1, parent = #dpas6, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @grf_pressure_test
  tt.func @grf_pressure_test(%arg0: tensor<256x64xf16, #blocked6>, %arg1: tensor<64x16xf16, #dot_b6>, %arg2: tensor<256x16xf32, #dpas6>) -> tensor<256x16xf32, #dpas6> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<256x64xf16, #{{.*}}> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<256x16xf32, #dpas6>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<256x64xf16, #blocked6> -> tensor<256x64xf16, #dot_a6>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<256x64xf16, #dot_a6> * tensor<64x16xf16, #dot_b6> -> tensor<256x16xf32, #dpas6>
      scf.yield %dot : tensor<256x16xf32, #dpas6>
    }
    tt.return %result : tensor<256x16xf32, #dpas6>
  }
}

// -----

// COM: Case 7: Hoist ConvertLayoutOp when source is a constant defined outside
// COM: the loop. The convert_layout should be moved after the arith.constant.
// COM:
// COM: This also covers a rematerializable source under the substitution model:
// COM: the hoist does retire %cst from the loop, but a constant contributes
// COM: nothing to live-in pressure in the first place (it is filtered out), so
// COM: there is nothing to credit back and the hoist is charged its full cost
// COM: (measured: liveIn=32 + 64 - 0 = 96). Crediting the source's raw type size
// COM: here would have produced a bogus negative.

#blocked7 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas7 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a7 = #ttg.dot_op<{opIdx = 0, parent = #dpas7, kWidth = 1}>
#dot_b7 = #ttg.dot_op<{opIdx = 1, parent = #dpas7, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_constant_source
  tt.func @hoist_constant_source(%arg0: tensor<16x16xf16, #dot_b7>, %arg1: tensor<128x16xf32, #dpas7>) -> tensor<128x16xf32, #dpas7> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked7>
    // CHECK: arith.constant dense<0.000000e+00> : tensor<128x16xf16, #{{.*}}>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK: scf.for
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<128x16xf32, #dpas7>) : i32 {
      %cvt = ttg.convert_layout %cst : tensor<128x16xf16, #blocked7> -> tensor<128x16xf16, #dot_a7>
      %dot = tt.dot %cvt, %arg0, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a7> * tensor<16x16xf16, #dot_b7> -> tensor<128x16xf32, #dpas7>
      scf.yield %dot : tensor<128x16xf32, #dpas7>
    }
    tt.return %result : tensor<128x16xf32, #dpas7>
  }
}

// -----

// COM: Case 8: Hoist ConvertLayoutOp out of an inner loop when its source is
// COM: loop-invariant w.r.t. the inner loop. The convert_layout moves before
// COM: the inner scf.for but remains inside the outer loop. Hoisting further
// COM: out of the outer loop is left as a follow-up.
// COM: (Same pressure arithmetic as case 1, so likewise taken in every GRF
// COM: mode: the hoist retires the fatter #blocked source from the inner loop.)

#blocked8 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas8 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a8 = #ttg.dot_op<{opIdx = 0, parent = #dpas8, kWidth = 1}>
#dot_b8 = #ttg.dot_op<{opIdx = 1, parent = #dpas8, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_from_inner_loop
  tt.func @hoist_from_inner_loop(%arg0: tensor<128x16xf16, #blocked8>, %arg1: tensor<16x16xf16, #dot_b8>, %arg2: tensor<128x16xf32, #dpas8>) -> tensor<128x16xf32, #dpas8> {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %outer = scf.for %oi = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%oacc = %arg2) -> (tensor<128x16xf32, #dpas8>) : i32 {
      %inner = scf.for %ii = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%iacc = %oacc) -> (tensor<128x16xf32, #dpas8>) : i32 {
        %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked8> -> tensor<128x16xf16, #dot_a8>
        %dot = tt.dot %cvt, %arg1, %iacc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a8> * tensor<16x16xf16, #dot_b8> -> tensor<128x16xf32, #dpas8>
        scf.yield %dot : tensor<128x16xf32, #dpas8>
      }
      scf.yield %inner : tensor<128x16xf32, #dpas8>
    }
    tt.return %outer : tensor<128x16xf32, #dpas8>
  }
}

// -----

// COM: Case 9: Do NOT hoist ConvertLayoutOp when the source is an iter_arg.
// COM: Iter args are block arguments of the loop body, so getDefiningOp()
// COM: returns null. They are loop-carried values that change every iteration,
// COM: so the convert_layout must remain inside the loop.

#blocked9 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas9 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a9 = #ttg.dot_op<{opIdx = 0, parent = #dpas9, kWidth = 1}>
#dot_b9 = #ttg.dot_op<{opIdx = 1, parent = #dpas9, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_hoist_iter_arg_source
  tt.func @no_hoist_iter_arg_source(%arg0: tensor<128x16xf16, #blocked9>, %arg1: tensor<16x16xf16, #dot_b9>, %arg2: tensor<128x16xf32, #dpas9>) -> tensor<128x16xf32, #dpas9> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: scf.for
    // CHECK: ttg.convert_layout
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%iter = %arg0) -> (tensor<128x16xf16, #blocked9>) : i32 {
      %cvt = ttg.convert_layout %iter : tensor<128x16xf16, #blocked9> -> tensor<128x16xf16, #dot_a9>
      %dot = tt.dot %cvt, %arg1, %arg2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a9> * tensor<16x16xf16, #dot_b9> -> tensor<128x16xf32, #dpas9>
      %updated = arith.addf %iter, %iter : tensor<128x16xf16, #blocked9>
      scf.yield %updated : tensor<128x16xf16, #blocked9>
    }
    tt.return %arg2 : tensor<128x16xf32, #dpas9>
  }
}

// -----

// COM: Case 10: Hoist ConvertLayoutOp at every GRF mode when the live-ins and
// COM: the hoisted tensor are all small enough to fit comfortably: live-in 96
// COM: bytes/lane, and the hoist swaps a 64-byte source for a 64-byte result, so
// COM: the projection stays at 96 vs. a 204-byte 128-GRF threshold.

#blocked10 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas10 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a10 = #ttg.dot_op<{opIdx = 0, parent = #dpas10, kWidth = 1}>
#dot_b10 = #ttg.dot_op<{opIdx = 1, parent = #dpas10, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @hoist_small_tensor
  tt.func @hoist_small_tensor(%arg0: tensor<8x16xf16, #blocked10>, %arg1: tensor<16x16xf16, #dot_b10>, %arg2: tensor<8x16xf32, #dpas10>) -> tensor<8x16xf32, #dpas10> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<8x16xf16, #{{.*}}> -> tensor<8x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: scf.for
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<8x16xf32, #dpas10>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<8x16xf16, #blocked10> -> tensor<8x16xf16, #dot_a10>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<8x16xf16, #dot_a10> * tensor<16x16xf16, #dot_b10> -> tensor<8x16xf32, #dpas10>
      scf.yield %dot : tensor<8x16xf32, #dpas10>
    }
    tt.return %result : tensor<8x16xf32, #dpas10>
  }
}

// -----

// COM: Case 11: Exercise the divide-by-32 default in
// COM: RegisterPressureAnalysis::getPerLaneGRFBudgetInBytes. This module
// COM: deliberately omits ttg.threads-per-warp, so getThreadsPerWarp returns its
// COM: 32 default and the per-hardware-thread budget is divided by 32 rather than
// COM: the DPAS-typical 16. All layouts here are 32-lane so the module still
// COM: verifies (deleting the attribute from a SIMD16 module would not).
// COM:
// COM: Measured pressure: live-in 144 bytes/lane (arg0 #blocked 128 + arg1
// COM: #dot_b 16; the iter args are block arguments, so they are not live-in),
// COM: plus 16 bytes/lane for the hoisted #dot_a value = 160 bytes/lane total.
// COM:
// COM: %arg0 is deliberately also consumed by the in-loop arith.addf, so the
// COM: hoist does NOT retire it and the cost is genuinely additive (this is also
// COM: the coverage for the not-retired branch of the substitution model). Were
// COM: the convert_layout its only in-loop use, the hoist would instead swap a
// COM: 128-byte source for a 16-byte result and pass at every GRF mode.
// COM:
// COM: Thresholds are 80% of the per-lane budget:
// COM:   grf-mode=default -> 4096/32 = 128 B/lane -> threshold 102 -> reject
// COM:   grf-mode=256     -> 8192/32 = 256 B/lane -> threshold 204 -> hoist
// COM: Were the divisor wrongly 16, the thresholds would be 204 and 409 and this
// COM: case would hoist at both modes, so the GRF128 rejection below is what pins
// COM: the divide-by-32 behavior.

#blocked11 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#dpas11 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a11 = #ttg.dot_op<{opIdx = 0, parent = #dpas11, kWidth = 1}>
#dot_b11 = #ttg.dot_op<{opIdx = 1, parent = #dpas11, kWidth = 2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func @default_threads_per_warp
  tt.func @default_threads_per_warp(%arg0: tensor<16x16xf16, #blocked11>, %arg1: tensor<16x16xf16, #dot_b11>, %arg2: tensor<16x16xf32, #dpas11>) -> tensor<16x16xf32, #dpas11> {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // GRF256: ttg.convert_layout %{{.*}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF128: scf.for
    // GRF128: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %result:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2, %bacc = %arg0) -> (tensor<16x16xf32, #dpas11>, tensor<16x16xf16, #blocked11>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked11> -> tensor<16x16xf16, #dot_a11>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<16x16xf16, #dot_a11> * tensor<16x16xf16, #dot_b11> -> tensor<16x16xf32, #dpas11>
      %sum = arith.addf %bacc, %arg0 : tensor<16x16xf16, #blocked11>
      scf.yield %dot, %sum : tensor<16x16xf32, #dpas11>, tensor<16x16xf16, #blocked11>
    }
    tt.return %result#0 : tensor<16x16xf32, #dpas11>
  }
}

// -----

// COM: Case 12: Two independent loop-invariant dot operands in one loop, the
// COM: shape of the _attn_bwd_dkdv inner loop (BLOCK_M1=32, BLOCK_N1=64,
// COM: HEAD_DIM=128, num_warps=8). Both conversions are pressure-*neutral*:
// COM: source and result are both 128 bytes/lane, and each hoist retires its own
// COM: source, so each contributes 128 - 128 = 0 to the running per-loop total.
// COM: The second candidate is therefore judged on the same projection as the
// COM: first, and at 256-GRF both are hoisted (measured: loop live-in 256
// COM: bytes/lane before and after, loop peak 1540 -> 1476, function peak
// COM: unchanged at 2560 -- so unlike case 6 this pair costs nothing at the
// COM: hoist site either).
// COM:
// COM: This is the regression this accounting fixes. Under additive-only
// COM: accounting the first hoist charged +128 to the loop, which pushed the
// COM: second over the threshold and rejected it, splitting the two operands.
// COM: See https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.
// COM:
// COM: The pass-through iter_args (%q_i, %do_i) are load-bearing, not noise:
// COM: routing %qT and %doT through the loop-carried arguments keeps them out of
// COM: the body's live-in set. Reading them directly would raise live-in from 256
// COM: to 1280 bytes/lane, which is over budget in every GRF mode, and the case
// COM: would then demonstrate nothing.

#blocked12 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#dpas12 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a12 = #ttg.dot_op<{opIdx = 0, parent = #dpas12, kWidth = 1}>
#dot_b12 = #ttg.dot_op<{opIdx = 1, parent = #dpas12, kWidth = 2}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @two_neutral_operands
  tt.func @two_neutral_operands(%k: tensor<64x128xf16, #blocked12>, %v: tensor<64x128xf16, #blocked12>,
                                %qT: tensor<128x32xf16, #dot_b12>, %doT: tensor<128x32xf16, #dot_b12>,
                                %dk0: tensor<64x32xf32, #dpas12>, %dv0: tensor<64x32xf32, #dpas12>)
                                -> (tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // GRF256: ttg.convert_layout %{{.*}} : tensor<64x128xf16, #{{.*}}> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: ttg.convert_layout %{{.*}} : tensor<64x128xf16, #{{.*}}> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF128: scf.for
    // GRF128: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<64x128xf16, #{{.*}}> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF128: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<64x128xf16, #{{.*}}> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %res:4 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32
        iter_args(%dk = %dk0, %dv = %dv0, %q_i = %qT, %do_i = %doT)
        -> (tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>,
            tensor<128x32xf16, #dot_b12>, tensor<128x32xf16, #dot_b12>) : i32 {
      %kc = ttg.convert_layout %k : tensor<64x128xf16, #blocked12> -> tensor<64x128xf16, #dot_a12>
      %qk = tt.dot %kc, %q_i, %dk, inputPrecision = tf32 : tensor<64x128xf16, #dot_a12> * tensor<128x32xf16, #dot_b12> -> tensor<64x32xf32, #dpas12>
      %vc = ttg.convert_layout %v : tensor<64x128xf16, #blocked12> -> tensor<64x128xf16, #dot_a12>
      %dp = tt.dot %vc, %do_i, %dv, inputPrecision = tf32 : tensor<64x128xf16, #dot_a12> * tensor<128x32xf16, #dot_b12> -> tensor<64x32xf32, #dpas12>
      scf.yield %qk, %dp, %q_i, %do_i : tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>, tensor<128x32xf16, #dot_b12>, tensor<128x32xf16, #dot_b12>
    }
    tt.return %res#0, %res#1 : tensor<64x32xf32, #dpas12>, tensor<64x32xf32, #dpas12>
  }
}

// -----

// COM: Case 13: Do NOT credit a retired source that is read *after* the loop.
// COM: Byte-for-byte the same kernel as case 1, plus one use of %arg0 below the
// COM: loop. %arg0 therefore occupies a register for the loop's whole duration
// COM: no matter where the conversion sits, so the hoist frees nothing and earns
// COM: no credit: the projection is the full 288 + 64 = 352 bytes/lane instead of
// COM: case 1's 288 + 64 - 256 = 96, and the hoist is rejected at 128-GRF
// COM: (threshold 204) while still fitting at 256-GRF (threshold 409).
// COM: Measured: loop live-in is 288 bytes/lane before the hoist and 352 after,
// COM: i.e. the projection is exact and case 1's credit would have undercounted
// COM: the real occupancy by 3.7x.

#blocked13 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas13 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a13 = #ttg.dot_op<{opIdx = 0, parent = #dpas13, kWidth = 1}>
#dot_b13 = #ttg.dot_op<{opIdx = 1, parent = #dpas13, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @no_credit_for_source_used_after_loop
  tt.func @no_credit_for_source_used_after_loop(%arg0: tensor<128x16xf16, #blocked13>, %arg1: tensor<16x16xf16, #dot_b13>, %arg2: tensor<128x16xf32, #dpas13>) -> (tensor<128x16xf32, #dpas13>, tensor<128x16xf16, #blocked13>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // GRF256: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // GRF256-NEXT: scf.for
    // GRF128: scf.for
    // GRF128: ttg.convert_layout %{{.*}} {tt.no_licm} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    %result = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2) -> (tensor<128x16xf32, #dpas13>) : i32 {
      %cvt = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked13> -> tensor<128x16xf16, #dot_a13>
      %dot = tt.dot %cvt, %arg1, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a13> * tensor<16x16xf16, #dot_b13> -> tensor<128x16xf32, #dpas13>
      scf.yield %dot : tensor<128x16xf32, #dpas13>
    }
    // COM: The use that keeps %arg0 live across the loop.
    %post = arith.addf %arg0, %arg0 : tensor<128x16xf16, #blocked13>
    tt.return %result, %post : tensor<128x16xf32, #dpas13>, tensor<128x16xf16, #blocked13>
  }
}

// -----

// COM: Cases 14 and 15: the same two candidates in one loop, written in the two
// COM: possible orders, must produce the same result. Two candidates:
// COM:   - "lean" %arg0: 128x16xf16 #blocked (256 bytes/lane) -> #dot_a (64), and
// COM:     the hoist retires it, so it contributes 64 - 256 = -192.
// COM:   - "fat" %argF: 16x16xf16 #blocked (32 bytes/lane) -> #dot_b (32), but
// COM:     %argF is also read by an in-loop arith.addf that no hoist can move, so
// COM:     it is never retired and the conversion contributes its full +32.
// COM: Loop live-in is 256 (%arg0) + 32 (%argB) + 32 (%argF) = 320 bytes/lane,
// COM: already past the 128-GRF threshold of 204, so the order of the two
// COM: decisions decides the outcome:
// COM:   - fat first:  320 + 32 = 352 >= 204, rejected -- and a rejection is
// COM:     irrevocable, since it stamps tt.no_licm.
// COM:   - lean first: 320 - 192 = 128 < 204, hoisted; the fat candidate is then
// COM:     judged at 128 + 32 = 160 < 204 and hoisted too.
// COM: The pass therefore decides a loop's candidates cheapest-first rather than
// COM: in program order, and both spellings below hoist both conversions (which
// COM: also fixes the hoisted pair's order: lean before fat, by cost).
// COM: See https://github.com/intel/intel-xpu-backend-for-triton/issues/7993.

#blocked14 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas14 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#dot_a14 = #ttg.dot_op<{opIdx = 0, parent = #dpas14, kWidth = 1}>
#dot_b14 = #ttg.dot_op<{opIdx = 1, parent = #dpas14, kWidth = 2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // COM: Case 14: the fat candidate is written first.
  // CHECK-LABEL: tt.func @order_independent_fat_first
  tt.func @order_independent_fat_first(%arg0: tensor<128x16xf16, #blocked14>, %argB: tensor<16x16xf16, #dot_b14>,
                                       %argF: tensor<16x16xf16, #blocked14>, %arg2: tensor<128x16xf32, #dpas14>)
                                       -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %res:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2, %acc2 = %arg2)
        -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) : i32 {
      %fat = ttg.convert_layout %argF : tensor<16x16xf16, #blocked14> -> tensor<16x16xf16, #dot_b14>
      %lean = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked14> -> tensor<128x16xf16, #dot_a14>
      %d1 = tt.dot %lean, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      %d2 = tt.dot %lean, %fat, %acc2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      // COM: Pins %argF inside the loop, so the fat conversion earns no credit.
      %pin = arith.addf %argF, %argF : tensor<16x16xf16, #blocked14>
      scf.yield %d1, %d2 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
    }
    tt.return %res#0, %res#1 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
  }

  // COM: Case 15: the same loop with the lean candidate written first.
  // CHECK-LABEL: tt.func @order_independent_lean_first
  tt.func @order_independent_lean_first(%arg0: tensor<128x16xf16, #blocked14>, %argB: tensor<16x16xf16, #dot_b14>,
                                        %argF: tensor<16x16xf16, #blocked14>, %arg2: tensor<128x16xf32, #dpas14>)
                                        -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.convert_layout %{{.*}} : tensor<128x16xf16, #{{.*}}> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #{{.*}}, kWidth = 1}>>
    // CHECK-NEXT: ttg.convert_layout %{{.*}} : tensor<16x16xf16, #{{.*}}> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #{{.*}}, kWidth = 2}>>
    // CHECK-NEXT: scf.for
    // CHECK-NOT: ttg.convert_layout
    %res:2 = scf.for %iv = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%acc = %arg2, %acc2 = %arg2)
        -> (tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>) : i32 {
      %lean = ttg.convert_layout %arg0 : tensor<128x16xf16, #blocked14> -> tensor<128x16xf16, #dot_a14>
      %fat = ttg.convert_layout %argF : tensor<16x16xf16, #blocked14> -> tensor<16x16xf16, #dot_b14>
      %d1 = tt.dot %lean, %argB, %acc, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      %d2 = tt.dot %lean, %fat, %acc2, inputPrecision = tf32 : tensor<128x16xf16, #dot_a14> * tensor<16x16xf16, #dot_b14> -> tensor<128x16xf32, #dpas14>
      // COM: Pins %argF inside the loop, so the fat conversion earns no credit.
      %pin = arith.addf %argF, %argF : tensor<16x16xf16, #blocked14>
      scf.yield %d1, %d2 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
    }
    tt.return %res#0, %res#1 : tensor<128x16xf32, #dpas14>, tensor<128x16xf32, #dpas14>
  }
}
