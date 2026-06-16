// RUN: triton-opt %s -split-input-file -triton-intel-reassociate-dot-scale=fast-math=true | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-intel-reassociate-dot-scale=fast-math=false | FileCheck %s --check-prefix=NOFAST

//===----------------------------------------------------------------------===//
// POSITIVE TESTS — fast-math=true must rewrite
//===----------------------------------------------------------------------===//

// -----

module {
  // COM: Basic case — bf16 operands, f32 scale via tt.splat, A loop-invariant, B loop-variant
  // COM: Expect: splat scale to A's shape → extf A to f32 → mulf <reassoc> → truncf back to bf16 → new tt.dot
  // CHECK-LABEL: tt.func public @scale_invariant_a
  tt.func public @scale_invariant_a(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    // COM: Loop — A is invariant (defined before loop), B is variant (loaded inside)
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      // COM: Original pattern: dot result used by mulf with scale
      %dot_res = tt.dot %a_inv, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[B_LOAD:.*]] = tt.load
  // COM: The scale should be splatted to A's shape in compute type (f32)
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<128x64xf32>
  // COM: Extend A from bf16 to f32
  // CHECK: %[[A_EXT:.*]] = arith.extf %{{.*}} : tensor<128x64xbf16> to tensor<128x64xf32>
  // COM: Multiply with reassoc flag
  // CHECK: %[[SCALED_EXT:.*]] = arith.mulf %[[A_EXT]], %[[SCALE_SPLAT]] fastmath<reassoc> : tensor<128x64xf32>
  // COM: Truncate back to bf16
  // CHECK: %[[SCALED_A:.*]] = arith.truncf %[[SCALED_EXT]] : tensor<128x64xf32> to tensor<128x64xbf16>
  // COM: New dot with scaled A operand
  // CHECK: %[[NEW_DOT:.*]] = tt.dot %[[SCALED_A]], %[[B_LOAD]], {{.*}} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  // COM: Original mulf consuming dot should be GONE — new dot directly replaces it
  // CHECK-NOT: arith.mulf {{.*}}, %{{.*dot.*}}
  // CHECK: arith.addf {{.*}}, %[[NEW_DOT]]
}

// -----

module {
  // COM: Symmetric case — B loop-invariant, A loop-variant → scale pulled into B
  // CHECK-LABEL: tt.func public @scale_invariant_b
  tt.func public @scale_invariant_b(%a_ptr: !tt.ptr<bf16>, %b_inv: tensor<64x128xbf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %a_offset = arith.muli %iv, %c64 : i32
      %a_ptr_off = tt.addptr %a_ptr, %a_offset : !tt.ptr<bf16>, i32
      %a_ptr_splat = tt.splat %a_ptr_off : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
      %a_var = tt.load %a_ptr_splat : tensor<128x64x!tt.ptr<bf16>>

      %dot_res = tt.dot %a_var, %b_inv, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %dot_res, %scale_splat : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[A_LOAD:.*]] = tt.load
  // COM: Scale splatted to B's shape
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<64x128xf32>
  // CHECK: %[[B_EXT:.*]] = arith.extf %{{.*}} : tensor<64x128xbf16> to tensor<64x128xf32>
  // CHECK: %[[SCALED_EXT:.*]] = arith.mulf %[[B_EXT]], %[[SCALE_SPLAT]] fastmath<reassoc> : tensor<64x128xf32>
  // CHECK: %[[SCALED_B:.*]] = arith.truncf %[[SCALED_EXT]] : tensor<64x128xf32> to tensor<64x128xbf16>
  // CHECK: %[[NEW_DOT:.*]] = tt.dot %[[A_LOAD]], %[[SCALED_B]], {{.*}} : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
  // CHECK-NOT: arith.mulf {{.*}}, %{{.*dot.*}}
  // CHECK: arith.addf {{.*}}, %[[NEW_DOT]]
}

// -----

module {
  // COM: Scale via arith.constant dense<X> instead of tt.splat — still rewrites
  // CHECK-LABEL: tt.func public @scale_constant_splat
  tt.func public @scale_constant_splat(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %scale_splat = arith.constant dense<2.0> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      %dot_res = tt.dot %a_inv, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // COM: Constant splat hoisted outside loop before scf.for
  // CHECK: %[[SCALE_CONST:.*]] = arith.constant dense<2.000000e+00> : tensor<128x64xf32>
  // CHECK: scf.for
  // CHECK: %[[B_LOAD:.*]] = tt.load
  // CHECK: %[[A_EXT:.*]] = arith.extf
  // CHECK: %[[SCALED_EXT:.*]] = arith.mulf %[[A_EXT]], %[[SCALE_CONST]] fastmath<reassoc>
  // CHECK: %[[SCALED_A:.*]] = arith.truncf %[[SCALED_EXT]]
  // CHECK: %[[NEW_DOT:.*]] = tt.dot %[[SCALED_A]], %[[B_LOAD]]
}

// -----

module {
  // COM: Operands already f32 (scale also f32) — NO extf/truncf emitted
  // CHECK-LABEL: tt.func public @scale_no_type_conversion
  tt.func public @scale_no_type_conversion(%a_inv: tensor<128x64xf32>, %b_ptr: !tt.ptr<f32>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<f32>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<f32>>

      %dot_res = tt.dot %a_inv, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[B_LOAD:.*]] = tt.load
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<128x64xf32>
  // COM: No extf — operand already f32
  // CHECK-NOT: arith.extf
  // CHECK: %[[SCALED_A:.*]] = arith.mulf %{{.*}}, %[[SCALE_SPLAT]] fastmath<reassoc> : tensor<128x64xf32>
  // COM: No truncf — no type widening
  // CHECK-NOT: arith.truncf
  // CHECK: %[[NEW_DOT:.*]] = tt.dot %[[SCALED_A]], %[[B_LOAD]]
}

//===----------------------------------------------------------------------===//
// NEGATIVE TESTS — must NOT rewrite (IR structure preserved)
//===----------------------------------------------------------------------===//

// -----

module {
  // COM: fast-math=false — pass is no-op
  // NOFAST-LABEL: tt.func public @no_rewrite_fast_math_disabled
  tt.func public @no_rewrite_fast_math_disabled(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      %dot_res = tt.dot %a_inv, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // NOFAST: scf.for
  // NOFAST: %[[DOT:.*]] = tt.dot
  // NOFAST: %[[SCALE_SPLAT:.*]] = tt.splat
  // NOFAST: %[[SCALED_DOT:.*]] = arith.mulf %[[SCALE_SPLAT]], %[[DOT]]
  // NOFAST: arith.addf {{.*}}, %[[SCALED_DOT]]
}

// -----

module {
  // COM: Nonzero accumulator — no rewrite
  // CHECK-LABEL: tt.func public @no_rewrite_nonzero_acc
  tt.func public @no_rewrite_nonzero_acc(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %nonzero_acc = arith.constant dense<1.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      // COM: accumulator is nonzero constant
      %dot_res = tt.dot %a_inv, %b_var, %nonzero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[DOT:.*]] = tt.dot {{.*}}, {{.*}}, %{{.*}}
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat
  // CHECK: %[[SCALED_DOT:.*]] = arith.mulf %[[SCALE_SPLAT]], %[[DOT]]
  // CHECK: arith.addf {{.*}}, %[[SCALED_DOT]]
  // COM: No scaled operand (no mulf with reassoc flag), no new dot with pre-scaled input
  // CHECK-NOT: arith.mulf {{.*}} fastmath<reassoc>
}

// -----

module {
  // COM: Dot result has TWO uses — no rewrite
  // CHECK-LABEL: tt.func public @no_rewrite_dot_two_uses
  tt.func public @no_rewrite_dot_two_uses(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      %dot_res = tt.dot %a_inv, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      // COM: Second use of dot_res
      %acc_tmp = arith.addf %acc, %dot_res : tensor<128x128xf32>
      %acc_new = arith.addf %acc_tmp, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[DOT:.*]] = tt.dot
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat
  // CHECK: %[[SCALED_DOT:.*]] = arith.mulf %[[SCALE_SPLAT]], %[[DOT]]
  // CHECK: arith.addf {{.*}}, %[[DOT]]
  // CHECK: arith.addf {{.*}}, %[[SCALED_DOT]]
  // COM: Original structure preserved
  // CHECK-NOT: arith.mulf {{.*}} fastmath<reassoc>
}

// -----

module {
  // COM: Both operands loop-invariant (whole dot is invariant) — no rewrite
  // CHECK-LABEL: tt.func public @no_rewrite_both_invariant
  tt.func public @no_rewrite_both_invariant(%a_inv: tensor<128x64xbf16>, %b_inv: tensor<64x128xbf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %dot_res = tt.dot %a_inv, %b_inv, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[DOT:.*]] = tt.dot
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat
  // CHECK: arith.mulf %[[SCALE_SPLAT]], %[[DOT]]
  // CHECK-NOT: arith.mulf {{.*}} fastmath<reassoc>
}

// -----

module {
  // COM: Both operands loop-variant — no rewrite
  // CHECK-LABEL: tt.func public @no_rewrite_both_variant
  tt.func public @no_rewrite_both_variant(%a_ptr: !tt.ptr<bf16>, %b_ptr: !tt.ptr<bf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %a_offset = arith.muli %iv, %c64 : i32
      %a_ptr_off = tt.addptr %a_ptr, %a_offset : !tt.ptr<bf16>, i32
      %a_ptr_splat = tt.splat %a_ptr_off : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
      %a_var = tt.load %a_ptr_splat : tensor<128x64x!tt.ptr<bf16>>

      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      %dot_res = tt.dot %a_var, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[A_LOAD:.*]] = tt.load
  // CHECK: %[[B_LOAD:.*]] = tt.load
  // CHECK: %[[DOT:.*]] = tt.dot %[[A_LOAD]], %[[B_LOAD]]
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat
  // CHECK: arith.mulf %[[SCALE_SPLAT]], %[[DOT]]
  // CHECK-NOT: arith.mulf {{.*}} fastmath<reassoc>
}

// -----

module {
  // COM: Dot NOT inside any loop — no rewrite
  // CHECK-LABEL: tt.func public @no_rewrite_no_loop
  tt.func public @no_rewrite_no_loop(%a: tensor<128x64xbf16>, %b: tensor<64x128xbf16>, %scale_scalar: f32) -> tensor<128x128xf32> {
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>

    %dot_res = tt.dot %a, %b, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
    %scale_splat = tt.splat %scale_scalar : f32 -> tensor<128x128xf32>
    %scaled_dot = arith.mulf %scale_splat, %dot_res : tensor<128x128xf32>

    tt.return %scaled_dot : tensor<128x128xf32>
  }

  // CHECK: %[[DOT:.*]] = tt.dot
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat
  // CHECK: %[[SCALED_DOT:.*]] = arith.mulf %[[SCALE_SPLAT]], %[[DOT]]
  // CHECK: tt.return %[[SCALED_DOT]]
  // CHECK-NOT: arith.mulf {{.*}} fastmath<reassoc>
}

// -----

module {
  // COM: Non-uniform scale (a loaded tensor, not a scalar splat/constant) — no
  // COM: rewrite. This is the out-of-scope per-token-head scale case: the scale
  // COM: is not a uniform scalar broadcast so it cannot be pulled into an operand.
  // CHECK-LABEL: tt.func public @no_rewrite_non_uniform_scale
  tt.func public @no_rewrite_non_uniform_scale(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %scale_ptr: !tt.ptr<f32>, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
    %c64 = arith.constant 64 : i32
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %scale_splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    %scale_non_uniform = tt.load %scale_splat : tensor<128x128x!tt.ptr<f32>>

    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero_acc) -> (tensor<128x128xf32>) : i32 {
      %b_offset = arith.muli %iv, %c64 : i32
      %b_ptr_off = tt.addptr %b_ptr, %b_offset : !tt.ptr<bf16>, i32
      %b_ptr_splat = tt.splat %b_ptr_off : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
      %b_var = tt.load %b_ptr_splat : tensor<64x128x!tt.ptr<bf16>>

      %dot_res = tt.dot %a_inv, %b_var, %zero_acc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %scaled_dot = arith.mulf %scale_non_uniform, %dot_res : tensor<128x128xf32>

      %acc_new = arith.addf %acc, %scaled_dot : tensor<128x128xf32>
      scf.yield %acc_new : tensor<128x128xf32>
    }
    tt.return %result : tensor<128x128xf32>
  }

  // CHECK: scf.for
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}
  // CHECK: arith.mulf %{{.*}}, %[[DOT]]
  // CHECK-NOT: arith.mulf {{.*}} fastmath<reassoc>
}
