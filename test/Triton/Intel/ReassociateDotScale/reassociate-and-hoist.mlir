// RUN: triton-opt %s -triton-intel-reassociate-dot-scale=fast-math=true -triton-licm | FileCheck %s

//===----------------------------------------------------------------------===//
// COMBINED HOIST CHECK — verify the reassociation enables LICM
//===----------------------------------------------------------------------===//

module {
  // COM: After reassociation, the scaled operand should be hoisted above the loop
  // COM: Original: scale * dot(A_inv, B_var) inside loop
  // COM: After reassoc: dot(scale*A_inv, B_var) — the scale*A_inv is loop-invariant
  // COM: After LICM: scale*A_inv hoisted above loop, dot inside loop uses the hoisted value
  // CHECK-LABEL: tt.func public @hoist_after_reassociate
  tt.func public @hoist_after_reassociate(%a_inv: tensor<128x64xbf16>, %b_ptr: !tt.ptr<bf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
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

  // COM: The scaled operand computation should appear BEFORE the scf.for
  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<128x64xf32>
  // CHECK: %[[A_EXT:.*]] = arith.extf %{{.*}} : tensor<128x64xbf16> to tensor<128x64xf32>
  // CHECK: %[[SCALED_EXT:.*]] = arith.mulf %[[A_EXT]], %[[SCALE_SPLAT]] fastmath<reassoc> : tensor<128x64xf32>
  // CHECK: %[[SCALED_A:.*]] = arith.truncf %[[SCALED_EXT]] : tensor<128x64xf32> to tensor<128x64xbf16>
  // CHECK: %[[RESULT:.*]] = scf.for
  // CHECK: %[[B_LOAD:.*]] = tt.load
  // CHECK: %[[DOT:.*]] = tt.dot %[[SCALED_A]], %[[B_LOAD]]
  // CHECK: arith.addf {{.*}}, %[[DOT]]
}

// -----

module {
  // COM: Same test with B invariant instead of A
  // CHECK-LABEL: tt.func public @hoist_after_reassociate_b
  tt.func public @hoist_after_reassociate_b(%a_ptr: !tt.ptr<bf16>, %b_inv: tensor<64x128xbf16>, %scale_scalar: f32, %lb: i32, %ub: i32, %step: i32) -> tensor<128x128xf32> {
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

  // CHECK: %[[SCALE_SPLAT:.*]] = tt.splat %{{.*}} : f32 -> tensor<64x128xf32>
  // CHECK: %[[B_EXT:.*]] = arith.extf %{{.*}} : tensor<64x128xbf16> to tensor<64x128xf32>
  // CHECK: %[[SCALED_EXT:.*]] = arith.mulf %[[B_EXT]], %[[SCALE_SPLAT]] fastmath<reassoc> : tensor<64x128xf32>
  // CHECK: %[[SCALED_B:.*]] = arith.truncf %[[SCALED_EXT]] : tensor<64x128xf32> to tensor<64x128xbf16>
  // CHECK: %[[RESULT:.*]] = scf.for
  // CHECK: %[[A_LOAD:.*]] = tt.load
  // CHECK: %[[DOT:.*]] = tt.dot %[[A_LOAD]], %[[SCALED_B]]
  // CHECK: arith.addf {{.*}}, %[[DOT]]
}
