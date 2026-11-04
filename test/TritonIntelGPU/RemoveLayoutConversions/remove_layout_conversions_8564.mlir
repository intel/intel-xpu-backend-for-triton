// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s --enable-var-scope

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/8564
// COM: Two stores on different branches share the same pointer. One store's
// COM: value is MMA-derived, so forward propagation retargets the shared
// COM: pointer -- and with it the other branch's arith.constant store value --
// COM: to that encoding. Updating the constant in place used to only change
// COM: its result type, leaving `value` at the old type and tripping the
// COM: arith.constant verifier ("all of {value, result} have same type").

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#slice = #ttg.slice<{dim = 1, parent = #mma}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @constant_store_shares_ptr_with_mma_derived_store
  tt.func @constant_store_shares_ptr_with_mma_derived_store(%ptr_scalar: !tt.ptr<f32>, %cond: i1, %mma_val: tensor<16xf32, #slice>) {
    %ptr = tt.splat %ptr_scalar : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked>
    %converted = ttg.convert_layout %mma_val : tensor<16xf32, #slice> -> tensor<16xf32, #blocked>
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // CHECK: arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    tt.store %ptr, %zero : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  ^bb2:
    tt.store %ptr, %converted : tensor<16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
