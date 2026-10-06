// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// COM: An always false mask (the largest offset, 511+31, is smaller than the
// COM: bound 1000 for a sge comparison) on a load that has no `other` operand.
// COM: Because the mask is false no element is loaded, and since `other`
// COM: is absent the result of the load must be assigned zero. Replacing the
// COM: load result with the (null) `other` value used to crash the compiler.
tt.func public @always_false_no_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<32xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c512_i32 = arith.constant 512 : i32
  %bound = arith.constant dense<1000> : tensor<32xi32>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %base_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %r = scf.for %iv = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i32 {
    %iv_splat = tt.splat %iv : i32 -> tensor<32xi32>
    %offsets = arith.addi %iv_splat, %range : tensor<32xi32>
    %mask = arith.cmpi sge, %offsets, %bound : tensor<32xi32>
    %ptrs = tt.addptr %base_ptrs, %offsets : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %load = tt.load %ptrs, %mask : tensor<32x!tt.ptr<f32>>
    %new = arith.addf %acc, %load : tensor<32xf32>
    scf.yield %new : tensor<32xf32>
  }
  tt.return %r : tensor<32xf32>
}

// CHECK-LABEL: @always_false_no_other
// CHECK:       scf.for %[[IV:.*]] = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%[[ACC:.*]] = %cst)
// COM: The accumulation uses a zero constant rather than the load result.
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32xf32>
// CHECK:         arith.addf %[[ACC]], %[[ZERO]] : tensor<32xf32>
