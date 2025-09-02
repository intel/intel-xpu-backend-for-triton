// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

// COM: Test that `make_tensor_descriptor` is not rewritten when it is used by `descriptor_gather`.
tt.func public @test_descriptor_gather(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: tensor<32xi32>, %arg3: i32) {
  // CHECK-NOT: make_tensor_ptr
  // CHECK: tt.make_tensor_descriptor
 %c128_i32 = arith.constant 128 : i32
  %0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c128_i32], [%arg1, %arg1] : <f32>, <tensor<1x32xf32>>
  %1 = tt.descriptor_gather %0[%arg2, %arg3] : (!tt.tensordesc<tensor<1x32xf32>>, tensor<32xi32>, i32) -> tensor<32x32xf32>
  tt.return
}

// COM: Test that `descriptor_load/descriptor_store` operations are not rewritten if it they use a tensor descriptor function arg.
tt.func public @test_host_descriptor(%desc: !tt.tensordesc<tensor<2x16xf16>>) {
  // CHECK: tt.func public @test_host_descriptor([[DESC:%.*]]: !tt.tensordesc<tensor<2x16xf16>>) {
  // CHECK: tt.descriptor_load [[DESC]]
  // CHECK: tt.descriptor_store [[DESC]]
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = tt.descriptor_load %desc[%c2_i32, %c32_i32] : !tt.tensordesc<tensor<2x16xf16>> -> tensor<2x16xf16>
  tt.descriptor_store %desc[%c2_i32, %c32_i32], %0 : !tt.tensordesc<tensor<2x16xf16>>, tensor<2x16xf16>
  tt.return
}
