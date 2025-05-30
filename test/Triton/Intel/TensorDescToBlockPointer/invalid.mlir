// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

// COM: Test make_tensor_descriptor is not rewritten when it is used by descriptor_gather.
// CHECK-NOT: make_tensor_ptr
// CHECK: tt.make_tensor_descriptor
module {
  tt.func public @test_descriptor_gather(%arg0: !tt.ptr<f32>, %arg1: i64, %arg2: tensor<32xi32>, %arg3: i32) {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c128_i32, %c128_i32], [%arg1, %arg1] : <f32>, <tensor<1x32xf32>>
    %1 = tt.descriptor_gather %0[%arg2, %arg3] : (!tt.tensordesc<tensor<1x32xf32>>, tensor<32xi32>, i32) -> tensor<32x32xf32>
    tt.return
  }
}
