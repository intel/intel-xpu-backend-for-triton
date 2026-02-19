// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 support-2d-block-io=true support-dpas=true support-block-scale-dpas=false threads-per-warp=32' | FileCheck %s --check-prefix=CHECK-NO-BDPAS
// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 support-2d-block-io=true support-dpas=true support-block-scale-dpas=true threads-per-warp=32' | FileCheck %s --check-prefix=CHECK-BDPAS

module {
  // COM: Ensure that the 'threads-per-warp' attribute is set according to the option.
  // CHECK-NO-BDPAS: module attributes {"ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"}
  // CHECK-BDPAS: module attributes {"ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.target_arch = "spir64"}

  tt.func @kernel() {
    tt.return
  }
}

// -----

module {
  // COM: Ensure that the 'threads-per-warp' attribute is overwritten when the kernel contains a 'tt.dot'
  //      operation that can be lowered to DPAS instructions.
  // CHECK-NO-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"}
  // CHECK-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.target_arch = "spir64"}

  tt.func @kernel() {
    %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
    %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}

// -----

module {
  // COM: Ensure that the 'threads-per-warp' attribute is overwritten when the kernel contains a 'tt.dot'
  //      operation that can be lowered to DPAS instructions and uses tensor pointers.
  // CHECK-NO-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"}
  // CHECK-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.target_arch = "spir64"}

  tt.func @kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %a_ptr = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c1_i32] {order = array<i32: 0, 1>} : <tensor<128x32xf16>>
    %b_ptr = tt.make_tensor_ptr %arg1, [%c32_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c1_i32] {order = array<i32: 0, 1>} : <tensor<32x128xf16>>
    %c_ptr = tt.make_tensor_ptr %arg2, [%c128_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c1_i32] {order = array<i32: 0, 1>} : <tensor<128x128xf32>>
    %a = tt.load %a_ptr : !tt.ptr<tensor<128x32xf16>>
    %b = tt.load %b_ptr : !tt.ptr<tensor<32x128xf16>>
    %c = tt.load %c_ptr : !tt.ptr<tensor<128x128xf32>>
    %d = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}

// -----

module {
  // COM: Ensure that the 'threads-per-warp' attribute is overwritten when the kernel contains a 'tt.dot'
  //      operation that can be lowered to DPAS instructions and uses tensor descriptors.
  // CHECK-NO-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"}
  // CHECK-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_subgroup_scaled_matrix_multiply_accumulate, ttig.target_arch = "spir64"}

  tt.func @kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %a_desc = tt.make_tensor_descriptor %arg0, [%c128_i32, %c32_i32], [%c32_i64, %c1_i64] : <f16>, <tensor<128x32xf16>>
    %b_desc = tt.make_tensor_descriptor %arg1, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64] : <f16>, <tensor<32x128xf16>>
    %c_desc = tt.make_tensor_descriptor %arg2, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64] : <f32>, <tensor<128x128xf32>>
    %a = tt.descriptor_load %a_desc[%c0_i32]  : !tt.tensordesc<tensor<128x32xf16>> -> tensor<128x32xf16>
    %b = tt.descriptor_load %b_desc[%c0_i32]  : !tt.tensordesc<tensor<32x128xf16>> -> tensor<32x128xf16>
    %c = tt.descriptor_load %c_desc[%c0_i32]  : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}
