// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 support-2d-block-io=true support-dpas=true threads-per-warp=32' | FileCheck %s

// Test that module annotations are applied correctly without dot operations
// CHECK: module attributes {"ttg.threads-per-warp" = 32 : i32{{.*}}ttig.min_sg_size = 16
module {
  tt.func @kernel_no_dots(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    tt.store %arg0, %cst : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Test that warp size IS adjusted when DPAS-compatible dot is present
// CHECK: module attributes {"ttg.threads-per-warp" = 16 : i32{{.*}}ttig.min_sg_size = 16
module {
  tt.func @kernel_with_dpas_dot(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.00e+00> : tensor<128x64xf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Test with integer dot operation
// CHECK: module attributes {"ttg.threads-per-warp" = 16 : i32{{.*}}ttig.min_sg_size = 16
module {
  tt.func @kernel_with_int_dot(%arg0: tensor<128x128x!tt.ptr<i32>>) {
    %a = arith.constant dense<1> : tensor<128x64xi8>
    %b = arith.constant dense<2> : tensor<64x128xi8>
    %c = arith.constant dense<0> : tensor<128x128xi32>
    %d = tt.dot %a, %b, %c : tensor<128x64xi8> * tensor<64x128xi8> -> tensor<128x128xi32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// Test with incompatible dot (mixed types) - warp size should NOT change
// CHECK: module attributes {"ttg.threads-per-warp" = 32 : i32{{.*}}ttig.min_sg_size = 16
module {
  tt.func @kernel_incompatible_dot(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.00e+00> : tensor<128x64xf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x128xbf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=8 support-2d-block-io=true support-dpas=true threads-per-warp=32' | FileCheck %s --check-prefix=CHECK-SG8

// Test with min_sg_size=8 - should NOT use DPAS, warp size stays 32
// CHECK-SG8: module attributes {"ttg.threads-per-warp" = 32 : i32{{.*}}ttig.min_sg_size = 8
module {
  tt.func @kernel_min_sg_8(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.00e+00> : tensor<128x64xf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 support-dpas=false threads-per-warp=16' | FileCheck %s --check-prefix=CHECK-NO-DPAS

// Test without DPAS support - warp size stays as configured
// CHECK-NO-DPAS: module attributes {"ttg.threads-per-warp" = 16 : i32{{.*}}ttig.min_sg_size = 16
// CHECK-NO-DPAS-NOT: ttig.support_subgroup_matrix_multiply_accumulate
module {
  tt.func @kernel_no_dpas_attr(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.00e+00> : tensor<128x64xf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Test multiple functions - all must be DPAS-compatible for warp adjustment
// CHECK: module attributes {"ttg.threads-per-warp" = 16 : i32
module {
  tt.func @kernel_func1(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.00e+00> : tensor<128x64xf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @kernel_func2(%arg0: tensor<128x128x!tt.ptr<f32>>) {
    %a = arith.constant dense<1.00e+00> : tensor<128x64xbf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x128xbf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
    tt.store %arg0, %d : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}
