// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 support-sg-2d-block=true support-dpas=true support-block-scale-dpas=false threads-per-warp=32' | FileCheck %s --check-prefix=CHECK-NO-BDPAS
// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 support-sg-2d-block=true support-dpas=true support-block-scale-dpas=true threads-per-warp=32' | FileCheck %s --check-prefix=CHECK-BDPAS

module {
  // COM: Ensure that the 'threads-per-warp' attribute is set according to the option.
  // CHECK-NO-BDPAS: module attributes {"ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"}
  // CHECK-BDPAS: module attributes {"ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_block_scale_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"}

  tt.func @kernel() {
    tt.return
  }
}

// -----

module {
  // COM: Ensure that the 'threads-per-warp' attribute is overwritten when the kernel contains a 'tt.dot'
  //      operation that can be lowered to DPAS instructions.
  // CHECK-NO-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"}
  // CHECK-BDPAS: module attributes {"ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_block_scale_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"}

  tt.func @kernel() {
    %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
    %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}
