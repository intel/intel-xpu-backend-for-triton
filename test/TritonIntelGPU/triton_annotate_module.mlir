// RUN: triton-opt %s --split-input-file -triton-annotate-module='target=xpu:DEVICE_ARCH.PVC support-sg-2d-block=true support-dpas=true threads-per-warp=32' | FileCheck %s

module {
  // COM: Ensure that the 'threads-per-warp' attribute is set according to the option.
  // CHECK: module attributes {triton_gpu.target = "xpu:DEVICE_ARCH.PVC", "triton_gpu.threads-per-warp" = 32 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block}
  tt.func @kernel() {
    tt.return
  }
}

// -----

module {
  // COM: Ensure that the 'threads-per-warp' attribute is overwritten when the kernel contains a 'tt.dot'
  //      operation that can be lowered to DPAS instructions.
  // CHECK: module attributes {triton_gpu.target = "xpu:DEVICE_ARCH.PVC", "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block}
  tt.func @kernel() {
    %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
    %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
    %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
    %d = tt.dot %a, %b, %c : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
    tt.return
  }
}
