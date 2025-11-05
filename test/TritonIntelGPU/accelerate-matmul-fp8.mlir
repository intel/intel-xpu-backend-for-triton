// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_block_scale_dpas, ttig.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel_fp8_native(%arg1:  tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg2 :tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg3: tensor<128x128x!tt.ptr<f64>, #blocked1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // COM: Checks that we don't convert to fp16 when support_fp8_dpas is set
    // CHECK: tt.dot{{.*}} : tensor<128x128xf8E5M2{{.*}}*{{.*}}tensor<128x128xf8E5M2{{.*}}-> tensor<128x128xf32
    %res = tt.dot %arg1, %arg2, %cst : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %52 = ttg.convert_layout %res : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
    %53 = arith.extf %52 : tensor<128x128xf32, #blocked1> to tensor<128x128xf64, #blocked1>
    tt.store %arg3, %53 : tensor<128x128x!tt.ptr<f64>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @kernel_fp8_non_native(%arg1:  tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, %arg2 :tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, %arg3: tensor<128x128x!tt.ptr<f64>, #blocked1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    // COM: Checks that we convert to fp16 when support_fp8_dpas is not set
    // CHECK: %[[CONV1:.*]] = ttg.convert_layout %arg0 : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf8E5M2
    // CHECK: %[[CONV2:.*]] = ttg.convert_layout %arg1 : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf8E5M2
    // CHECK: %[[FPTOFP1:.*]] = tt.fp_to_fp %[[CONV1]] : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf16
    // CHECK: %[[FPTOFP2:.*]] = tt.fp_to_fp %[[CONV2]] : tensor<128x128xf8E5M2{{.*}} -> tensor<128x128xf16
    // CHECK: tt.dot %[[FPTOFP1]], %[[FPTOFP2]]{{.*}} : tensor<128x128xf16{{.*}}*{{.*}}tensor<128x128xf16{{.*}}-> tensor<128x128xf32

    %res = tt.dot %arg1, %arg2, %cst : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %52 = ttg.convert_layout %res : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked1>
    %53 = arith.extf %52 : tensor<128x128xf32, #blocked1> to tensor<128x128xf64, #blocked1>
    tt.store %arg3, %53 : tensor<128x128x!tt.ptr<f64>, #blocked1>
    tt.return
  }
}
