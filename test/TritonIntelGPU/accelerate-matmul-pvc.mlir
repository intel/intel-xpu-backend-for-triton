// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
// CHECK: #[[$DPAS_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: dpas_chain_loop
  tt.func public @dpas_chain_loop(
   %170: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
   %171: tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
   %179: tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %164: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>,
   %165: tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>>,
   %173: tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %153: tensor<128x64x!tt.ptr<f16, 1>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked2>
    // CHECK: scf.for
    // CHECK:   tt.dot {{.*}} -> tensor<128x16xf16, #[[$DPAS]]>
    // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #[[$DPAS_1]]>
    %115 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_0) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %172 = tt.dot %170, %171, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf16, #blocked>
      %178 = triton_gpu.convert_layout %172 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
      %180 = tt.dot %178, %179, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %180 : tensor<128x64xf16, #blocked1>
    }
    // CHECK: scf.for
    // CHECK:   tt.dot {{.*}} -> tensor<128x32xf16, #[[$DPAS_1]]>
    // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #[[$DPAS_1]]>
    %149 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %115) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %166 = tt.dot %164, %165, %cst_2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #blocked2>
      %172 = triton_gpu.convert_layout %166 : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
      %174 = tt.dot %172, %173, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %174 : tensor<128x64xf16, #blocked1>
    }
    tt.store %153, %149 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 8], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: chained_dot
  tt.func public @chained_dot(
    %arg0: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = triton_gpu.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK-NOT: triton_intel_gpu.dpas
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 8], threadsPerWarp = [4, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 8], threadsPerWarp = [1, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 8], threadsPerWarp = [2, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 8 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: dpas_chain_loop_ats
  tt.func public @dpas_chain_loop_ats(
   %170: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
   %171: tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
   %179: tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %164: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>,
   %165: tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>>,
   %173: tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %153: tensor<128x64x!tt.ptr<f16, 1>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked2>
    %115 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_0) -> (tensor<128x64xf16, #blocked1>) : i32 {
      // CHECK: scf.for
      // CHECK:   tt.dot {{.*}} -> tensor<128x16xf16, #blocked>
      %172 = tt.dot %170, %171, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf16, #blocked>
      %178 = triton_gpu.convert_layout %172 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
      // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #blocked1>
      %180 = tt.dot %178, %179, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %180 : tensor<128x64xf16, #blocked1>
    }
    %149 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %115) -> (tensor<128x64xf16, #blocked1>) : i32 {
      // CHECK: scf.for
      // CHECK:   tt.dot {{.*}} -> tensor<128x32xf16, #blocked2>
      %166 = tt.dot %164, %165, %cst_2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #blocked2>
      %172 = triton_gpu.convert_layout %166 : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>

      // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #blocked1>
      %174 = tt.dot %172, %173, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %174 : tensor<128x64xf16, #blocked1>
    }
    tt.store %153, %149 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-NOT: triton_intel_gpu.dpas
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 16], threadsPerWarp = [2, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 16], threadsPerWarp = [1, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 32], threadsPerWarp = [2, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 8 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: chained_dot
  tt.func public @chained_dot(
    %arg0: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #blocked>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = triton_gpu.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #blocked1>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK-LABEL: check_rep_cluster_size
  tt.func @check_rep_cluster_size(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %mask = arith.constant dense<true> : tensor<128x128xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #[[$DPAS]]>
    %result = tt.dot %a, %b, %zero_f32 : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  tt.func @check_rep_cluster_size(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: check_rep_cluster_size
    %mask = arith.constant dense<true> : tensor<16x128xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK: tt.dot {{.*}} -> tensor<16x128xf32, #[[$DPAS]]>
    %result = tt.dot %a, %b, %zero_f32 : tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x128xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  tt.func @check_rep_cluster_size(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: check_rep_cluster_size
    %mask = arith.constant dense<true> : tensor<128x16xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK: tt.dot {{.*}} -> tensor<128x16xf32, #[[$DPAS]]>
    %result = tt.dot %a, %b, %zero_f32 : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
