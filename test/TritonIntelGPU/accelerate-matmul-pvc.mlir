// RUN: TRITON_INTEL_UPCASTMXFP_DOTOP_ENCODING=1 triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
// CHECK: #[[$DPAS_1:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: dpas_chain_loop
  tt.func public @dpas_chain_loop(
   %170: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   %171: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %179: tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %164: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
   %165: tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>,
   %173: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
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
      %172 = tt.dot %170, %171, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf16, #blocked>
      %178 = ttg.convert_layout %172 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %180 = tt.dot %178, %179, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %180 : tensor<128x64xf16, #blocked1>
    }
    // CHECK: scf.for
    // CHECK:   tt.dot {{.*}} -> tensor<128x32xf16, #[[$DPAS_1]]>
    // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #[[$DPAS_1]]>
    %149 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %115) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %166 = tt.dot %164, %165, %cst_2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #blocked2>
      %172 = ttg.convert_layout %166 : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %174 = tt.dot %172, %173, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %174 : tensor<128x64xf16, #blocked1>
    }
    tt.store %153, %149 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: chained_dot
  tt.func public @chained_dot(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK-NOT: triton_intel_gpu.dpas
#blocked = #ttg.blocked<{sizePerThread = [8, 8], threadsPerWarp = [4, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 8], threadsPerWarp = [1, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 8], threadsPerWarp = [2, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 8 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: dpas_chain_loop_ats
  tt.func public @dpas_chain_loop_ats(
   %170: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   %171: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %179: tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %164: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
   %165: tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>,
   %173: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>,
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
      %172 = tt.dot %170, %171, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf16, #blocked>
      %178 = ttg.convert_layout %172 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #blocked1>
      %180 = tt.dot %178, %179, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %180 : tensor<128x64xf16, #blocked1>
    }
    %149 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %115) -> (tensor<128x64xf16, #blocked1>) : i32 {
      // CHECK: scf.for
      // CHECK:   tt.dot {{.*}} -> tensor<128x32xf16, #blocked2>
      %166 = tt.dot %164, %165, %cst_2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #blocked2>
      %172 = ttg.convert_layout %166 : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>

      // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #blocked1>
      %174 = tt.dot %172, %173, %arg16 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %174 : tensor<128x64xf16, #blocked1>
    }
    tt.store %153, %149 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-NOT: triton_intel_gpu.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [2, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [1, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [2, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 8 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK: chained_dot
  tt.func public @chained_dot(
    %arg0: tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #blocked>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #blocked1>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK-LABEL: check_rep_cluster_size
  tt.func @check_rep_cluster_size(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %mask = arith.constant dense<true> : tensor<128x128xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #[[$DPAS]]>
    %result = tt.dot %a, %b, %zero_f32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  tt.func @check_rep_cluster_size(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: check_rep_cluster_size
    %mask = arith.constant dense<true> : tensor<16x128xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK: tt.dot {{.*}} -> tensor<16x128xf32, #[[$DPAS]]>
    %result = tt.dot %a, %b, %zero_f32 : tensor<16x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x128xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[$DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1], A = [32, 16], B = [16, 16], C = [32, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  tt.func @check_rep_cluster_size(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: check_rep_cluster_size
    %mask = arith.constant dense<true> : tensor<128x16xi1, #blocked>
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

    // CHECK: tt.dot {{.*}} -> tensor<128x16xf32, #[[$DPAS]]>
    %result = tt.dot %a, %b, %zero_f32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result, %mask : tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>

module attributes {"ttg.target" = "xpu", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32} {
  // CHECK-DAG: [[BLOCKED:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: [[BLOCKED1:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: [[BLOCKED2:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
  // CHECK-DAG: [[DPAS:#.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
  // CHECK-DAG: [[DPAS1:#.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>

  // CHECK: tt.func @dot_scaled([[ARG0:%.*]]: tensor<128x32xi8, [[BLOCKED]]>, [[ARG1:%.*]]: tensor<128x2xi8, [[BLOCKED1]]>, [[ARG2:%.*]]: tensor<64x128xbf16, [[BLOCKED2]]>) -> tensor<128x128xf32, [[BLOCKED2]]> {
  tt.func @dot_scaled(%a: tensor<128x32xi8, #blocked2>, %scale: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xbf16, #blocked>) -> tensor<128x128xf32, #blocked> {
    // CHECK: [[CST:%.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, [[BLOCKED2]]>
    // CHECK: [[C:%.*]] = ttg.convert_layout [[CST]] : tensor<128x128xf32, [[BLOCKED2]]> -> tensor<128x128xf32, [[DPAS]]>
    // CHECK: [[CVT_ARG0:%.*]] = ttg.convert_layout [[ARG0]] : tensor<128x32xi8, [[BLOCKED]]> -> tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = [[DPAS1]], kWidth = 4}>>
    // CHECK: [[CVT_ARG1:%.*]] = ttg.convert_layout [[ARG1]] : tensor<128x2xi8, [[BLOCKED1]]> -> tensor<128x2xi8, [[BLOCKED1]]>
    // CHECK: [[A:%.*]] = ttg.upcast_mxfp [[CVT_ARG0]], [[CVT_ARG1]] fp_type = e2m1 : tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = [[DPAS1]], kWidth = 4}>>, tensor<128x2xi8, [[BLOCKED1]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>>
    // CHECK: [[B:%.*]] = ttg.convert_layout [[ARG2]] : tensor<64x128xbf16, [[BLOCKED2]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>>
    // CHECK: [[D:%.*]] = tt.dot [[A]], [[B]], [[C]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>> -> tensor<128x128xf32, [[DPAS]]>
    // CHECK: [[RES:%.*]] = ttg.convert_layout {{.*}} : tensor<128x128xf32, [[DPAS]]> -> tensor<128x128xf32, [[BLOCKED2]]>

    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %dot_res1 = tt.dot_scaled %a scale %scale, %b, %cst lhs = e2m1 rhs = bf16 : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xbf16, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %dot_res1 : tensor<128x128xf32, #blocked>
  }

  // CHECK: tt.func @dot_scaled_fp8([[ARG0:%.*]]: tensor<128x32xi8, [[BLOCKED]]>, [[ARG1:%.*]]: tensor<128x2xi8, [[BLOCKED1]]>, [[ARG2:%.*]]: tensor<64x128xf8E4M3FN, [[BLOCKED2]]>) -> tensor<128x128xf32, [[BLOCKED2]]> {
  tt.func @dot_scaled_fp8(%a: tensor<128x32xi8, #blocked2>, %scale: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf8E4M3FN, #blocked>) -> tensor<128x128xf32, #blocked> {
    // CHECK: [[CST:%.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, [[BLOCKED2]]>
    // CHECK: [[C:%.*]] = ttg.convert_layout [[CST]] : tensor<128x128xf32, [[BLOCKED2]]> -> tensor<128x128xf32, [[DPAS]]>
    // CHECK: [[CVT_ARG0:%.*]] = ttg.convert_layout %arg0 : tensor<128x32xi8, [[BLOCKED]]> -> tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = [[DPAS1]], kWidth = 4}>>
    // CHECK: [[CVT_ARG1:%.*]] = ttg.convert_layout %arg1 : tensor<128x2xi8, [[BLOCKED1]]> -> tensor<128x2xi8, [[BLOCKED1]]>
    // CHECK: [[A:%.*]] = ttg.upcast_mxfp [[CVT_ARG0]], [[CVT_ARG1]] fp_type = e2m1 : tensor<128x32xi8, #ttg.dot_op<{opIdx = 0, parent = [[DPAS1]], kWidth = 4}>>, tensor<128x2xi8, [[BLOCKED1]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>>
    // CHECK: [[CVT_ARG2:%.*]] = ttg.convert_layout [[ARG2]] : tensor<64x128xf8E4M3FN, [[BLOCKED2]]> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>>
    // CHECK: [[B:%.*]] = tt.fp_to_fp [[CVT_ARG2]] : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>>
    // CHECK: [[D:%.*]] = tt.dot [[A]], [[B]], [[C]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>> -> tensor<128x128xf32, [[DPAS]]>
    // CHECK: [[RES:%.*]] = ttg.convert_layout {{.*}} : tensor<128x128xf32, [[DPAS]]> -> tensor<128x128xf32, [[BLOCKED2]]>

    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result = tt.dot_scaled %a scale %scale, %b, %cst lhs = e2m1 rhs = e4m3 : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf8E4M3FN, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 2], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {ttg.target = "xpu", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32} {
  // CHECK-DAG: [[BLOCKED:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: [[BLOCKED1:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: [[BLOCKED2:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: [[BLOCKED3:#.+]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-DAG: [[DPAS:#.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
  // CHECK-DAG: [[DPAS1:#.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>

  // CHECK: tt.func @dot_scale_transpose([[ARG0:%.*]]: tensor<128x64xf8E4M3FN, [[BLOCKED]]>, [[ARG1:%.*]]: tensor<32x32xi8, [[BLOCKED1]]>, [[ARG2:%.*]]: tensor<32x2xi8, [[BLOCKED2]]>, %arg3: tensor<128x32x!tt.ptr<bf16>, [[BLOCKED3]]>) {
  tt.func @dot_scale_transpose(%a: tensor<128x64xf8E4M3FN, #blocked>, %b: tensor<32x32xi8, #blocked1>, %scale: tensor<32x2xi8, #blocked2>, %d: tensor<128x32x!tt.ptr<bf16>, #blocked3>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #blocked3>
    %cst_1 = arith.constant dense<2> : tensor<32x1xi32, #blocked2>
    // CHECK: scf.for {{.*}} iter_args([[ARG5:%.*]] = %cst)
    %0 = scf.for %arg4 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<128x32xf32, #blocked1>)  : i32 {
      // CHECK: [[C:%.*]] = ttg.convert_layout [[ARG5]] : tensor<128x32xf32, [[BLOCKED1]]> -> tensor<128x32xf32, [[DPAS]]>
      // CHECK: [[CVT_ARG1:%.*]] = ttg.convert_layout [[ARG1]] : tensor<32x32xi8, [[BLOCKED1]]> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = [[DPAS1]], kWidth = 4}>>
      // CHECK: [[CVT_ARG2:%.*]] = ttg.convert_layout [[ARG2]] : tensor<32x2xi8, [[BLOCKED2]]> -> tensor<32x2xi8, [[BLOCKED2]]>
      // CHECK: [[B:%.*]] = ttg.upcast_mxfp [[CVT_ARG1]], [[CVT_ARG2]] fp_type = e2m1 : tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = [[DPAS1]], kWidth = 4}>>, tensor<32x2xi8, [[BLOCKED2]]> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>>
      // CHECK: [[CVT_ARG0:%.*]] = ttg.convert_layout [[ARG0]] : tensor<128x64xf8E4M3FN, [[BLOCKED]]> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>>
      // CHECK: [[A:%.*]] = tt.fp_to_fp [[CVT_ARG0]] : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>>
      // CHECK: [[D:%.*]] = tt.dot [[A]], [[B]], [[C]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[DPAS]], kWidth = 2}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = [[DPAS]], kWidth = 2}>> -> tensor<128x32xf32, [[DPAS]]>
      // CHECK: [[RES:%.*]] = ttg.convert_layout [[D]] : tensor<128x32xf32, [[DPAS]]> -> tensor<128x32xf32, [[BLOCKED1]]>
      // CHECK: scf.yield [[RES]] : tensor<128x32xf32, [[BLOCKED1]]>
      %3 = tt.dot_scaled %a, %b scale %scale, %arg5 lhs = e4m3 rhs = e2m1 : tensor<128x64xf8E4M3FN, #blocked> * tensor<32x32xi8, #blocked1>, tensor<32x2xi8, #blocked2> -> tensor<128x32xf32, #blocked1>
      scf.yield %3 : tensor<128x32xf32, #blocked1>
    }
    %1 = arith.truncf %0 : tensor<128x32xf32, #blocked1> to tensor<128x32xbf16, #blocked1>
    %2 = ttg.convert_layout %1 : tensor<128x32xbf16, #blocked1> -> tensor<128x32xbf16, #blocked3>
    tt.store %d, %2 : tensor<128x32x!tt.ptr<bf16>, #blocked3>
    tt.return
  }
}
