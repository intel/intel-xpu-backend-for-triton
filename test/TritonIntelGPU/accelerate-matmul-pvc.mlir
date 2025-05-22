// RUN: env TRITON_INTEL_DECOMPOSE_SCALED_BLOCKED=1 triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

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
  // CHECK-DAG: [[BLOCKED3:#.+]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
  // CHECK-DAG: [[BLOCKED4:#.+]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
  // CHECK-DAG: [[LINEAR:#.+]] = #ttg.linear<{{.*}}>

  // CHECK: tt.func @dot_scaled([[ARG0:%.*]]: tensor<128x32xi8, [[BLOCKED]]>, [[ARG1:%.*]]: tensor<128x2xi8, [[BLOCKED1]]>, [[ARG2:%.*]]: tensor<64x128xbf16, [[BLOCKED2]]>) -> tensor<128x128xf32, [[BLOCKED2]]> {
  tt.func @dot_scaled(%a: tensor<128x32xi8, #blocked2>, %scale: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xbf16, #blocked>) -> tensor<128x128xf32, #blocked> {
    // CHECK: [[NAN:%.*]] = arith.constant dense<0x7FC0> : tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[C:%.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, [[BLOCKED2]]>
    // CHECK: [[FP4TOFP:%.*]] = ttg.fp4_to_fp [[ARG0]] {axis = 1 : i32} : tensor<128x32xi8, #blocked> -> tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[SCALE:%.*]] = ttg.convert_layout {{.*}} : tensor<128x64xbf16, [[LINEAR]]> -> tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[UPCAST:%.*]] = arith.mulf [[FP4TOFP]], [[SCALE]] : tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[MASKNAN:%.*]] = arith.select {{.*}}, [[NAN]], [[UPCAST]] : tensor<128x64xi1, [[BLOCKED3]]>, tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[CVT_ARG0:%.*]] = ttg.convert_layout [[MASKNAN]] : tensor<128x64xbf16, [[BLOCKED3]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>>
    // CHECK: [[CVT_ARG2:%.*]] = ttg.convert_layout [[ARG2]] : tensor<64x128xbf16, [[BLOCKED2]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>>
    // CHECK: [[A:%.*]] = tt.fp_to_fp [[CVT_ARG0]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>>
    // CHECK: [[B:%.*]] = tt.fp_to_fp [[CVT_ARG2]] : tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>>
    // CHECK: [[D:%.*]] = tt.dot [[A]], [[B]], [[C]] : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>> -> tensor<128x128xf32, [[BLOCKED2]]>
    // CHECK: [[RES:%.*]] = ttg.convert_layout [[D]] : tensor<128x128xf32, [[BLOCKED2]]> -> tensor<128x128xf32, [[BLOCKED2]]>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %dot_res1 = tt.dot_scaled %a scale %scale, %b, %cst lhs = e2m1 rhs = bf16 {fastMath = false} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xbf16, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %dot_res1 : tensor<128x128xf32, #blocked>
  }

  // CHECK: tt.func @dot_scaled_fp8([[ARG0:%.*]]: tensor<128x32xi8, [[BLOCKED]]>, [[ARG1:%.*]]: tensor<128x2xi8, [[BLOCKED1]]>, [[ARG2:%.*]]: tensor<64x128xf8E4M3FN, [[BLOCKED2]]>) -> tensor<128x128xf32, [[BLOCKED2]]> {
  tt.func @dot_scaled_fp8(%a: tensor<128x32xi8, #blocked2>, %scale: tensor<128x2xi8, #blocked1>, %b: tensor<64x128xf8E4M3FN, #blocked>) -> tensor<128x128xf32, #blocked> {
    // CHECK: [[C:%.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32, [[BLOCKED2]]>
    // CHECK: [[FP4TOFP:%.*]] = ttg.fp4_to_fp [[ARG0]] {axis = 1 : i32} : tensor<128x32xi8, #blocked> -> tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[SCALE:%.*]] = ttg.convert_layout {{.*}} : tensor<128x64xbf16, [[LINEAR]]> -> tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[UPCAST:%.*]] = arith.mulf [[FP4TOFP]], [[SCALE]] : tensor<128x64xbf16, [[BLOCKED3]]>
    // CHECK: [[CVT_ARG0:%.*]] = ttg.convert_layout [[UPCAST]] : tensor<128x64xbf16, [[BLOCKED3]]> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>>
    // CHECK: [[FPTOFP:%.*]] = tt.fp_to_fp [[ARG2]] : tensor<64x128xf8E4M3FN, [[BLOCKED2]]> -> tensor<64x128xbf16, [[BLOCKED2]]>
    // CHECK: [[CVT_ARG2:%.*]] = ttg.convert_layout [[FPTOFP]] : tensor<64x128xbf16, [[BLOCKED2]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>>
    // CHECK: [[A:%.*]] = tt.fp_to_fp [[CVT_ARG0]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>>
    // CHECK: [[B:%.*]] = tt.fp_to_fp [[CVT_ARG2:%.*]] : tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>>
    // CHECK: [[D:%.*]] = tt.dot [[A]], [[B]], [[C]] : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED2]]}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED2]]}>> -> tensor<128x128xf32, [[BLOCKED2]]>
    // CHECK: [[RES:%.*]] = ttg.convert_layout [[D]] : tensor<128x128xf32, [[BLOCKED2]]> -> tensor<128x128xf32, [[BLOCKED2]]>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result = tt.dot_scaled %a scale %scale, %b, %cst lhs = e2m1 rhs = e4m3 {fastMath = true} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<64x128xf8E4M3FN, #blocked> -> tensor<128x128xf32, #blocked>
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
  // CHECK-DAG: [[BLOCKED4:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK-DAG: [[BLOCKED5:#.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
  // CHECK-DAG: [[BLOCKED6:#.+]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
  // CHECK-DAG: [[BLOCKED7:#.+]] = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
  // CHECK: [[LINEAR:#.*]] = #ttg.linear<{{.*}}>
  // CHECK-NEXT: [[LINEAR1:#.+]] = #ttg.linear<{{.*}}>

  // CHECK: tt.func @dot_scale_transpose([[ARG0:%.*]]: tensor<128x64xf8E4M3FN, [[BLOCKED]]>, [[ARG1:%.*]]: tensor<32x32xi8, [[BLOCKED1]]>, [[ARG2:%.*]]: tensor<32x2xi8, [[BLOCKED2]]>, %arg3: tensor<128x32x!tt.ptr<bf16>, [[BLOCKED3]]>) {
  tt.func @dot_scale_transpose(%a: tensor<128x64xf8E4M3FN, #blocked>, %b: tensor<32x32xi8, #blocked1>, %scale: tensor<32x2xi8, #blocked2>, %d: tensor<128x32x!tt.ptr<bf16>, #blocked3>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #blocked3>
    %cst_1 = arith.constant dense<2> : tensor<32x1xi32, #blocked2>
    // CHECK: [[NAN:%.*]] = arith.constant dense<0x7FC0> : tensor<32x64xbf16, [[LINEAR]]>
    // CHECK: [[CST:%.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32, [[BLOCKED4]]>
    // CHECK: [[DOT_RES:%.*]] = scf.for {{.*}} iter_args([[ARG5:%.*]] = [[CST]]) -> (tensor<32x128xf32, [[BLOCKED4]]>)  : i32 {
    %0 = scf.for %arg4 = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<128x32xf32, #blocked1>)  : i32 {
      // CHECK: [[TRANS_A:%.*]] = tt.trans [[ARG0]] {{.*}} : tensor<128x64xf8E4M3FN, [[BLOCKED]]> -> tensor<64x128xf8E4M3FN, [[BLOCKED5]]>
      // CHECK: [[TRANS_B:%.*]] = tt.trans [[ARG1]] {{.*}} : tensor<32x32xi8, [[BLOCKED1]]> -> tensor<32x32xi8, [[BLOCKED4]]>
      // CHECK: [[FP4TOFP:%.*]] = ttg.fp4_to_fp [[TRANS_B]] {axis = 1 : i32} : tensor<32x32xi8, [[BLOCKED4]]> -> tensor<32x64xbf16, [[LINEAR]]>
      // CHECK: [[SCALE:%.*]] = ttg.convert_layout {{.*}} : tensor<32x64xbf16, [[LINEAR1]]> -> tensor<32x64xbf16, [[LINEAR]]>
      // CHECK: [[UPCAST:%.*]] = arith.mulf [[FP4TOFP]], [[SCALE]] : tensor<32x64xbf16, [[LINEAR]]>
      // CHECK: [[MASKNAN:%.*]] = arith.select {{.*}}, [[NAN]], [[UPCAST]] : tensor<32x64xi1, [[LINEAR]]>, tensor<32x64xbf16, [[LINEAR]]>
      // CHECK: [[CVT_ARG1:%.*]] = ttg.convert_layout [[MASKNAN]] : tensor<32x64xbf16, [[LINEAR]]> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED4]]}>>
      // CHECK: [[FPTOFP:%.*]] = tt.fp_to_fp [[TRANS_A]] : tensor<64x128xf8E4M3FN, [[BLOCKED5]]> -> tensor<64x128xbf16, [[BLOCKED5]]>
      // CHECK: [[CVT_ARG0:%.*]] = ttg.convert_layout [[FPTOFP]] : tensor<64x128xbf16, [[BLOCKED5]]> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED4]]}>>
      // CHECK: [[A:%.*]] = tt.fp_to_fp [[CVT_ARG1]] : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED4]]}>> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED4]]}>>
      // CHECK: [[B:%.*]] = tt.fp_to_fp [[CVT_ARG0]] : tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED4]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED4]]}>>
      // CHECK: [[D:%.*]] = tt.dot [[A]], [[B]], [[ARG5]] : tensor<32x64xf32, #ttg.dot_op<{opIdx = 0, parent = [[BLOCKED4]]}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = [[BLOCKED4]]}>> -> tensor<32x128xf32, [[BLOCKED4]]>
      // CHECK: [[RES:%.*]] = ttg.convert_layout [[D]] : tensor<32x128xf32, [[BLOCKED4]]> -> tensor<32x128xf32, [[BLOCKED4]]>
      // CHECK: scf.yield [[RES]] : tensor<32x128xf32, [[BLOCKED4]]>
      %3 = tt.dot_scaled %a, %b scale %scale, %arg5 lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<128x64xf8E4M3FN, #blocked> * tensor<32x32xi8, #blocked1>, tensor<32x2xi8, #blocked2> -> tensor<128x32xf32, #blocked1>
      scf.yield %3 : tensor<128x32xf32, #blocked1>
    }
    // CHECK: [[TRUNC:%.*]] = arith.truncf [[DOT_RES]] : tensor<32x128xf32, [[BLOCKED4]]> to tensor<32x128xbf16, [[BLOCKED4]]>
    // CHECK: [[CVT:%.*]] = ttg.convert_layout [[TRUNC]] : tensor<32x128xbf16, [[BLOCKED4]]> -> tensor<32x128xbf16, [[BLOCKED7]]>
    // CHECK: [[TRANS:%.*]] = tt.trans [[CVT]] {{.*}} : tensor<32x128xbf16, [[BLOCKED7]]> -> tensor<128x32xbf16, [[BLOCKED3]]>
    %1 = arith.truncf %0 : tensor<128x32xf32, #blocked1> to tensor<128x32xbf16, #blocked1>
    %2 = ttg.convert_layout %1 : tensor<128x32xbf16, #blocked1> -> tensor<128x32xbf16, #blocked3>
    tt.store %d, %2 : tensor<128x32x!tt.ptr<bf16>, #blocked3>
    tt.return
  }
}

// -----

// CHECK-NOT: triton_intel_gpu.dpas
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "triton_intel_gpu.min_sg_size" = 16 : i32, "triton_intel_gpu.support_dpas"} {
  // CHECK-LABEL: check_dpas_cap
  tt.func @check_dpas_cap(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked>
    %a = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

    %result = tt.dot %a, %b, %zero_f32, inputPrecision = tf32 : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf32, #blocked>
    %result_ptr = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.store %result_ptr, %result : tensor<128x16x!tt.ptr<f32>, #blocked>

    %result2 = tt.dot %a, %b, %zero_f32 : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x16xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf32, #blocked>
    tt.store %result_ptr, %result2 : tensor<128x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
